//! Bundled Ollama: download on first run (macOS / Windows), run `ollama serve` on a dedicated port
//! so we do not interfere with a user-owned daemon on :11434. On quit we only kill the child we started.

use flate2::read::GzDecoder;
use futures_util::StreamExt;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager};
use tokio::process::{Child, Command};
use walkdir::WalkDir;

pub const MANAGED_PORT: u16 = 11437;

const OLLAMA_RELEASE_TAG: &str = "v0.21.0";
const DEFAULT_MODEL: &str = "llama3:8b";

static RESOLVED_URL: OnceLock<String> = OnceLock::new();
static MANAGED_CHILD: Mutex<Option<Child>> = Mutex::new(None);
static WE_STARTED_OLLAMA: AtomicBool = AtomicBool::new(false);

pub fn use_managed_ollama() -> bool {
    if std::env::var("JOURNAL_BUDDY_USE_SYSTEM_OLLAMA")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        return false;
    }
    if std::env::var("JOURNAL_BUDDY_MANAGED_OLLAMA")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        return true;
    }
    !cfg!(debug_assertions)
}

pub fn effective_ollama_base() -> String {
    RESOLVED_URL
        .get()
        .cloned()
        .unwrap_or_else(|| "http://127.0.0.1:11434".to_string())
}

fn bundle_dir(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(app
        .path()
        .app_local_data_dir()
        .map_err(|e| e.to_string())?
        .join("journal-buddy-ollama"))
}

fn ollama_binary_path(dir: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        dir.join("ollama.exe")
    } else {
        dir.join("ollama")
    }
}

fn download_url_mac() -> String {
    format!(
        "https://github.com/ollama/ollama/releases/download/{OLLAMA_RELEASE_TAG}/ollama-darwin.tgz"
    )
}

fn download_url_win() -> String {
    format!(
        "https://github.com/ollama/ollama/releases/download/{OLLAMA_RELEASE_TAG}/ollama-windows-amd64.zip"
    )
}

#[derive(serde::Serialize, Clone)]
struct OllamaSetupPayload {
    stage: String,
    message: String,
    pct: i32,
    done: bool,
}

fn emit_setup(app: &AppHandle, stage: &str, message: String, pct: i32, done: bool) {
    let payload = OllamaSetupPayload {
        stage: stage.to_string(),
        message,
        pct: pct.clamp(0, 100),
        done,
    };
    app.emit("ollama-setup", payload).ok();
}

async fn http_download_with_progress(app: &AppHandle, url: &str) -> Result<Vec<u8>, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client.get(url).send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("download HTTP {} for {}", resp.status(), url));
    }
    let total = resp.content_length();
    let mut downloaded: u64 = 0;
    let mut out: Vec<u8> = Vec::with_capacity(total.unwrap_or(0) as usize);

    let mut stream = resp.bytes_stream();
    let mut last_pct: i32 = -1;
    while let Some(item) = stream.next().await {
        let chunk = item.map_err(|e| e.to_string())?;
        downloaded += chunk.len() as u64;
        out.extend_from_slice(&chunk);
        if let Some(t) = total {
            let pct = ((downloaded as f64 / t as f64) * 100.0).round() as i32;
            if pct != last_pct && (pct % 2 == 0 || pct == 100) {
                last_pct = pct;
                emit_setup(
                    app,
                    "downloading",
                    format!("Setting up local AI engine… downloading ({pct}%)"),
                    pct,
                    false,
                );
            }
        } else if downloaded % (8 * 1024 * 1024) < chunk.len() as u64 {
            emit_setup(
                app,
                "downloading",
                "Setting up local AI engine… downloading".to_string(),
                20,
                false,
            );
        }
    }
    Ok(out)
}

fn extract_tgz_find_ollama(bytes: &[u8], dest_bin: &Path) -> Result<(), String> {
    let cursor = Cursor::new(bytes);
    let dec = GzDecoder::new(cursor);
    let mut arch = tar::Archive::new(dec);
    let tmp = dest_bin
        .parent()
        .ok_or_else(|| "no parent".to_string())?
        .join("ollama_extract_tmp");
    if tmp.exists() {
        let _ = std::fs::remove_dir_all(&tmp);
    }
    std::fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;
    arch.unpack(&tmp).map_err(|e| format!("tar unpack: {e}"))?;

    let want = if cfg!(target_os = "windows") {
        "ollama.exe"
    } else {
        "ollama"
    };
    let mut found: Option<PathBuf> = None;
    for e in WalkDir::new(&tmp).into_iter().filter_map(|e| e.ok()) {
        if !e.file_type().is_file() {
            continue;
        }
        if e.path().file_name().and_then(|n| n.to_str()) == Some(want) {
            found = Some(e.path().to_path_buf());
            break;
        }
    }
    let src = found.ok_or_else(|| format!("no {want} inside archive"))?;
    if dest_bin.exists() {
        let _ = std::fs::remove_file(dest_bin);
    }
    std::fs::rename(&src, dest_bin).or_else(|_| {
        std::fs::copy(&src, dest_bin)?;
        Ok::<(), std::io::Error>(())
    })
    .map_err(|e| e.to_string())?;
    let _ = std::fs::remove_dir_all(&tmp);

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(dest_bin)
            .map_err(|e| e.to_string())?
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(dest_bin, perms).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn extract_zip_find_ollama(bytes: &[u8], dest_bin: &Path) -> Result<(), String> {
    let cursor = Cursor::new(bytes);
    let mut arch = zip::ZipArchive::new(cursor).map_err(|e| format!("zip: {e}"))?;
    let tmp = dest_bin
        .parent()
        .ok_or_else(|| "no parent".to_string())?
        .join("ollama_extract_tmp");
    if tmp.exists() {
        let _ = std::fs::remove_dir_all(&tmp);
    }
    std::fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;
    for i in 0..arch.len() {
        let mut file = arch.by_index(i).map_err(|e| e.to_string())?;
        let outpath = match file.enclosed_name() {
            Some(p) => tmp.join(p),
            None => continue,
        };
        if file.name().ends_with('/') {
            std::fs::create_dir_all(&outpath).map_err(|e| e.to_string())?;
        } else {
            if let Some(p) = outpath.parent() {
                std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
            }
            let mut out = std::fs::File::create(&outpath).map_err(|e| e.to_string())?;
            std::io::copy(&mut file, &mut out).map_err(|e| e.to_string())?;
        }
    }
    let want = "ollama.exe";
    let mut found: Option<PathBuf> = None;
    for e in WalkDir::new(&tmp).into_iter().filter_map(|e| e.ok()) {
        if !e.file_type().is_file() {
            continue;
        }
        if e.path().file_name().and_then(|n| n.to_str()) == Some(want) {
            found = Some(e.path().to_path_buf());
            break;
        }
    }
    let src = found.ok_or_else(|| "no ollama.exe inside zip".to_string())?;
    if dest_bin.exists() {
        let _ = std::fs::remove_file(dest_bin);
    }
    std::fs::rename(&src, dest_bin).or_else(|_| {
        std::fs::copy(&src, dest_bin)?;
        Ok::<(), std::io::Error>(())
    })
    .map_err(|e| e.to_string())?;
    let _ = std::fs::remove_dir_all(&tmp);
    Ok(())
}

async fn ping_ollama(base: &str) -> bool {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_millis(600))
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    let url = format!("{}/api/tags", base.trim_end_matches('/'));
    client
        .get(url)
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

#[cfg(target_os = "linux")]
fn find_system_ollama_linux() -> Option<PathBuf> {
    for p in ["/usr/local/bin/ollama", "/usr/bin/ollama"] {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return Some(pb);
        }
    }
    None
}

async fn has_model(base: &str, model: &str) -> Result<bool, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(1200))
        .build()
        .map_err(|e| e.to_string())?;
    let url = format!("{}/api/tags", base.trim_end_matches('/'));
    let resp = client.get(url).send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("tags HTTP {}", resp.status()));
    }
    let body = resp.text().await.map_err(|e| e.to_string())?;
    let json = serde_json::from_str::<serde_json::Value>(&body).map_err(|e| e.to_string())?;
    let models = json["models"].as_array().cloned().unwrap_or_default();
    let found = models.iter().any(|m| {
        m["name"]
            .as_str()
            .map(|n| n == model || n.starts_with(&(model.to_string() + ":")) || n.contains("llama3")) // tolerate minor variants
            .unwrap_or(false)
    });
    Ok(found)
}

async fn pull_model(app: &AppHandle, bin: &Path, host: &str, models_dir: &Path, model: &str) -> Result<(), String> {
    emit_setup(
        app,
        "pulling",
        format!("Setting up local AI engine... installing {model} (this may take a few minutes)"),
        98,
        false,
    );

    let mut cmd = Command::new(bin);
    cmd.arg("pull")
        .arg(model)
        .env("OLLAMA_HOST", host)
        .env("OLLAMA_MODELS", models_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| format!("spawn ollama pull: {e}"))?;
    let start = std::time::Instant::now();

    // We don’t have a stable machine-readable progress format here; show periodic “still working”.
    loop {
        tokio::select! {
            res = child.wait() => {
                let st = res.map_err(|e| e.to_string())?;
                if st.success() { return Ok(()); }
                return Err(format!("ollama pull exited with {st}"));
            }
            _ = tokio::time::sleep(Duration::from_secs(2)) => {
                let secs = start.elapsed().as_secs();
                emit_setup(
                    app,
                    "pulling",
                    format!("Setting up local AI engine... installing {model} ({secs}s)"),
                    98,
                    false,
                );
            }
        }
    }
}

async fn spawn_ollama_process(bin: &Path, host: &str, models_dir: &Path) -> Result<Child, String> {
    let mut cmd = Command::new(bin);
    cmd.arg("serve")
        .env("OLLAMA_HOST", host)
        .env("OLLAMA_MODELS", models_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    cmd.spawn().map_err(|e| format!("spawn ollama: {e}"))
}

async fn wait_ready(base: &str) -> bool {
    for _ in 0..150 {
        if ping_ollama(base).await {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    false
}

async fn start_managed_on_port(app: &AppHandle, bin: &Path) -> bool {
    let dir = match bundle_dir(app) {
        Ok(d) => d,
        Err(_) => return false,
    };
    let models_dir = dir.join("models");
    let _ = std::fs::create_dir_all(&models_dir);
    let host = format!("127.0.0.1:{MANAGED_PORT}");
    let base = format!("http://{host}");

    if ping_ollama(&base).await {
        let _ = RESOLVED_URL.set(base.clone());
        return true;
    }

    match spawn_ollama_process(bin, &host, &models_dir).await {
        Ok(child) => {
            *MANAGED_CHILD.lock().expect("lock") = Some(child);
            WE_STARTED_OLLAMA.store(true, Ordering::SeqCst);
        }
        Err(e) => {
            eprintln!("journal-buddy: ollama serve spawn failed: {e}");
            return false;
        }
    }

    if wait_ready(&base).await {
        let _ = RESOLVED_URL.set(base.clone());

        // Ensure the default model exists so the onboarding health check can go green without CLI steps.
        match has_model(&base, DEFAULT_MODEL).await {
            Ok(true) => true,
            Ok(false) => {
                match pull_model(app, bin, &host, &models_dir, DEFAULT_MODEL).await {
                    Ok(()) => true,
                    Err(e) => {
                        eprintln!("journal-buddy: model pull failed: {e}");
                        emit_setup(
                            app,
                            "error",
                            format!("Could not install {DEFAULT_MODEL}: {e}"),
                            0,
                            true,
                        );
                        true // Ollama is up; model may be installed later.
                    }
                }
            }
            Err(e) => {
                eprintln!("journal-buddy: tag check failed: {e}");
                true
            }
        }
    } else {
        eprintln!("journal-buddy: managed Ollama did not become ready");
        shutdown_managed_internal();
        false
    }
}

pub async fn initialize(app: &AppHandle) {
    if !use_managed_ollama() {
        let _ = RESOLVED_URL.set("http://127.0.0.1:11434".to_string());
        return;
    }

    // Prefer the managed base immediately so all callers point at the same local endpoint.
    let managed_base = format!("http://127.0.0.1:{MANAGED_PORT}");
    let _ = RESOLVED_URL.set(managed_base.clone());

    let dir = match bundle_dir(app) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("journal-buddy: bundle dir: {e}");
            let _ = RESOLVED_URL.set("http://127.0.0.1:11434".to_string());
            return;
        }
    };
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("journal-buddy: ollama bundle mkdir: {e}");
        let _ = RESOLVED_URL.set("http://127.0.0.1:11434".to_string());
        return;
    }

    let bin = ollama_binary_path(&dir);

    #[cfg(target_os = "linux")]
    {
        if let Some(sys) = find_system_ollama_linux() {
            if start_managed_on_port(app, &sys).await {
                eprintln!("journal-buddy: managed Ollama (system binary) at {}", effective_ollama_base());
                return;
            }
        }
        eprintln!(
            "journal-buddy: Linux: install `ollama` from https://ollama.com or set JOURNAL_BUDDY_USE_SYSTEM_OLLAMA=1"
        );
        let _ = RESOLVED_URL.set("http://127.0.0.1:11434".to_string());
        return;
    }

    #[cfg(not(target_os = "linux"))]
    {
        emit_setup(
            app,
            "checking",
            "Setting up local AI engine… checking installation".to_string(),
            5,
            false,
        );
        if !bin.exists() {
            let url = if cfg!(target_os = "macos") {
                download_url_mac()
            } else {
                download_url_win()
            };
            eprintln!("journal-buddy: downloading Ollama ({OLLAMA_RELEASE_TAG}) — first launch only…");
            emit_setup(
                app,
                "downloading",
                "Setting up local AI engine… downloading (this may take a minute)".to_string(),
                10,
                false,
            );
            match http_download_with_progress(app, &url).await {
                Ok(bytes) => {
                    emit_setup(
                        app,
                        "extracting",
                        "Setting up local AI engine… unpacking".to_string(),
                        92,
                        false,
                    );
                    let res = if cfg!(target_os = "macos") {
                        extract_tgz_find_ollama(&bytes, &bin)
                    } else {
                        extract_zip_find_ollama(&bytes, &bin)
                    };
                    if let Err(e) = res {
                        eprintln!("journal-buddy: extract failed: {e}");
                        emit_setup(
                            app,
                            "error",
                            format!("Local AI engine setup failed: {e}"),
                            0,
                            true,
                        );
                        let _ = RESOLVED_URL.set("http://127.0.0.1:11434".to_string());
                        return;
                    }
                }
                Err(e) => {
                    eprintln!("journal-buddy: download failed: {e}");
                    emit_setup(
                        app,
                        "error",
                        format!("Local AI engine download failed: {e}"),
                        0,
                        true,
                    );
                    let _ = RESOLVED_URL.set("http://127.0.0.1:11434".to_string());
                    return;
                }
            }
        }

        emit_setup(
            app,
            "starting",
            "Setting up local AI engine… starting".to_string(),
            96,
            false,
        );
        if start_managed_on_port(app, &bin).await {
            eprintln!("journal-buddy: managed Ollama at {}", effective_ollama_base());
            emit_setup(app, "ready", "Local AI engine ready".to_string(), 100, true);
        } else {
            emit_setup(
                app,
                "error",
                "Local AI engine could not start. You can still use your own Ollama install.".to_string(),
                0,
                true,
            );
            let _ = RESOLVED_URL.set("http://127.0.0.1:11434".to_string());
        }
    }
}

fn shutdown_managed_internal() {
    if !WE_STARTED_OLLAMA.load(Ordering::SeqCst) {
        return;
    }
    if let Ok(mut g) = MANAGED_CHILD.lock() {
        if let Some(mut child) = g.take() {
            let _ = child.start_kill();
            let _ = std::thread::spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build();
                if let Ok(rt) = rt {
                    let _ = rt.block_on(child.wait());
                }
            });
        }
    }
    WE_STARTED_OLLAMA.store(false, Ordering::SeqCst);
}

pub fn shutdown_managed() {
    shutdown_managed_internal();
}
