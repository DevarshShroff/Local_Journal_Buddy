// src-tauri/src/lib.rs

pub mod python;
pub mod health;
pub mod ollama_managed;

use tauri::{Emitter, AppHandle, RunEvent};
use serde::{Deserialize, Serialize};
use std::path::Path;

// ── Data types (Made pub so health.rs can see them) ──────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IngestReport {
    pub source_path: String,
    pub date: String,
    pub chunks_stored: i32,
    pub skipped: i32,
    pub errors: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct JournalEntry {
    /// SQLite primary key — use for read/delete (avoids string mismatch through the webview).
    #[serde(default)]
    pub id: Option<i64>,
    pub source_path: String,
    pub date: String,
    pub ingested_at: String,
    pub total_chunks: i32,
    #[serde(default)]
    pub preview: Option<String>,
    #[serde(default)]
    pub word_count: Option<i32>,
    #[serde(default)]
    pub mood: Option<String>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BrainResponse {
    pub question: String,
    pub answer: String,
    pub chunks_used: i32,
    pub fallback_used: bool,
    pub context_dates: Vec<String>,
    pub errors: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct JournalEntryText {
    pub ok: bool,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub date: String,
    #[serde(default)]
    pub source_path: String,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub word_count: Option<i32>,
    #[serde(default)]
    pub errors: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DeleteEntryResult {
    pub ok: bool,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub deleted_id: Option<i64>,
    #[serde(default)]
    pub errors: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HealthStatus {
    pub python_ok: bool,
    pub ollama_ok: bool,
    pub model_ok: bool,
    pub db_ok: bool,
    pub db_entry_count: i32,
    pub python_version: String,
    pub model_name: String,
    pub errors: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OcrResult {
    pub source_path: String,
    pub text: String,
    pub timestamp: String,
    pub word_count: i32,
    pub confidence_avg: f32,
    pub errors: Vec<String>,
}

// Library read/delete use stable SQLite `entry_id` (i64) from the list payload — avoids
// mismatched date/source strings through the webview. From the webview, pass Tauri v2 camelCase keys
// (e.g. `imagePath`, `sourcePath`, `topK`, `folderPath`).

// ── Commands ──────────────────────────────────────────────────────────────────

#[tauri::command]
async fn health_check(app: AppHandle) -> Result<HealthStatus, String> {
    health::check_all(&app).await
}

#[tauri::command]
async fn ingest_text(app: AppHandle, text: String, date: String, _source: Option<String>) -> Result<IngestReport, String> {
    let py = python::resolver(&app);
    // Unique source_path is generated in librarian.py (typed_{date}_{id}); optional --source reserved for future hints.

    let output = py.run_json("librarian.py", &["--ingest-text", &text, "--date", &date, "--json"], None).await?;
    serde_json::from_str(&output).map_err(|e| format!("Parse error: {e}"))
}


#[tauri::command]
async fn ocr_image(app: AppHandle, image_path: String) -> Result<OcrResult, String> {
    let py = python::resolver(&app);
    app.emit("ocr-progress", serde_json::json!({"step": 1, "label": "Running OCR...", "pct": 15})).ok();
    let ocr_output = py.run_json("ocr_engine.py", &["--image", &image_path, "--no-save", "--json"], None).await?;
    let mut ocr: OcrResult = serde_json::from_str(&ocr_output).map_err(|e| format!("OCR error: {e}"))?;
    
    app.emit("ocr-progress", serde_json::json!({"step": 2, "label": "Correcting text...", "pct": 45})).ok();
    let corrected = py.run_json("ocr_corrector.py", &["--text", &ocr.text, "--json"], None).await?;
    #[derive(Deserialize)] struct Corr { corrected_text: String }
    if let Ok(cr) = serde_json::from_str::<Corr>(&corrected) { ocr.text = cr.corrected_text; }
    
    app.emit("ocr-progress", serde_json::json!({"step": 3, "label": "Ready", "pct": 80})).ok();
    Ok(ocr)
}

#[tauri::command]
async fn ingest_ocr_result(app: AppHandle, text: String, date: String, source_path: String) -> Result<IngestReport, String> {
    let py = python::resolver(&app);
    let output = py
        .run_json(
            "librarian.py",
            &["--ingest-text", &text, "--date", &date, "--source", &source_path, "--json"],
            None,
        )
        .await?;
    Ok(serde_json::from_str(&output).map_err(|e| e.to_string())?)
}

#[tauri::command]
async fn get_all_entries(app: AppHandle) -> Result<Vec<JournalEntry>, String> {
    let py = python::resolver(&app);
    let output = py.run_json("librarian.py", &["--list-entries", "--json"], None).await?;
    serde_json::from_str(&output).map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_journal_entry_text(app: AppHandle, entry_id: i64) -> Result<JournalEntryText, String> {
    let py = python::resolver(&app);
    let id = entry_id.to_string();
    let output = py
        .run_json(
            "librarian.py",
            &["--read-entry", "--entry-id", &id, "--json"],
            None,
        )
        .await?;
    serde_json::from_str(&output).map_err(|e| e.to_string())
}

#[tauri::command]
async fn delete_journal_entry(app: AppHandle, entry_id: i64) -> Result<DeleteEntryResult, String> {
    let py = python::resolver(&app);
    let id = entry_id.to_string();
    let output = py
        .run_json(
            "librarian.py",
            &["--delete-entry", "--entry-id", &id, "--json"],
            None,
        )
        .await?;
    serde_json::from_str(&output).map_err(|e| e.to_string())
}

#[tauri::command]
async fn ask_brain(app: AppHandle, question: String, top_k: Option<i32>) -> Result<BrainResponse, String> {
    let py = python::resolver(&app);
    let k = top_k.unwrap_or(4).to_string();
    let ob = ollama_managed::effective_ollama_base();
    let output = py
        .run_json(
            "brain.py",
            &["--ask", &question, "--top-k", &k, "--ollama-url", &ob, "--json"],
            None,
        )
        .await?;
    serde_json::from_str(&output).map_err(|e| e.to_string())
}

#[tauri::command]
async fn ask_brain_stream(app: AppHandle, question: String, top_k: Option<i32>) -> Result<(), String> {
    let resp = ask_brain(app.clone(), question, top_k).await?;

    // Minimal "streaming" to satisfy the UI: emit small chunks of the final answer.
    // (If you later switch `brain.py` to true token streaming, swap this for `python::run_streaming`.)
    let answer = resp.answer;
    for chunk in answer.as_bytes().chunks(24) {
        if let Ok(s) = std::str::from_utf8(chunk) {
            app.emit("brain-token", s).ok();
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    app.emit("brain-done", ()).ok();
    Ok(())
}

#[derive(Serialize, Clone)]
struct BatchProgressPayload {
    filename: String,
    current: i32,
    total: i32,
    pct: i32,
    done: bool,
}

#[tauri::command]
async fn pick_batch_folder() -> Option<String> {
    tokio::task::spawn_blocking(|| {
        rfd::FileDialog::new()
            .set_title("Choose a folder of journal photos")
            .pick_folder()
            .map(|p| p.to_string_lossy().to_string())
    })
    .await
    .ok()
    .flatten()
}

#[tauri::command]
async fn batch_ingest_folder(
    app: AppHandle,
    folder_path: String,
    date_mode: Option<String>,      // "file" (default) | "default"
    default_date: Option<String>,   // YYYY-MM-DD used when date_mode=default or when file date missing
) -> Result<(), String> {
    let mode = date_mode.unwrap_or_else(|| "file".to_string());
    let fallback_date = default_date.unwrap_or_else(|| chrono::Local::now().format("%Y-%m-%d").to_string());

    let fp = Path::new(&folder_path);
    if !fp.is_dir() {
        app
            .emit(
                "batch-progress",
                BatchProgressPayload {
                    filename: "That folder path is not valid — use “Choose folder…”".to_string(),
                    current: 0,
                    total: 0,
                    pct: 0,
                    done: true,
                },
            )
            .ok();
        return Ok(());
    }

    let py = python::resolver(&app);

    let mut files: Vec<String> = Vec::new();
    for entry in walkdir::WalkDir::new(&folder_path).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        let p = entry.path();
        let ext = p
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if !matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "heic") {
            continue;
        }
        files.push(p.to_string_lossy().to_string());
    }
    files.sort();

    let total = files.len() as i32;
    if total == 0 {
        let payload = BatchProgressPayload {
            filename: "No .jpg/.png/.heic files found in that folder".to_string(),
            current: 0,
            total: 0,
            pct: 0,
            done: true,
        };
        app.emit("batch-progress", payload).ok();
        return Ok(());
    }

    app.emit(
        "batch-progress",
        BatchProgressPayload {
            filename: format!("Found {total} image(s) — starting OCR…"),
            current: 0,
            total,
            pct: 0,
            done: false,
        },
    )
    .ok();

    for (i, path_str) in files.iter().enumerate() {
        let current = (i as i32) + 1;
        let pct = ((current as f32 / total as f32) * 100.0).round().clamp(0.0, 100.0) as i32;
        let filename = Path::new(path_str)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(path_str)
            .to_string();

        app.emit(
            "batch-progress",
            BatchProgressPayload {
                filename: filename.clone(),
                current,
                total,
                pct,
                done: false,
            },
        )
        .ok();

        // Decide date
        let date = if mode == "default" {
            fallback_date.clone()
        } else {
            // mode == "file": use mtime local date, else fallback
            match std::fs::metadata(path_str)
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| chrono::DateTime::<chrono::Local>::from(t).date_naive().format("%Y-%m-%d").to_string().into())
            {
                Some(d) => d,
                None => fallback_date.clone(),
            }
        };

        // OCR
        let ocr_output = py
            .run_json("ocr_engine.py", &["--image", path_str, "--no-save", "--json"], None)
            .await;
        let ocr = match ocr_output {
            Ok(s) => s,
            Err(e) => {
                app.emit(
                    "batch-progress",
                    BatchProgressPayload {
                        filename: format!("{filename} (OCR failed)"),
                        current,
                        total,
                        pct,
                        done: false,
                    },
                )
                .ok();
                eprintln!("batch OCR failed for {path_str}: {e}");
                continue;
            }
        };
        #[derive(Deserialize)]
        struct OcrJson {
            text: String,
        }
        let ocr_text = serde_json::from_str::<OcrJson>(&ocr).map(|o| o.text).unwrap_or_default();

        // Correct
        let corrected_out = py.run_json("ocr_corrector.py", &["--text", &ocr_text, "--json"], None).await;
        #[derive(Deserialize)]
        struct Corr {
            corrected_text: String,
        }
        let corrected_text = corrected_out
            .ok()
            .and_then(|s| serde_json::from_str::<Corr>(&s).ok())
            .map(|c| c.corrected_text)
            .unwrap_or(ocr_text);

        // Ingest
        let _ = py
            .run_json(
                "librarian.py",
                &[
                    "--ingest-text",
                    &corrected_text,
                    "--date",
                    &date,
                    "--source",
                    path_str,
                    "--json",
                ],
                None,
            )
            .await;
    }

    app.emit(
        "batch-progress",
        BatchProgressPayload {
            filename: "Done".to_string(),
            current: total,
            total,
            pct: 100,
            done: true,
        },
    )
    .ok();
    Ok(())
}


// ── Entry Point for Desktop/Mobile ───────────────────────────────────────────

#[cfg_attr(all(not(debug_assertions), target_os = "ios"), tauri::mobile_entry_point)]
pub fn run() {
    let context = tauri::generate_context!();
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                ollama_managed::initialize(&handle).await;
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            health_check,
            ingest_text,
            ocr_image,
            ingest_ocr_result,
            get_all_entries,
            get_journal_entry_text,
            delete_journal_entry,
            ask_brain,
            ask_brain_stream,
            pick_batch_folder,
            batch_ingest_folder,
        ])
        .build(context)
        .expect("error building Journal Buddy");
    app.run(|_app_handle, event| {
        if matches!(event, RunEvent::Exit) {
            ollama_managed::shutdown_managed();
        }
    });
}