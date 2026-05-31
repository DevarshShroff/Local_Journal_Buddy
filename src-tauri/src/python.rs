// src-tauri/src/python.rs
// Resolves the bundled Python interpreter and runs scripts as subprocesses
// Handles both dev mode (venv at src-tauri/python) and production (Resources/python)

use std::path::{Path, PathBuf};
use std::process::Stdio;
use tauri::{AppHandle, Emitter, Manager};

use crate::ollama_managed;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

pub struct PythonResolver {
    python_bin: PathBuf,
    scripts_dir: PathBuf,
    model_cache_dir: PathBuf,
}

fn resolve_resource_dir(app: &AppHandle) -> PathBuf {
    app.path()
        .resource_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
}

fn resolve_python_bin(resource_dir: &Path) -> PathBuf {
    let bundled = resource_dir.join("python").join("bin").join("python3");
    if bundled.exists() {
        return bundled;
    }

    let dev_venv = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join("bin")
        .join("python3");
    if dev_venv.exists() {
        return dev_venv;
    }

    for candidate in [
        PathBuf::from("/opt/homebrew/bin/python3"),
        PathBuf::from("/usr/local/bin/python3"),
        PathBuf::from("/usr/bin/python3"),
    ] {
        if candidate.exists() {
            return candidate;
        }
    }

    PathBuf::from("python3")
}

fn resolve_scripts_dir(resource_dir: &Path) -> PathBuf {
    let from_scripts = resource_dir.join("python_scripts");
    if from_scripts.join("librarian.py").exists() {
        return from_scripts;
    }

    // bundle_python.sh also copies scripts into the venv root
    let from_python = resource_dir.join("python");
    if from_python.join("librarian.py").exists() {
        return from_python;
    }

    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python_scripts")
}

fn resolve_model_cache_dir(resource_dir: &Path) -> PathBuf {
    let bundled = resource_dir.join("python").join("models");
    if bundled.exists() {
        return bundled;
    }

    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join("models")
}

/// Get the resolver — call this at the start of every command
pub fn resolver(app: &AppHandle) -> PythonResolver {
    let resource_dir = resolve_resource_dir(app);
    PythonResolver {
        python_bin: resolve_python_bin(&resource_dir),
        scripts_dir: resolve_scripts_dir(&resource_dir),
        model_cache_dir: resolve_model_cache_dir(&resource_dir),
    }
}

impl PythonResolver {
    pub fn python_bin(&self) -> &PathBuf {
        &self.python_bin
    }

    fn script_path(&self, script: &str) -> PathBuf {
        self.scripts_dir.join(script)
    }

    fn base_command(&self, script: &str) -> Command {
        let script_path = self.script_path(script);
        let cache = self.model_cache_dir.to_string_lossy().to_string();

        let mut cmd = Command::new(&self.python_bin);
        cmd.arg(&script_path)
            .env("SENTENCE_TRANSFORMERS_HOME", &cache)
            .env("HF_HOME", &cache)
            .env("PYTHONUNBUFFERED", "1")
            .env("OLLAMA_URL", ollama_managed::effective_ollama_base())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        cmd
    }

    pub async fn run_text(&self, script: &str, args: &[&str], _stdin: Option<&str>) -> Result<String, String> {
        let mut cmd = self.base_command(script);
        cmd.args(args);

        let output = cmd
            .output()
            .await
            .map_err(|e| format!("Failed to spawn {script}: {e}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("{script} exited with error:\n{stderr}"));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    pub async fn run_json(&self, script: &str, args: &[&str], stdin: Option<&str>) -> Result<String, String> {
        let text = self.run_text(script, args, stdin).await?;

        serde_json::from_str::<serde_json::Value>(&text)
            .map_err(|e| format!("Invalid JSON from {script}: {e}\nOutput: {text}"))?;

        Ok(text)
    }

    pub async fn run_streaming(
        &self,
        script: &str,
        args: &[&str],
        app: AppHandle,
        token_event: &str,
        done_event: &str,
    ) -> Result<(), String> {
        let mut cmd = self.base_command(script);
        cmd.args(args);

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("Failed to spawn {script}: {e}"))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Could not capture stdout".to_string())?;

        let mut reader = BufReader::new(stdout).lines();
        let token_event = token_event.to_string();
        let done_event = done_event.to_string();

        while let Some(line) = reader
            .next_line()
            .await
            .map_err(|e| format!("Read error: {e}"))?
        {
            if !line.is_empty() {
                app.emit(&token_event, &line).ok();
            }
        }

        child
            .wait()
            .await
            .map_err(|e| format!("Process wait error: {e}"))?;

        app.emit(&done_event, ()).ok();
        Ok(())
    }
}
