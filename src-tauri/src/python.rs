// src-tauri/src/python.rs
// Resolves the bundled Python interpreter and runs scripts as subprocesses
// Handles both dev mode (system python3) and production (bundled .app python)

use std::path::PathBuf;
use std::process::Stdio;
use tauri::{AppHandle, Emitter, Manager};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

pub struct PythonResolver {
    python_bin: PathBuf,
    scripts_dir: PathBuf,
}

/// Get the resolver — call this at the start of every command
pub fn resolver(app: &AppHandle) -> PythonResolver {
    let resource_dir = app
        .path()
        .resource_dir()
        .unwrap_or_else(|_| PathBuf::from("."));

    // In production: bundled Python inside .app/Contents/Resources/python/
    let bundled_python = resource_dir.join("python").join("bin").join("python3");

    // In dev: prefer the checked-in venv at src-tauri/python/bin/python3 so
    // dependencies (chromadb, sentence-transformers, etc) are available.
    let dev_venv_python = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join("bin")
        .join("python3");

    // Fallback: use system python3
    let python_bin = if bundled_python.exists() {
        bundled_python
    } else if dev_venv_python.exists() {
        dev_venv_python
    } else {
        // Try common Homebrew locations
        let homebrew_arm = PathBuf::from("/opt/homebrew/bin/python3");
        let homebrew_intel = PathBuf::from("/usr/local/bin/python3");
        let system = PathBuf::from("/usr/bin/python3");

        if homebrew_arm.exists() {
            homebrew_arm
        } else if homebrew_intel.exists() {
            homebrew_intel
        } else {
            system
        }
    };

    // Scripts:
    // - in production: bundled in Resources/python_scripts/
    // - in dev: use the checked-in scripts at src-tauri/python_scripts/
    let bundled_scripts = resource_dir.join("python_scripts");
    let scripts_dir = if bundled_scripts.exists() {
        bundled_scripts
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python_scripts")
    };

    PythonResolver { python_bin, scripts_dir }
}

impl PythonResolver {
    pub fn python_bin(&self) -> &PathBuf {
        &self.python_bin
    }

    /// Full path to a Python script in the resources/python/ directory
    fn script_path(&self, script: &str) -> PathBuf {
        self.scripts_dir.join(script)
    }

    /// Set up common env vars for all Python calls:
    /// - PYTHONUNBUFFERED ensures stdout isn't buffered
    fn base_command(&self, script: &str) -> Command {
        let script_path = self.script_path(script);

        let mut cmd = Command::new(&self.python_bin);
        cmd.arg(&script_path)
           // Make it easier to run fully offline with local caches.
           .env(
               "SENTENCE_TRANSFORMERS_HOME",
               PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                   .join("python")
                   .join("models")
                   .to_string_lossy()
                   .to_string(),
           )
           .env(
               "HF_HOME",
               PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                   .join("python")
                   .join("models")
                   .to_string_lossy()
                   .to_string(),
           )
           .env("PYTHONUNBUFFERED", "1")  // ensures stdout isn't buffered
           .stdout(Stdio::piped())
           .stderr(Stdio::piped());

        cmd
    }

    /// Run a script and return its full stdout as a String
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

    /// Run a script that outputs JSON — returns the raw JSON string
    pub async fn run_json(&self, script: &str, args: &[&str], stdin: Option<&str>) -> Result<String, String> {
        let text = self.run_text(script, args, stdin).await?;

        // Validate it's parseable JSON before returning
        serde_json::from_str::<serde_json::Value>(&text)
            .map_err(|e| format!("Invalid JSON from {script}: {e}\nOutput: {text}"))?;

        Ok(text)
    }

    /// Run a script in streaming mode — emits `event_name` for each token line
    /// and `done_event` when the process exits
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

        // Stream lines as they arrive
        while let Some(line) = reader.next_line().await
            .map_err(|e| format!("Read error: {e}"))? {
            if !line.is_empty() {
                app.emit(&token_event, &line).ok();
            }
        }

        // Wait for process to finish
        child.wait().await
            .map_err(|e| format!("Process wait error: {e}"))?;

        app.emit(&done_event, ()).ok();
        Ok(())
    }
}
