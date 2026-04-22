// src-tauri/src/health.rs
// Checks Python, Ollama, model availability, and ChromaDB on first launch

use tauri::AppHandle;
use crate::{HealthStatus, ollama_managed, python};
use std::time::Duration;

pub async fn check_all(app: &AppHandle) -> Result<HealthStatus, String> {
    let py = python::resolver(app);
    let mut status = HealthStatus {
        python_ok: false,
        ollama_ok: false,
        model_ok: false,
        db_ok: false,
        db_entry_count: 0,
        python_version: String::new(),
        model_name: "llama3:8b".into(),
        errors: Vec::new(),
    };

    // ── Check Python ──────────────────────────────────────────────────────────
    let out = tokio::time::timeout(
        Duration::from_millis(1200),
        tokio::process::Command::new(py.python_bin()).arg("--version").output(),
    )
    .await;
    match out {
        Ok(Ok(o)) if o.status.success() => {
            status.python_ok = true;
            let stdout = String::from_utf8_lossy(&o.stdout);
            let stderr = String::from_utf8_lossy(&o.stderr);
            status.python_version = stdout.trim().to_string();
            if status.python_version.is_empty() {
                status.python_version = stderr.trim().to_string();
            }
        }
        Ok(Ok(o)) => status.errors.push(format!(
            "Python check failed (exit {}): {}",
            o.status,
            String::from_utf8_lossy(&o.stderr).trim()
        )),
        Ok(Err(e)) => status.errors.push(format!("Python check failed: {e}")),
        Err(_) => status.errors.push("Python check timed out".into()),
    };

    // ── Check Ollama reachability ─────────────────────────────────────────────
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(800))
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {e}"))?;

    let ollama_base = ollama_managed::effective_ollama_base();
    let tags_url = format!("{}/api/tags", ollama_base.trim_end_matches('/'));
    match client.get(tags_url).send().await {
        Ok(resp) if resp.status().is_success() => {
            status.ollama_ok = true;

            // Check model is pulled
            match resp.text().await {
                Ok(body) => {
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                        let models = json["models"].as_array();
                        let model_found = models
                            .map(|ms| {
                                ms.iter().any(|m| {
                                    m["name"]
                                        .as_str()
                                        .map(|n| n.contains("llama3:8b") || n == "llama3:8b")
                                        .unwrap_or(false)
                                })
                            })
                            .unwrap_or(false);

                        status.model_ok = model_found;
                        // If missing, the managed Ollama installer will attempt to pull it automatically on first run.
                    } else {
                        status.errors.push("Ollama responded but JSON was invalid".into());
                    }
                }
                Err(e) => status.errors.push(format!("Ollama response read failed: {e}")),
            }
        }
        Ok(resp) => {
            status.errors.push(format!("Ollama returned HTTP {}", resp.status()));
        }
        Err(_) => {
            // Important: never block onboarding on this.
            status.errors.push(format!(
                "Ollama not reachable at {}. If you use your own install, set JOURNAL_BUDDY_USE_SYSTEM_OLLAMA=1 and run `ollama serve`.",
                ollama_base
            ));
        }
    };

    // ── Check ChromaDB / journal DB ───────────────────────────────────────────
    match tokio::time::timeout(
        Duration::from_millis(2000),
        py.run_text("librarian.py", &["--count-entries", "--json"], None),
    )
    .await
    {
        Ok(Ok(output)) => {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&output) {
                status.db_ok = true;
                status.db_entry_count = json["entry_count"].as_i64().unwrap_or(0) as i32;
            } else {
                status.errors.push("DB check returned invalid JSON".into());
            }
        }
        Ok(Err(e)) => status.errors.push(format!("DB check failed: {e}")),
        Err(_) => status.errors.push("DB check timed out".into()),
    };

    Ok(status)
}
