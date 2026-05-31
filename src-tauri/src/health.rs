// src-tauri/src/health.rs
// Checks Python, Ollama, model availability, and ChromaDB on first launch

use crate::{ollama_managed, python, HealthStatus};
use std::time::Duration;
use tauri::AppHandle;

async fn check_python(py: &python::PythonResolver) -> (bool, String, Option<String>) {
    let out = tokio::time::timeout(
        Duration::from_millis(1200),
        tokio::process::Command::new(py.python_bin())
            .arg("--version")
            .output(),
    )
    .await;

    match out {
        Ok(Ok(o)) if o.status.success() => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            let stderr = String::from_utf8_lossy(&o.stderr);
            let mut version = stdout.trim().to_string();
            if version.is_empty() {
                version = stderr.trim().to_string();
            }
            (true, version, None)
        }
        Ok(Ok(o)) => (
            false,
            String::new(),
            Some(format!(
                "Python check failed (exit {}): {}",
                o.status,
                String::from_utf8_lossy(&o.stderr).trim()
            )),
        ),
        Ok(Err(e)) => (false, String::new(), Some(format!("Python check failed: {e}"))),
        Err(_) => (false, String::new(), Some("Python check timed out".into())),
    }
}

async fn check_ollama_and_model() -> (bool, bool, Vec<String>) {
    let mut errors = Vec::new();
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_millis(2500))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            errors.push(format!("Failed to build HTTP client: {e}"));
            return (false, false, errors);
        }
    };

    let mut bases = vec![ollama_managed::effective_ollama_base()];
    if ollama_managed::use_managed_ollama() {
        let managed = ollama_managed::managed_ollama_base();
        if !bases.iter().any(|b| b == &managed) {
            bases.push(managed);
        }
    }

    for base in bases {
        let tags_url = format!("{}/api/tags", base.trim_end_matches('/'));
        match client.get(&tags_url).send().await {
            Ok(resp) if resp.status().is_success() => {
                match resp.text().await {
                    Ok(body) => {
                        let model_found = ollama_managed::model_present_in_tags_json(&body);
                        return (true, model_found, errors);
                    }
                    Err(e) => errors.push(format!("Ollama response read failed ({base}): {e}")),
                }
            }
            Ok(resp) => errors.push(format!("Ollama returned HTTP {} at {base}", resp.status())),
            Err(_) => errors.push(format!("Ollama not reachable at {base}")),
        }
    }

    if ollama_managed::use_managed_ollama() {
        let hint = ollama_managed::last_setup_message();
        if !hint.is_empty() {
            errors.push(hint);
        }
    } else {
        errors.push(
            "Ollama not reachable. Set JOURNAL_BUDDY_USE_SYSTEM_OLLAMA=1 and run `ollama serve`, or use the bundled engine.".into(),
        );
    }

    (false, false, errors)
}

async fn check_db(_app: &AppHandle, py: &python::PythonResolver) -> (bool, i32, Option<String>) {
    match tokio::time::timeout(
        Duration::from_millis(2000),
        py.run_text("librarian.py", &["--count-entries", "--json"], None),
    )
    .await
    {
        Ok(Ok(output)) => {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&output) {
                let count = json["entry_count"].as_i64().unwrap_or(0) as i32;
                (true, count, None)
            } else {
                (false, 0, Some("DB check returned invalid JSON".into()))
            }
        }
        Ok(Err(e)) => (false, 0, Some(format!("DB check failed: {e}"))),
        Err(_) => (false, 0, Some("DB check timed out".into())),
    }
}

pub async fn check_all(app: &AppHandle) -> Result<HealthStatus, String> {
    let py = python::resolver(app);

    let py_fut = check_python(&py);
    let ollama_fut = check_ollama_and_model();
    let db_fut = check_db(app, &py);

    let ((python_ok, python_version, py_err), (ollama_ok, model_ok, mut ollama_errs), (db_ok, db_entry_count, db_err)) =
        tokio::join!(py_fut, ollama_fut, db_fut);

    let mut errors = Vec::new();
    if let Some(e) = py_err {
        errors.push(e);
    }
    errors.append(&mut ollama_errs);
    if let Some(e) = db_err {
        errors.push(e);
    }

    if ollama_ok && !model_ok && ollama_managed::use_managed_ollama() {
        let setup = ollama_managed::last_setup_message();
        if setup.is_empty() {
            errors.push(format!(
                "Downloading {} to {} — this can take several minutes on first launch.",
                "llama3:8b",
                ollama_managed::effective_ollama_base()
            ));
        }
        ollama_managed::schedule_ensure_default_model(app.clone());
    }

    Ok(HealthStatus {
        python_ok,
        ollama_ok,
        model_ok,
        db_ok,
        db_entry_count,
        python_version,
        model_name: "llama3:8b".into(),
        errors,
    })
}
