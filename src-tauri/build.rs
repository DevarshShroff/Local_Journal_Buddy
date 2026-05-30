use std::path::Path;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let python_bin = Path::new(&manifest_dir).join("python").join("bin").join("python3");

    if !python_bin.exists() {
        println!(
            "cargo:warning=src-tauri/python/ is missing. Run ./bundle_python.sh from the repo root before `npm run build`, or the release .app will not include OCR/RAG (~1.3GB bundle)."
        );
    }

    tauri_build::build()
}
