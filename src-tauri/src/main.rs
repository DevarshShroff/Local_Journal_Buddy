#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // This calls the 'run' function inside src/lib.rs
    journal_buddy_lib::run();
}