mod app;
mod buffer;
mod player;
mod playlist;
mod spectrum;

use std::io;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use crossterm::{
    cursor::{Hide, Show},
    event::{self, Event, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, Clear, ClearType},
    ExecutableCommand,
};
use rand::seq::SliceRandom;
use rand::thread_rng;

use app::App;
use playlist::Track;

/// A minimal terminal music player with real-time spectrum visualization
#[derive(Parser, Debug)]
#[command(name = "autotui", version, about, long_about = None)]
struct Args {
    /// Shuffle the playlist or folder of files
    #[arg(short, long)]
    shuffle: bool,

    /// Scan directories recursively
    #[arg(short, long)]
    recursive: bool,

    /// Write playlist to file and exit (utility mode)
    #[arg(short, long, value_name = "FILE")]
    write: Option<PathBuf>,

    /// Path to folder, playlist (.json), or audio file
    #[arg(value_name = "PATH")]
    path: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut app = App::new();
    app.recursive = args.recursive;

    // Load tracks based on arguments - infer type from path
    if let Some(path) = &args.path {
        // Normalize path for Windows compatibility
        let path_str = path.to_string_lossy();
        let path_str = path_str.trim_matches('"').trim_matches('\'');
        let path_str = path_str.trim_end_matches(|c| c == '/' || c == '\\');
        let normalized_path = PathBuf::from(path_str);
        let canonical_path = normalized_path
            .canonicalize()
            .unwrap_or(normalized_path.clone());

        if canonical_path.is_dir() {
            app.scan_directory(&canonical_path.to_string_lossy(), args.recursive);
        } else if canonical_path.is_file() {
            if let Some(ext) = canonical_path.extension() {
                let ext_lower = ext.to_string_lossy().to_lowercase();
                if ext_lower == "json" {
                    if let Err(e) = app.load_playlist(&canonical_path) {
                        eprintln!("Error loading playlist: {}", e);
                        return Err(e);
                    }
                } else if matches!(ext_lower.as_str(), "mp3" | "wav" | "flac" | "ogg") {
                    app.tracks.push(Track::from_path(canonical_path.clone()));
                } else {
                    eprintln!("Unsupported file type: {}", ext_lower);
                    return Ok(());
                }
            }
        } else {
            // Try as directory anyway (might work with network paths)
            app.scan_directory(path_str, args.recursive);
            if app.tracks.is_empty() {
                eprintln!("Path not found or no tracks: {}", path.display());
                return Ok(());
            }
        }
    } else {
        app.scan_directory(".", args.recursive);
    }

    // Shuffle tracks if requested
    if args.shuffle {
        app.shuffle = true;
        let mut rng = thread_rng();
        app.tracks.shuffle(&mut rng);
    }

    // Write mode: save playlist and exit
    if let Some(output_path) = &args.write {
        if app.tracks.is_empty() {
            eprintln!("No tracks found to save");
            return Ok(());
        }
        match app.save_playlist(output_path) {
            Ok(()) => {
                println!("Saved {} tracks to {}", app.tracks.len(), output_path.display());
                return Ok(());
            }
            Err(e) => {
                eprintln!("Error saving playlist: {}", e);
                return Err(e);
            }
        }
    }

    // Normal playback mode
    enable_raw_mode()?;
    io::stdout().execute(Hide)?;

    if !app.tracks.is_empty() {
        app.play_current();
    }

    while !app.should_quit {
        app.check_playback();
        app.update();
        app.render()?;

        if event::poll(Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    app.handle_key(key.code);
                }
            }
        }
    }

    app.stop();

    io::stdout().execute(crossterm::cursor::MoveToColumn(0))?;
    io::stdout().execute(Clear(ClearType::CurrentLine))?;
    io::stdout().execute(Show)?;
    disable_raw_mode()?;

    println!("Enjoy the silence!");

    Ok(())
}
