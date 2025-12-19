use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// A track with path and display name
pub struct Track {
    pub path: PathBuf,
    pub name: String,
}

impl Track {
    pub fn from_path(path: PathBuf) -> Self {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Unknown")
            .to_string();
        Self { path, name }
    }
}

/// JSON playlist format (for reading)
#[derive(Debug, Deserialize)]
pub struct Playlist {
    #[serde(default)]
    #[allow(dead_code)]
    pub name: Option<String>,
    pub tracks: Vec<PlaylistEntry>,
}

/// Simple playlist format (for writing)
#[derive(Debug, Serialize)]
pub struct PlaylistOutput {
    pub tracks: Vec<String>,
}

/// A playlist entry can be a simple path string or an object with path
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum PlaylistEntry {
    Simple(String),
    Detailed { path: String },
}

impl Playlist {
    pub fn load(path: &PathBuf) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let playlist: Playlist = serde_json::from_reader(reader)?;
        Ok(playlist)
    }
}

impl PlaylistOutput {
    pub fn save(&self, path: &PathBuf) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

