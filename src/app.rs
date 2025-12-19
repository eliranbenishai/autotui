use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use crossterm::{
    cursor::MoveToColumn,
    event::KeyCode,
    style::{Color, Print, ResetColor, SetForegroundColor},
    terminal::{Clear, ClearType},
    ExecutableCommand,
};
use rand::seq::SliceRandom;
use rand::thread_rng;
use walkdir::WalkDir;

use crate::player::{Player, STATE_BUFFERING, STATE_ERROR, STATE_PAUSED, STATE_PLAYING, STATE_STOPPED};
use crate::playlist::{Playlist, PlaylistEntry, PlaylistOutput, Track};
use crate::spectrum::NUM_BARS;

pub struct App {
    pub should_quit: bool,
    pub tracks: Vec<Track>,
    pub current_index: usize,
    playing_index: Option<usize>,
    player: Player,
    last_update: Instant,
    pub shuffle: bool,
    pub recursive: bool,
}

impl App {
    pub fn new() -> Self {
        Self {
            should_quit: false,
            tracks: Vec::new(),
            current_index: 0,
            playing_index: None,
            player: Player::new(),
            last_update: Instant::now(),
            shuffle: false,
            recursive: false,
        }
    }

    pub fn save_playlist(&self, path: &PathBuf) -> Result<()> {
        let playlist = PlaylistOutput {
            tracks: self.tracks.iter().map(|t| t.path.to_string_lossy().to_string()).collect(),
        };
        playlist.save(path)
    }

    pub fn scan_directory(&mut self, path: &str, recursive: bool) {
        self.tracks.clear();

        let path = path.trim_matches('"').trim_matches('\'');
        let path = path.trim_end_matches(|c| c == '/' || c == '\\');

        let scan_path = PathBuf::from(path);
        let scan_path = scan_path.canonicalize().unwrap_or(scan_path);

        let walker = if recursive {
            WalkDir::new(&scan_path).follow_links(true)
        } else {
            WalkDir::new(&scan_path).follow_links(true).max_depth(1)
        };

        for entry in walker.into_iter().filter_map(|e| e.ok()) {
            let file_path = entry.path();
            if file_path.is_file() {
                if let Some(ext) = file_path.extension() {
                    let ext_lower = ext.to_string_lossy().to_lowercase();
                    if matches!(ext_lower.as_str(), "mp3" | "wav" | "flac" | "ogg") {
                        self.tracks.push(Track::from_path(file_path.to_path_buf()));
                    }
                }
            }
        }

        self.tracks.sort_by(|a, b| a.name.cmp(&b.name));
        if !self.tracks.is_empty() {
            self.current_index = 0;
        }
    }

    pub fn load_playlist(&mut self, path: &PathBuf) -> Result<()> {
        self.tracks.clear();

        let playlist = Playlist::load(path)?;

        for entry in playlist.tracks {
            let track_path = match entry {
                PlaylistEntry::Simple(p) => PathBuf::from(p),
                PlaylistEntry::Detailed { path: p } => PathBuf::from(p),
            };

            if track_path.exists() {
                if let Some(ext) = track_path.extension() {
                    let ext_lower = ext.to_string_lossy().to_lowercase();
                    if matches!(ext_lower.as_str(), "mp3" | "wav" | "flac" | "ogg") {
                        self.tracks.push(Track::from_path(track_path));
                    }
                }
            }
        }

        if !self.tracks.is_empty() {
            self.current_index = 0;
        }

        Ok(())
    }

    pub fn play_current(&mut self) {
        if self.tracks.is_empty() {
            return;
        }
        let track = &self.tracks[self.current_index];
        let _ = self.player.play(&track.path);
        self.playing_index = Some(self.current_index);
    }

    pub fn toggle_pause(&mut self) {
        match self.player.state() {
            STATE_PLAYING => self.player.pause(),
            STATE_PAUSED => {
                if self.playing_index != Some(self.current_index) {
                    self.play_current();
                } else {
                    self.player.resume();
                }
            }
            STATE_ERROR | STATE_STOPPED => self.play_current(),
            _ => {}
        }
    }

    pub fn stop(&mut self) {
        self.player.stop();
        self.playing_index = None;
    }

    pub fn set_volume(&mut self, delta: f32) {
        let vol = self.player.volume() + delta;
        self.player.set_volume(vol);
    }

    pub fn select_prev(&mut self) {
        if self.tracks.is_empty() {
            return;
        }
        let was_playing = self.player.state() == STATE_PLAYING || self.player.state() == STATE_BUFFERING;
        self.current_index = if self.current_index == 0 {
            self.tracks.len() - 1
        } else {
            self.current_index - 1
        };
        if was_playing {
            self.play_current();
        }
    }

    pub fn select_next(&mut self) {
        if self.tracks.is_empty() {
            return;
        }
        let was_playing = self.player.state() == STATE_PLAYING || self.player.state() == STATE_BUFFERING;
        self.current_index = (self.current_index + 1) % self.tracks.len();
        if was_playing {
            self.play_current();
        }
    }

    pub fn next_track(&mut self) {
        self.select_next();
        self.play_current();
    }

    pub fn prev_track(&mut self) {
        self.select_prev();
        self.play_current();
    }

    pub fn check_playback(&mut self) {
        let state = self.player.state();
        if (state == STATE_PLAYING || state == STATE_BUFFERING) && self.player.is_finished() {
            self.next_track();
        }
    }

    pub fn toggle_shuffle(&mut self) {
        self.shuffle = !self.shuffle;
        if self.shuffle && self.tracks.len() > 1 {
            let current_path = if !self.tracks.is_empty() {
                Some(self.tracks[self.current_index].path.clone())
            } else {
                None
            };

            let mut rng = thread_rng();
            self.tracks.shuffle(&mut rng);

            if let Some(path) = current_path {
                if let Some(pos) = self.tracks.iter().position(|t| t.path == path) {
                    self.tracks.swap(self.current_index, pos);
                }
            }
        }
    }

    pub fn update(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f32();
        self.last_update = now;

        let playing = self.player.state() == STATE_PLAYING;
        if let Ok(mut spectrum) = self.player.spectrum.try_lock() {
            spectrum.update_smooth(dt, playing);
        }
    }

    pub fn handle_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
            KeyCode::Left | KeyCode::Char('h') => self.select_prev(),
            KeyCode::Right | KeyCode::Char('l') => self.select_next(),
            KeyCode::Enter => self.play_current(),
            KeyCode::Char(' ') => self.toggle_pause(),
            KeyCode::Char('s') => self.toggle_shuffle(),
            KeyCode::Char('n') => self.next_track(),
            KeyCode::Char('p') => self.prev_track(),
            KeyCode::Up | KeyCode::Char('+') | KeyCode::Char('=') => self.set_volume(0.05),
            KeyCode::Down | KeyCode::Char('-') | KeyCode::Char('_') => self.set_volume(-0.05),
            KeyCode::Char('r') => self.scan_directory(".", self.recursive),
            KeyCode::Char('w') => {
                let _ = self.save_playlist(&PathBuf::from("playlist.json"));
            }
            _ => {}
        }
    }

    pub fn render(&self) -> Result<()> {
        let mut stdout = io::stdout();

        stdout.execute(MoveToColumn(0))?;
        stdout.execute(Clear(ClearType::CurrentLine))?;

        let state = self.player.state();

        let (state_icon, state_color) = match state {
            STATE_PLAYING => ("▶", Color::Green),
            STATE_PAUSED => ("⏸", Color::Yellow),
            STATE_BUFFERING => ("◌", Color::Cyan),
            STATE_ERROR => ("✗", Color::Red),
            _ => ("⏹", Color::DarkGrey),
        };

        let track_name = if !self.tracks.is_empty() {
            &self.tracks[self.current_index].name
        } else {
            "No tracks"
        };

        let display_name: String = if track_name.len() > 30 {
            format!("{}…", &track_name[..29])
        } else {
            track_name.to_string()
        };

        let volume_pct = (self.player.volume() * 100.0) as u8;

        let eq_display = if let Ok(spectrum) = self.player.spectrum.try_lock() {
            spectrum.render()
        } else {
            " ".repeat(NUM_BARS)
        };

        stdout.execute(SetForegroundColor(Color::DarkGrey))?;
        stdout.execute(Print("♪ "))?;

        stdout.execute(SetForegroundColor(state_color))?;
        stdout.execute(Print(state_icon))?;
        stdout.execute(Print(" "))?;

        stdout.execute(SetForegroundColor(Color::Magenta))?;
        stdout.execute(Print(&eq_display))?;
        stdout.execute(Print(" "))?;

        stdout.execute(SetForegroundColor(Color::White))?;
        stdout.execute(Print(&display_name))?;

        let (current_secs, total_secs) = self.player.playback_time();
        if total_secs > 0 || current_secs > 0 {
            stdout.execute(SetForegroundColor(Color::DarkGrey))?;
            stdout.execute(Print(" "))?;
            stdout.execute(SetForegroundColor(Color::Cyan))?;
            stdout.execute(Print(format!(
                "{}:{:02}/{}:{:02}",
                current_secs / 60,
                current_secs % 60,
                total_secs / 60,
                total_secs % 60
            )))?;
        }

        stdout.execute(SetForegroundColor(Color::DarkGrey))?;
        stdout.execute(Print(" │ "))?;

        match state {
            STATE_BUFFERING => {
                stdout.execute(SetForegroundColor(Color::Cyan))?;
                let buf_pct = self.player.buffer_percent();
                stdout.execute(Print(format!("Buffering {}%", buf_pct)))?;
            }
            STATE_ERROR => {
                stdout.execute(SetForegroundColor(Color::Red))?;
                let err = self.player.error().unwrap_or_else(|| "Error".to_string());
                let short_err: String = if err.len() > 40 {
                    format!("{}…", &err[..39])
                } else {
                    err
                };
                stdout.execute(Print(short_err))?;
            }
            STATE_PLAYING => {
                let buf_pct = self.player.buffer_percent();
                stdout.execute(SetForegroundColor(Color::Green))?;
                stdout.execute(Print(format!("{}%", volume_pct)))?;
                if buf_pct < 50 {
                    stdout.execute(SetForegroundColor(Color::Yellow))?;
                    stdout.execute(Print(format!(" buf:{}%", buf_pct)))?;
                }
            }
            _ => {
                stdout.execute(SetForegroundColor(Color::Green))?;
                stdout.execute(Print(format!("{}%", volume_pct)))?;
            }
        }

        stdout.execute(SetForegroundColor(Color::DarkGrey))?;
        if !self.tracks.is_empty() {
            stdout.execute(Print(format!(
                " │ {}/{}",
                self.current_index + 1,
                self.tracks.len()
            )))?;
        }

        if self.shuffle {
            stdout.execute(Print(" "))?;
            stdout.execute(SetForegroundColor(Color::Cyan))?;
            stdout.execute(Print("🔀"))?;
        }

        stdout.execute(ResetColor)?;
        stdout.flush()?;

        Ok(())
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

