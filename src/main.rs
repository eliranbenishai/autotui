use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleRate, Stream, StreamConfig};
use crossterm::{
    cursor::{Hide, MoveToColumn, Show},
    event::{self, Event, KeyCode, KeyEventKind},
    style::{Color, Print, ResetColor, SetForegroundColor},
    terminal::{disable_raw_mode, enable_raw_mode, Clear, ClearType},
    ExecutableCommand,
};
use rustfft::{num_complex::Complex, FftPlanner};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use walkdir::WalkDir;

const NUM_BARS: usize = 16;
const FFT_SIZE: usize = 1024;
const BAR_CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
const SAMPLE_BUFFER_SIZE: usize = 8192;

struct Track {
    path: PathBuf,
    name: String,
}

impl Track {
    fn from_path(path: PathBuf) -> Self {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Unknown")
            .to_string();
        Self { path, name }
    }
}

struct RingBuffer {
    data: Vec<f32>,
    write_pos: usize,
    read_pos: usize,
    len: usize,
}

impl RingBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            data: vec![0.0; capacity],
            write_pos: 0,
            read_pos: 0,
            len: 0,
        }
    }

    fn push(&mut self, samples: &[f32]) -> usize {
        let capacity = self.data.len();
        let available = capacity - self.len;
        let to_write = samples.len().min(available);

        for &sample in samples.iter().take(to_write) {
            self.data[self.write_pos] = sample;
            self.write_pos = (self.write_pos + 1) % capacity;
        }
        self.len += to_write;
        to_write
    }

    fn pop(&mut self, out: &mut [f32]) -> usize {
        let to_read = out.len().min(self.len);
        let capacity = self.data.len();

        for sample in out.iter_mut().take(to_read) {
            *sample = self.data[self.read_pos];
            self.read_pos = (self.read_pos + 1) % capacity;
        }
        self.len -= to_read;
        to_read
    }

    fn available(&self) -> usize {
        self.len
    }

    fn clear(&mut self) {
        self.write_pos = 0;
        self.read_pos = 0;
        self.len = 0;
    }
}

struct SpectrumAnalyzer {
    bands: [f32; NUM_BARS],
    smoothed: [f32; NUM_BARS],
    sample_buffer: Vec<f32>,
    fft_planner: FftPlanner<f32>,
}

impl SpectrumAnalyzer {
    fn new() -> Self {
        Self {
            bands: [0.0; NUM_BARS],
            smoothed: [0.0; NUM_BARS],
            sample_buffer: Vec::with_capacity(FFT_SIZE),
            fft_planner: FftPlanner::new(),
        }
    }

    fn push_samples(&mut self, samples: &[f32]) {
        self.sample_buffer.extend_from_slice(samples);
        if self.sample_buffer.len() > FFT_SIZE * 2 {
            self.sample_buffer.drain(0..self.sample_buffer.len() - FFT_SIZE);
        }
    }

    fn analyze(&mut self) {
        if self.sample_buffer.len() < FFT_SIZE {
            return;
        }

        let fft = self.fft_planner.plan_fft_forward(FFT_SIZE);
        let start = self.sample_buffer.len() - FFT_SIZE;

        let mut buffer: Vec<Complex<f32>> = self.sample_buffer[start..]
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let window =
                    0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / FFT_SIZE as f32).cos();
                Complex::new(s * window, 0.0)
            })
            .collect();

        fft.process(&mut buffer);

        let bin_count = FFT_SIZE / 2;

        for (band_idx, band) in self.bands.iter_mut().enumerate() {
            let freq_low = 20.0 * (1000.0_f32).powf(band_idx as f32 / NUM_BARS as f32);
            let freq_high = 20.0 * (1000.0_f32).powf((band_idx + 1) as f32 / NUM_BARS as f32);

            let bin_low = ((freq_low / 22050.0) * bin_count as f32) as usize;
            let bin_high = ((freq_high / 22050.0) * bin_count as f32) as usize;

            let bin_low = bin_low.clamp(0, bin_count - 1);
            let bin_high = bin_high.clamp(bin_low + 1, bin_count);

            let mut sum = 0.0;
            for i in bin_low..bin_high {
                sum += buffer[i].norm();
            }
            let avg = sum / (bin_high - bin_low).max(1) as f32;

            *band = (avg / 30.0).clamp(0.0, 1.0);
        }
    }

    fn update_smooth(&mut self, dt: f32, playing: bool) {
        self.analyze();

        for i in 0..NUM_BARS {
            let target = if playing { self.bands[i] } else { 0.0 };
            let speed = if playing { 20.0 } else { 10.0 };
            self.smoothed[i] += (target - self.smoothed[i]) * speed * dt;
            self.smoothed[i] = self.smoothed[i].clamp(0.0, 1.0);
        }
    }

    fn render(&self) -> String {
        self.smoothed
            .iter()
            .map(|&v| {
                let idx = ((v * 7.0) as usize).min(7);
                BAR_CHARS[idx]
            })
            .collect()
    }

    fn clear(&mut self) {
        self.sample_buffer.clear();
        self.bands = [0.0; NUM_BARS];
    }
}

const STATE_STOPPED: u8 = 0;
const STATE_PLAYING: u8 = 1;
const STATE_PAUSED: u8 = 2;

struct Player {
    ring_buffer: Arc<Mutex<RingBuffer>>,
    spectrum: Arc<Mutex<SpectrumAnalyzer>>,
    state: Arc<AtomicU8>,
    stop_signal: Arc<AtomicBool>,
    volume: Arc<Mutex<f32>>,
    finished: Arc<AtomicBool>,
    _stream: Option<Stream>,
    _decoder_handle: Option<thread::JoinHandle<()>>,
}

impl Player {
    fn new() -> Self {
        Self {
            ring_buffer: Arc::new(Mutex::new(RingBuffer::new(SAMPLE_BUFFER_SIZE))),
            spectrum: Arc::new(Mutex::new(SpectrumAnalyzer::new())),
            state: Arc::new(AtomicU8::new(STATE_STOPPED)),
            stop_signal: Arc::new(AtomicBool::new(false)),
            volume: Arc::new(Mutex::new(0.5)),
            finished: Arc::new(AtomicBool::new(false)),
            _stream: None,
            _decoder_handle: None,
        }
    }

    fn play(&mut self, path: &PathBuf) -> Result<()> {
        self.stop();

        self.stop_signal.store(false, Ordering::SeqCst);
        self.finished.store(false, Ordering::SeqCst);

        if let Ok(mut rb) = self.ring_buffer.lock() {
            rb.clear();
        }
        if let Ok(mut sp) = self.spectrum.lock() {
            sp.clear();
        }

        let file = File::open(path)?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        let mut hint = Hint::new();
        if let Some(ext) = path.extension() {
            hint.with_extension(&ext.to_string_lossy());
        }

        let probed = symphonia::default::get_probe().format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )?;

        let mut format = probed.format;
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
            .ok_or_else(|| anyhow::anyhow!("No supported audio track"))?;

        let codec_params = track.codec_params.clone();
        let track_id = track.id;
        let sample_rate = codec_params.sample_rate.unwrap_or(44100);
        let channels = codec_params.channels.map(|c| c.count()).unwrap_or(2);

        let mut decoder =
            symphonia::default::get_codecs().make(&codec_params, &DecoderOptions::default())?;

        let ring_buffer = Arc::clone(&self.ring_buffer);
        let stop_signal = Arc::clone(&self.stop_signal);
        let state = Arc::clone(&self.state);
        let finished = Arc::clone(&self.finished);

        let decoder_handle = thread::spawn(move || {
            loop {
                if stop_signal.load(Ordering::SeqCst) {
                    break;
                }

                while state.load(Ordering::SeqCst) == STATE_PAUSED {
                    if stop_signal.load(Ordering::SeqCst) {
                        return;
                    }
                    thread::sleep(Duration::from_millis(10));
                }

                let packet = match format.next_packet() {
                    Ok(p) => p,
                    Err(_) => {
                        finished.store(true, Ordering::SeqCst);
                        break;
                    }
                };

                if packet.track_id() != track_id {
                    continue;
                }

                let decoded = match decoder.decode(&packet) {
                    Ok(d) => d,
                    Err(_) => continue,
                };

                let spec = *decoded.spec();
                let duration = decoded.capacity() as u64;
                let mut sample_buf = SampleBuffer::<f32>::new(duration, spec);
                sample_buf.copy_interleaved_ref(decoded);

                let samples = sample_buf.samples();
                let mut offset = 0;

                while offset < samples.len() {
                    if stop_signal.load(Ordering::SeqCst) {
                        return;
                    }

                    if let Ok(mut rb) = ring_buffer.try_lock() {
                        let written = rb.push(&samples[offset..]);
                        offset += written;
                        if written == 0 {
                            drop(rb);
                            thread::sleep(Duration::from_micros(500));
                        }
                    } else {
                        thread::sleep(Duration::from_micros(100));
                    }
                }
            }
        });

        self._decoder_handle = Some(decoder_handle);

        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow::anyhow!("No output device"))?;

        let config = StreamConfig {
            channels: channels as u16,
            sample_rate: SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let ring_buffer = Arc::clone(&self.ring_buffer);
        let state = Arc::clone(&self.state);
        let volume = Arc::clone(&self.volume);
        let spectrum = Arc::clone(&self.spectrum);

        let stream = device.build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let current_state = state.load(Ordering::SeqCst);

                if current_state == STATE_PAUSED || current_state == STATE_STOPPED {
                    data.fill(0.0);
                    return;
                }

                let vol = volume.try_lock().map(|v| *v).unwrap_or(0.5);

                if let Ok(mut rb) = ring_buffer.try_lock() {
                    let read = rb.pop(data);
                    for sample in data.iter_mut().take(read) {
                        *sample *= vol;
                    }
                    for sample in data.iter_mut().skip(read) {
                        *sample = 0.0;
                    }

                    if let Ok(mut sp) = spectrum.try_lock() {
                        sp.push_samples(&data[..read]);
                    }
                } else {
                    data.fill(0.0);
                }
            },
            |err| eprintln!("Audio error: {}", err),
            None,
        )?;

        stream.play()?;
        self._stream = Some(stream);
        self.state.store(STATE_PLAYING, Ordering::SeqCst);

        Ok(())
    }

    fn pause(&self) {
        self.state.store(STATE_PAUSED, Ordering::SeqCst);
    }

    fn resume(&self) {
        self.state.store(STATE_PLAYING, Ordering::SeqCst);
    }

    fn stop(&mut self) {
        self.stop_signal.store(true, Ordering::SeqCst);
        self.state.store(STATE_STOPPED, Ordering::SeqCst);

        self._stream = None;

        if let Some(handle) = self._decoder_handle.take() {
            let _ = handle.join();
        }
    }

    fn is_finished(&self) -> bool {
        if !self.finished.load(Ordering::SeqCst) {
            return false;
        }
        if let Ok(rb) = self.ring_buffer.try_lock() {
            rb.available() == 0
        } else {
            false
        }
    }

    fn state(&self) -> u8 {
        self.state.load(Ordering::SeqCst)
    }

    fn set_volume(&self, vol: f32) {
        if let Ok(mut v) = self.volume.try_lock() {
            *v = vol.clamp(0.0, 1.0);
        }
    }

    fn volume(&self) -> f32 {
        self.volume.try_lock().map(|v| *v).unwrap_or(0.5)
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        self.stop();
    }
}

struct App {
    should_quit: bool,
    tracks: Vec<Track>,
    current_index: usize,
    player: Player,
    last_update: Instant,
}

impl App {
    fn new() -> Self {
        Self {
            should_quit: false,
            tracks: Vec::new(),
            current_index: 0,
            player: Player::new(),
            last_update: Instant::now(),
        }
    }

    fn scan_directory(&mut self, path: &str) {
        self.tracks.clear();

        for entry in WalkDir::new(path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    let ext_lower = ext.to_string_lossy().to_lowercase();
                    if matches!(ext_lower.as_str(), "mp3" | "wav" | "flac" | "ogg") {
                        self.tracks.push(Track::from_path(path.to_path_buf()));
                    }
                }
            }
        }

        self.tracks.sort_by(|a, b| a.name.cmp(&b.name));
        if !self.tracks.is_empty() {
            self.current_index = 0;
        }
    }

    fn play_current(&mut self) {
        if self.tracks.is_empty() {
            return;
        }
        let track = &self.tracks[self.current_index];
        let _ = self.player.play(&track.path);
    }

    fn toggle_pause(&mut self) {
        match self.player.state() {
            STATE_PLAYING => self.player.pause(),
            STATE_PAUSED => self.player.resume(),
            _ => self.play_current(),
        }
    }

    fn stop(&mut self) {
        self.player.stop();
    }

    fn set_volume(&mut self, delta: f32) {
        let vol = self.player.volume() + delta;
        self.player.set_volume(vol);
    }

    fn select_prev(&mut self) {
        if self.tracks.is_empty() {
            return;
        }
        self.current_index = if self.current_index == 0 {
            self.tracks.len() - 1
        } else {
            self.current_index - 1
        };
    }

    fn select_next(&mut self) {
        if self.tracks.is_empty() {
            return;
        }
        self.current_index = (self.current_index + 1) % self.tracks.len();
    }

    fn next_track(&mut self) {
        self.select_next();
        self.play_current();
    }

    fn prev_track(&mut self) {
        self.select_prev();
        self.play_current();
    }

    fn check_playback(&mut self) {
        if self.player.state() == STATE_PLAYING && self.player.is_finished() {
            self.next_track();
        }
    }

    fn update(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f32();
        self.last_update = now;

        let playing = self.player.state() == STATE_PLAYING;
        if let Ok(mut spectrum) = self.player.spectrum.try_lock() {
            spectrum.update_smooth(dt, playing);
        }
    }

    fn handle_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
            KeyCode::Left | KeyCode::Char('h') => self.select_prev(),
            KeyCode::Right | KeyCode::Char('l') => self.select_next(),
            KeyCode::Enter => self.play_current(),
            KeyCode::Char(' ') => self.toggle_pause(),
            KeyCode::Char('s') => self.stop(),
            KeyCode::Char('n') => self.next_track(),
            KeyCode::Char('p') => self.prev_track(),
            KeyCode::Up | KeyCode::Char('+') | KeyCode::Char('=') => self.set_volume(0.05),
            KeyCode::Down | KeyCode::Char('-') | KeyCode::Char('_') => self.set_volume(-0.05),
            KeyCode::Char('o') => self.scan_directory("."),
            _ => {}
        }
    }

    fn render(&self) -> Result<()> {
        let mut stdout = io::stdout();

        stdout.execute(MoveToColumn(0))?;
        stdout.execute(Clear(ClearType::CurrentLine))?;

        let state_icon = match self.player.state() {
            STATE_PLAYING => "▶",
            STATE_PAUSED => "⏸",
            _ => "⏹",
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

        stdout.execute(SetForegroundColor(Color::Cyan))?;
        stdout.execute(Print(state_icon))?;
        stdout.execute(Print(" "))?;

        stdout.execute(SetForegroundColor(Color::Magenta))?;
        stdout.execute(Print(&eq_display))?;
        stdout.execute(Print(" "))?;

        stdout.execute(SetForegroundColor(Color::White))?;
        stdout.execute(Print(&display_name))?;

        stdout.execute(SetForegroundColor(Color::DarkGrey))?;
        stdout.execute(Print(" │ "))?;

        stdout.execute(SetForegroundColor(Color::Green))?;
        stdout.execute(Print(format!("{}%", volume_pct)))?;

        stdout.execute(SetForegroundColor(Color::DarkGrey))?;
        if !self.tracks.is_empty() {
            stdout.execute(Print(format!(
                " │ {}/{}",
                self.current_index + 1,
                self.tracks.len()
            )))?;
        }

        stdout.execute(ResetColor)?;
        stdout.flush()?;

        Ok(())
    }
}

fn main() -> Result<()> {
    enable_raw_mode()?;
    io::stdout().execute(Hide)?;

    let mut app = App::new();
    app.scan_directory(".");

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

    io::stdout().execute(MoveToColumn(0))?;
    io::stdout().execute(Clear(ClearType::CurrentLine))?;
    io::stdout().execute(Show)?;
    disable_raw_mode()?;

    println!("Goodbye!");

    Ok(())
}
