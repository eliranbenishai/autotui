use std::fs::File;
use std::io::{self, BufReader, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleRate, Stream, StreamConfig};
use crossterm::{
    cursor::{Hide, MoveToColumn, Show},
    event::{self, Event, KeyCode, KeyEventKind},
    style::{Color, Print, ResetColor, SetForegroundColor},
    terminal::{disable_raw_mode, enable_raw_mode, Clear, ClearType},
    ExecutableCommand,
};
use rubato::{SincFixedIn, SincInterpolationType, SincInterpolationParameters, Resampler, WindowFunction};
use rustfft::{num_complex::Complex, FftPlanner};
use serde::{Deserialize, Serialize};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use rand::seq::SliceRandom;
use rand::thread_rng;
use walkdir::WalkDir;

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

/// JSON playlist format (for reading)
#[derive(Debug, Deserialize)]
struct Playlist {
    /// Optional playlist name (reserved for future use)
    #[serde(default)]
    #[allow(dead_code)]
    name: Option<String>,
    /// List of track paths
    tracks: Vec<PlaylistEntry>,
}

/// Simple playlist format (for writing)
#[derive(Debug, Serialize)]
struct PlaylistOutput {
    tracks: Vec<String>,
}

/// A playlist entry can be a simple path string or an object with more details
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum PlaylistEntry {
    /// Simple path string
    Simple(String),
    /// Object with path and optional metadata
    Detailed { path: String },
}

const NUM_BARS: usize = 16;
const FFT_SIZE: usize = 1024;
const BAR_CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
const SAMPLE_BUFFER_SIZE: usize = 88200 * 8; // ~8 seconds at 44.1kHz stereo
const PREBUFFER_SIZE: usize = 44100;         // ~0.5 seconds before starting (reduces crackling)

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
            // Map bands from 30 Hz to 8000 Hz (log scale) for better music visualization
            // Most music energy is in bass/mids, so we cap at 8kHz instead of 20kHz
            let freq_low = 30.0 * (266.7_f32).powf(band_idx as f32 / NUM_BARS as f32);
            let freq_high = 30.0 * (266.7_f32).powf((band_idx + 1) as f32 / NUM_BARS as f32);

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
const STATE_BUFFERING: u8 = 3;
const STATE_ERROR: u8 = 4;

struct Player {
    ring_buffer: Arc<Mutex<RingBuffer>>,
    spectrum: Arc<Mutex<SpectrumAnalyzer>>,
    state: Arc<AtomicU8>,
    stop_signal: Arc<AtomicBool>,
    volume: Arc<Mutex<f32>>,
    finished: Arc<AtomicBool>,
    error_msg: Arc<Mutex<Option<String>>>,
    samples_played: Arc<AtomicU32>,
    source_sample_rate: Arc<AtomicU32>,
    output_sample_rate: Arc<AtomicU32>,
    total_duration_secs: Arc<AtomicU32>,
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
            error_msg: Arc::new(Mutex::new(None)),
            samples_played: Arc::new(AtomicU32::new(0)),
            source_sample_rate: Arc::new(AtomicU32::new(44100)),
            output_sample_rate: Arc::new(AtomicU32::new(44100)),
            total_duration_secs: Arc::new(AtomicU32::new(0)),
            _stream: None,
            _decoder_handle: None,
        }
    }

    fn play(&mut self, path: &PathBuf) -> Result<()> {
        self.stop();

        self.stop_signal.store(false, Ordering::SeqCst);
        self.finished.store(false, Ordering::SeqCst);
        self.state.store(STATE_BUFFERING, Ordering::SeqCst);
        
        if let Ok(mut err) = self.error_msg.lock() {
            *err = None;
        }

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

        // Calculate total duration from n_frames if available
        let total_duration_secs = codec_params.n_frames
            .map(|frames| (frames / sample_rate as u64) as u32)
            .unwrap_or(0);

        // Get output device and determine output sample rate FIRST
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow::anyhow!("No output device"))?;

        // Try to use the file's sample rate, fall back to device default
        let desired_config = StreamConfig {
            channels: channels as u16,
            sample_rate: SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        // Check if the device supports our desired config, otherwise use default
        let config = if device.supported_output_configs()
            .map(|mut configs| {
                configs.any(|c| {
                    c.channels() == channels as u16 &&
                    c.min_sample_rate().0 <= sample_rate &&
                    c.max_sample_rate().0 >= sample_rate
                })
            })
            .unwrap_or(false)
        {
            desired_config
        } else {
            // Fall back to device's default config
            device.default_output_config()
                .map(|c| StreamConfig {
                    channels: c.channels(),
                    sample_rate: c.sample_rate(),
                    buffer_size: cpal::BufferSize::Default,
                })
                .unwrap_or(desired_config)
        };

        let output_sample_rate = config.sample_rate.0;
        let output_channels = config.channels as usize;
        let source_channels = channels;

        // Store playback info
        self.source_sample_rate.store(sample_rate, Ordering::SeqCst);
        self.output_sample_rate.store(output_sample_rate, Ordering::SeqCst);
        self.total_duration_secs.store(total_duration_secs, Ordering::SeqCst);
        self.samples_played.store(0, Ordering::SeqCst);

        let mut decoder =
            symphonia::default::get_codecs().make(&codec_params, &DecoderOptions::default())?;

        let ring_buffer = Arc::clone(&self.ring_buffer);
        let stop_signal = Arc::clone(&self.stop_signal);
        let state = Arc::clone(&self.state);
        let finished = Arc::clone(&self.finished);
        let error_msg = Arc::clone(&self.error_msg);

        // Spawn decoder thread with resampling support
        let decoder_handle = thread::spawn(move || {
            let mut prebuffered = false;
            let mut decode_errors = 0;
            
            // Create resampler if sample rates differ
            let needs_resample = sample_rate != output_sample_rate;
            let resample_ratio = output_sample_rate as f64 / sample_rate as f64;
            
            let mut resampler: Option<SincFixedIn<f32>> = if needs_resample {
                // Try with quality settings first
                let params = SincInterpolationParameters {
                    sinc_len: 64,
                    f_cutoff: 0.925,
                    interpolation: SincInterpolationType::Linear,
                    oversampling_factor: 128,
                    window: WindowFunction::Blackman,
                };
                SincFixedIn::new(
                    resample_ratio,
                    2.0,
                    params,
                    512,
                    source_channels,
                ).or_else(|_| {
                    // Fallback to simpler settings if first attempt fails
                    let simple_params = SincInterpolationParameters {
                        sinc_len: 32,
                        f_cutoff: 0.9,
                        interpolation: SincInterpolationType::Nearest,
                        oversampling_factor: 64,
                        window: WindowFunction::Hann,
                    };
                    SincFixedIn::new(
                        resample_ratio,
                        2.0,
                        simple_params,
                        256,
                        source_channels,
                    )
                }).ok()
            } else {
                None
            };
            
            // If resampling is needed but resampler failed, signal an error
            if needs_resample && resampler.is_none() {
                if let Ok(mut err) = error_msg.lock() {
                    *err = Some(format!("Resampler init failed ({}Hz -> {}Hz)", sample_rate, output_sample_rate));
                }
                state.store(STATE_ERROR, Ordering::SeqCst);
                finished.store(true, Ordering::SeqCst);
                return;
            }

            // Buffers for resampling (same channels in and out - resampler doesn't change channel count)
            let mut resample_in: Vec<Vec<f32>> = vec![Vec::new(); source_channels];

            loop {
                if stop_signal.load(Ordering::SeqCst) {
                    break;
                }

                let current_state = state.load(Ordering::SeqCst);
                
                // Wait while paused
                while current_state == STATE_PAUSED {
                    if stop_signal.load(Ordering::SeqCst) {
                        return;
                    }
                    thread::sleep(Duration::from_millis(10));
                }

                let packet = match format.next_packet() {
                    Ok(p) => p,
                    Err(symphonia::core::errors::Error::IoError(e)) => {
                        let err_str = e.to_string();
                        if err_str.contains("end of stream") || e.kind() == std::io::ErrorKind::UnexpectedEof {
                            finished.store(true, Ordering::SeqCst);
                        } else {
                            if let Ok(mut err) = error_msg.lock() {
                                *err = Some(format!("IO: {}", err_str));
                            }
                            state.store(STATE_ERROR, Ordering::SeqCst);
                            finished.store(true, Ordering::SeqCst);
                        }
                        break;
                    }
                    Err(symphonia::core::errors::Error::DecodeError(e)) => {
                        if let Ok(mut err) = error_msg.lock() {
                            *err = Some(format!("Decode: {}", e));
                        }
                        state.store(STATE_ERROR, Ordering::SeqCst);
                        finished.store(true, Ordering::SeqCst);
                        break;
                    }
                    Err(e) => {
                        let err_str = e.to_string();
                        if err_str.contains("end of stream") {
                            finished.store(true, Ordering::SeqCst);
                        } else {
                            if let Ok(mut err) = error_msg.lock() {
                                *err = Some(format!("Error: {}", err_str));
                            }
                            state.store(STATE_ERROR, Ordering::SeqCst);
                            finished.store(true, Ordering::SeqCst);
                        }
                        break;
                    }
                };

                if packet.track_id() != track_id {
                    continue;
                }

                let decoded = match decoder.decode(&packet) {
                    Ok(d) => {
                        decode_errors = 0;
                        d
                    }
                    Err(e) => {
                        decode_errors += 1;
                        if decode_errors > 10 {
                            if let Ok(mut err) = error_msg.lock() {
                                *err = Some(format!("Decode error: {}", e));
                            }
                            state.store(STATE_ERROR, Ordering::SeqCst);
                            finished.store(true, Ordering::SeqCst);
                            break;
                        }
                        continue;
                    }
                };

                let spec = *decoded.spec();
                let duration = decoded.capacity() as u64;
                let mut sample_buf = SampleBuffer::<f32>::new(duration, spec);
                sample_buf.copy_interleaved_ref(decoded);

                let interleaved_samples = sample_buf.samples();

                // Process samples (with optional resampling)
                let resampled_samples: Vec<f32> = if let Some(ref mut resampler) = resampler {
                    // De-interleave into separate channel buffers (accumulate)
                    for (i, &sample) in interleaved_samples.iter().enumerate() {
                        resample_in[i % source_channels].push(sample);
                    }

                    // Process complete chunks through resampler
                    let chunk_size = resampler.input_frames_max();
                    let mut all_output = Vec::new();

                    while resample_in[0].len() >= chunk_size {
                        // Take a chunk from each channel
                        let chunk_in: Vec<Vec<f32>> = resample_in.iter_mut()
                            .map(|ch| ch.drain(..chunk_size).collect())
                            .collect();

                        // Resample - SincFixedIn returns Vec<Vec<f32>>
                        match resampler.process(&chunk_in, None) {
                            Ok(resampled) => {
                                // Re-interleave output (channel count stays the same as source)
                                if !resampled.is_empty() && !resampled[0].is_empty() {
                                    let out_len = resampled[0].len();
                                    for i in 0..out_len {
                                        for ch in 0..source_channels {
                                            all_output.push(resampled[ch][i]);
                                        }
                                    }
                                }
                            }
                            Err(_) => {
                                // On error, skip this chunk
                            }
                        }
                    }
                    all_output
                } else {
                    interleaved_samples.to_vec()
                };

                // Convert channels if needed (source_channels -> output_channels)
                let output_samples: Vec<f32> = if source_channels == output_channels {
                    resampled_samples
                } else if source_channels == 1 && output_channels == 2 {
                    // Mono to stereo: duplicate each sample
                    resampled_samples.iter().flat_map(|&s| [s, s]).collect()
                } else if source_channels == 2 && output_channels == 1 {
                    // Stereo to mono: average pairs
                    resampled_samples.chunks(2)
                        .map(|pair| (pair.get(0).unwrap_or(&0.0) + pair.get(1).unwrap_or(&0.0)) * 0.5)
                        .collect()
                } else {
                    // For other conversions, just use what we have (may not be ideal)
                    resampled_samples
                };

                // Push to ring buffer
                let mut offset = 0;
                while offset < output_samples.len() {
                    if stop_signal.load(Ordering::SeqCst) {
                        return;
                    }

                    if let Ok(mut rb) = ring_buffer.try_lock() {
                        let written = rb.push(&output_samples[offset..]);
                        offset += written;
                        
                        if !prebuffered && rb.available() >= PREBUFFER_SIZE {
                            prebuffered = true;
                            state.store(STATE_PLAYING, Ordering::SeqCst);
                        }
                        
                        if written == 0 {
                            drop(rb);
                            thread::sleep(Duration::from_millis(1));
                        }
                    } else {
                        thread::sleep(Duration::from_micros(100));
                    }
                }
            }
        });

        self._decoder_handle = Some(decoder_handle);

        let ring_buffer = Arc::clone(&self.ring_buffer);
        let state = Arc::clone(&self.state);
        let volume = Arc::clone(&self.volume);
        let spectrum = Arc::clone(&self.spectrum);
        let samples_played = Arc::clone(&self.samples_played);
        let output_channels = config.channels as u32;

        let stream = device.build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let current_state = state.load(Ordering::SeqCst);

                if current_state != STATE_PLAYING {
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

                    // Track playback position (samples / channels = frames)
                    let frames_played = (read as u32) / output_channels.max(1);
                    samples_played.fetch_add(frames_played, Ordering::SeqCst);

                    if let Ok(mut sp) = spectrum.try_lock() {
                        sp.push_samples(&data[..read]);
                    }
                } else {
                    data.fill(0.0);
                }
            },
            move |err| {
                // Log errors but don't crash
                eprintln!("Audio stream error: {}", err);
            },
            None,
        )?;

        stream.play()?;
        self._stream = Some(stream);

        Ok(())
    }

    fn pause(&self) {
        let current = self.state.load(Ordering::SeqCst);
        if current == STATE_PLAYING {
            self.state.store(STATE_PAUSED, Ordering::SeqCst);
        }
    }

    fn resume(&self) {
        let current = self.state.load(Ordering::SeqCst);
        if current == STATE_PAUSED {
            self.state.store(STATE_PLAYING, Ordering::SeqCst);
        }
    }

    fn stop(&mut self) {
        self.stop_signal.store(true, Ordering::SeqCst);
        self.state.store(STATE_STOPPED, Ordering::SeqCst);

        // Drop the stream first to stop audio callback
        if let Some(stream) = self._stream.take() {
            drop(stream);
        }

        // Wait for decoder thread to finish
        if let Some(handle) = self._decoder_handle.take() {
            let _ = handle.join();
        }

        // Reset playback position
        self.samples_played.store(0, Ordering::SeqCst);
        
        // Clear the buffer
        if let Ok(mut rb) = self.ring_buffer.lock() {
            rb.clear();
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

    fn error(&self) -> Option<String> {
        self.error_msg.lock().ok().and_then(|e| e.clone())
    }

    fn buffer_percent(&self) -> u8 {
        if let Ok(rb) = self.ring_buffer.try_lock() {
            ((rb.available() as f32 / SAMPLE_BUFFER_SIZE as f32) * 100.0) as u8
        } else {
            0
        }
    }

    fn set_volume(&self, vol: f32) {
        if let Ok(mut v) = self.volume.try_lock() {
            *v = vol.clamp(0.0, 1.0);
        }
    }

    fn volume(&self) -> f32 {
        self.volume.try_lock().map(|v| *v).unwrap_or(0.5)
    }

    fn playback_time(&self) -> (u32, u32) {
        // Use output sample rate for current time (samples_played is in output frames)
        let output_rate = self.output_sample_rate.load(Ordering::SeqCst);
        let samples = self.samples_played.load(Ordering::SeqCst);
        let current_secs = if output_rate > 0 { samples / output_rate } else { 0 };
        let total_secs = self.total_duration_secs.load(Ordering::SeqCst);
        (current_secs, total_secs)
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
    playing_index: Option<usize>, // Track which index is actually playing
    player: Player,
    last_update: Instant,
    shuffle: bool,
    recursive: bool,
}

impl App {
    fn new() -> Self {
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

    fn save_playlist(&self, path: &PathBuf) -> Result<()> {
        let playlist = PlaylistOutput {
            tracks: self.tracks.iter().map(|t| t.path.to_string_lossy().to_string()).collect(),
        };
        let json = serde_json::to_string_pretty(&playlist)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    fn scan_directory(&mut self, path: &str, recursive: bool) {
        self.tracks.clear();

        // Normalize the path: trim quotes and trailing slashes/backslashes
        let path = path.trim_matches('"').trim_matches('\'');
        let path = path.trim_end_matches(|c| c == '/' || c == '\\');
        
        // Convert to PathBuf and try to canonicalize for better compatibility
        let scan_path = PathBuf::from(path);
        let scan_path = scan_path.canonicalize().unwrap_or_else(|_| scan_path);

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

    fn load_playlist(&mut self, path: &PathBuf) -> Result<()> {
        self.tracks.clear();

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let playlist: Playlist = serde_json::from_reader(reader)?;

        for entry in playlist.tracks {
            let track_path = match entry {
                PlaylistEntry::Simple(p) => PathBuf::from(p),
                PlaylistEntry::Detailed { path: p } => PathBuf::from(p),
            };

            // Verify file exists and has valid extension
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

    fn play_current(&mut self) {
        if self.tracks.is_empty() {
            return;
        }
        let track = &self.tracks[self.current_index];
        let _ = self.player.play(&track.path);
        self.playing_index = Some(self.current_index);
    }

    fn toggle_pause(&mut self) {
        match self.player.state() {
            STATE_PLAYING => self.player.pause(),
            STATE_PAUSED => {
                // If the selected track changed while paused, play the new track
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

    fn stop(&mut self) {
        self.player.stop();
        self.playing_index = None;
    }

    fn set_volume(&mut self, delta: f32) {
        let vol = self.player.volume() + delta;
        self.player.set_volume(vol);
    }

    fn select_prev(&mut self) {
        if self.tracks.is_empty() {
            return;
        }
        let was_playing = self.player.state() == STATE_PLAYING || self.player.state() == STATE_BUFFERING;
        self.current_index = if self.current_index == 0 {
            self.tracks.len() - 1
        } else {
            self.current_index - 1
        };
        // If playing, immediately switch to the new track
        if was_playing {
            self.play_current();
        }
    }

    fn select_next(&mut self) {
        if self.tracks.is_empty() {
            return;
        }
        let was_playing = self.player.state() == STATE_PLAYING || self.player.state() == STATE_BUFFERING;
        self.current_index = (self.current_index + 1) % self.tracks.len();
        // If playing, immediately switch to the new track
        if was_playing {
            self.play_current();
        }
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
        let state = self.player.state();
        if (state == STATE_PLAYING || state == STATE_BUFFERING) && self.player.is_finished() {
            self.next_track();
        }
    }

    fn toggle_shuffle(&mut self) {
        self.shuffle = !self.shuffle;
        if self.shuffle && self.tracks.len() > 1 {
            // Shuffle remaining tracks (keep current track, shuffle the rest)
            let current_path = if !self.tracks.is_empty() {
                Some(self.tracks[self.current_index].path.clone())
            } else {
                None
            };
            
            let mut rng = thread_rng();
            self.tracks.shuffle(&mut rng);
            
            // Find and move current track back to current_index position
            if let Some(path) = current_path {
                if let Some(pos) = self.tracks.iter().position(|t| t.path == path) {
                    self.tracks.swap(self.current_index, pos);
                }
            }
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
            KeyCode::Char('s') => self.toggle_shuffle(),
            KeyCode::Char('n') => self.next_track(),
            KeyCode::Char('p') => self.prev_track(),
            KeyCode::Up | KeyCode::Char('+') | KeyCode::Char('=') => self.set_volume(0.05),
            KeyCode::Down | KeyCode::Char('-') | KeyCode::Char('_') => self.set_volume(-0.05),
            KeyCode::Char('r') => self.scan_directory(".", self.recursive),
            KeyCode::Char('w') => { let _ = self.save_playlist(&PathBuf::from("playlist.json")); }
            _ => {}
        }
    }

    fn render(&self) -> Result<()> {
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

        // Show playback time
        let (current_secs, total_secs) = self.player.playback_time();
        if total_secs > 0 || current_secs > 0 {
            stdout.execute(SetForegroundColor(Color::DarkGrey))?;
            stdout.execute(Print(" "))?;
            stdout.execute(SetForegroundColor(Color::Cyan))?;
            stdout.execute(Print(format!(
                "{}:{:02}/{}:{:02}",
                current_secs / 60, current_secs % 60,
                total_secs / 60, total_secs % 60
            )))?;
        }

        // Show status info
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
                // Show volume and buffer level
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

        // Show shuffle indicator
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
        let canonical_path = normalized_path.canonicalize().unwrap_or(normalized_path.clone());
        
        if canonical_path.is_dir() {
            // Directory: scan for audio files
            app.scan_directory(&canonical_path.to_string_lossy(), args.recursive);
        } else if canonical_path.is_file() {
            if let Some(ext) = canonical_path.extension() {
                let ext_lower = ext.to_string_lossy().to_lowercase();
                if ext_lower == "json" {
                    // JSON file: treat as playlist
                    if let Err(e) = app.load_playlist(&canonical_path) {
                        eprintln!("Error loading playlist: {}", e);
                        return Err(e);
                    }
                } else if matches!(ext_lower.as_str(), "mp3" | "wav" | "flac" | "ogg") {
                    // Audio file: add as single track
                    app.tracks.push(Track::from_path(canonical_path.clone()));
                } else {
                    eprintln!("Unsupported file type: {}", ext_lower);
                    return Ok(());
                }
            }
        } else {
            // Try as directory anyway (might work with network paths)
            app.scan_directory(&path_str, args.recursive);
            if app.tracks.is_empty() {
                eprintln!("Path not found or no tracks: {}", path.display());
                return Ok(());
            }
        }
    } else {
        // Default: scan current directory
        app.scan_directory(".", args.recursive);
    }

    // Shuffle tracks if requested via CLI
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

    io::stdout().execute(MoveToColumn(0))?;
    io::stdout().execute(Clear(ClearType::CurrentLine))?;
    io::stdout().execute(Show)?;
    disable_raw_mode()?;

    println!("Enjoy the silence!");

    Ok(())
}



