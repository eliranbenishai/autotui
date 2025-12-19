use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleRate, Stream, StreamConfig};
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::buffer::RingBuffer;
use crate::spectrum::SpectrumAnalyzer;

pub const SAMPLE_BUFFER_SIZE: usize = 88200 * 8; // ~8 seconds at 44.1kHz stereo
pub const PREBUFFER_SIZE: usize = 44100;         // ~0.5 seconds before starting

pub const STATE_STOPPED: u8 = 0;
pub const STATE_PLAYING: u8 = 1;
pub const STATE_PAUSED: u8 = 2;
pub const STATE_BUFFERING: u8 = 3;
pub const STATE_ERROR: u8 = 4;

pub struct Player {
    ring_buffer: Arc<Mutex<RingBuffer>>,
    pub spectrum: Arc<Mutex<SpectrumAnalyzer>>,
    state: Arc<AtomicU8>,
    stop_signal: Arc<AtomicBool>,
    volume: Arc<Mutex<f32>>,
    finished: Arc<AtomicBool>,
    error_msg: Arc<Mutex<Option<String>>>,
    samples_played: Arc<AtomicU32>,
    output_sample_rate: Arc<AtomicU32>,
    total_duration_secs: Arc<AtomicU32>,
    _stream: Option<Stream>,
    _decoder_handle: Option<thread::JoinHandle<()>>,
}

impl Player {
    pub fn new() -> Self {
        Self {
            ring_buffer: Arc::new(Mutex::new(RingBuffer::new(SAMPLE_BUFFER_SIZE))),
            spectrum: Arc::new(Mutex::new(SpectrumAnalyzer::new())),
            state: Arc::new(AtomicU8::new(STATE_STOPPED)),
            stop_signal: Arc::new(AtomicBool::new(false)),
            volume: Arc::new(Mutex::new(0.5)),
            finished: Arc::new(AtomicBool::new(false)),
            error_msg: Arc::new(Mutex::new(None)),
            samples_played: Arc::new(AtomicU32::new(0)),
            output_sample_rate: Arc::new(AtomicU32::new(44100)),
            total_duration_secs: Arc::new(AtomicU32::new(0)),
            _stream: None,
            _decoder_handle: None,
        }
    }

    pub fn play(&mut self, path: &PathBuf) -> Result<()> {
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

        let format = probed.format;
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
            .ok_or_else(|| anyhow::anyhow!("No supported audio track"))?;

        let codec_params = track.codec_params.clone();
        let track_id = track.id;
        let sample_rate = codec_params.sample_rate.unwrap_or(44100);
        let channels = codec_params.channels.map(|c| c.count()).unwrap_or(2);

        let total_duration_secs = codec_params
            .n_frames
            .map(|frames| (frames / sample_rate as u64) as u32)
            .unwrap_or(0);

        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow::anyhow!("No output device"))?;

        let desired_config = StreamConfig {
            channels: channels as u16,
            sample_rate: SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let config = if device
            .supported_output_configs()
            .map(|mut configs| {
                configs.any(|c| {
                    c.channels() == channels as u16
                        && c.min_sample_rate().0 <= sample_rate
                        && c.max_sample_rate().0 >= sample_rate
                })
            })
            .unwrap_or(false)
        {
            desired_config
        } else {
            device
                .default_output_config()
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

        self.output_sample_rate.store(output_sample_rate, Ordering::SeqCst);
        self.total_duration_secs.store(total_duration_secs, Ordering::SeqCst);
        self.samples_played.store(0, Ordering::SeqCst);

        let decoder =
            symphonia::default::get_codecs().make(&codec_params, &DecoderOptions::default())?;

        let ring_buffer = Arc::clone(&self.ring_buffer);
        let stop_signal = Arc::clone(&self.stop_signal);
        let state = Arc::clone(&self.state);
        let finished = Arc::clone(&self.finished);
        let error_msg = Arc::clone(&self.error_msg);

        let decoder_handle = thread::spawn(move || {
            decode_loop(
                format,
                decoder,
                track_id,
                sample_rate,
                output_sample_rate,
                source_channels,
                output_channels,
                ring_buffer,
                stop_signal,
                state,
                finished,
                error_msg,
            );
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
                eprintln!("Audio stream error: {}", err);
            },
            None,
        )?;

        stream.play()?;
        self._stream = Some(stream);

        Ok(())
    }

    pub fn pause(&self) {
        if self.state.load(Ordering::SeqCst) == STATE_PLAYING {
            self.state.store(STATE_PAUSED, Ordering::SeqCst);
        }
    }

    pub fn resume(&self) {
        if self.state.load(Ordering::SeqCst) == STATE_PAUSED {
            self.state.store(STATE_PLAYING, Ordering::SeqCst);
        }
    }

    pub fn stop(&mut self) {
        self.stop_signal.store(true, Ordering::SeqCst);
        self.state.store(STATE_STOPPED, Ordering::SeqCst);

        if let Some(stream) = self._stream.take() {
            drop(stream);
        }

        if let Some(handle) = self._decoder_handle.take() {
            let _ = handle.join();
        }

        self.samples_played.store(0, Ordering::SeqCst);

        if let Ok(mut rb) = self.ring_buffer.lock() {
            rb.clear();
        }
    }

    pub fn is_finished(&self) -> bool {
        if !self.finished.load(Ordering::SeqCst) {
            return false;
        }
        if let Ok(rb) = self.ring_buffer.try_lock() {
            rb.available() == 0
        } else {
            false
        }
    }

    pub fn state(&self) -> u8 {
        self.state.load(Ordering::SeqCst)
    }

    pub fn error(&self) -> Option<String> {
        self.error_msg.lock().ok().and_then(|e| e.clone())
    }

    pub fn buffer_percent(&self) -> u8 {
        if let Ok(rb) = self.ring_buffer.try_lock() {
            ((rb.available() as f32 / SAMPLE_BUFFER_SIZE as f32) * 100.0) as u8
        } else {
            0
        }
    }

    pub fn set_volume(&self, vol: f32) {
        if let Ok(mut v) = self.volume.try_lock() {
            *v = vol.clamp(0.0, 1.0);
        }
    }

    pub fn volume(&self) -> f32 {
        self.volume.try_lock().map(|v| *v).unwrap_or(0.5)
    }

    pub fn playback_time(&self) -> (u32, u32) {
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

impl Default for Player {
    fn default() -> Self {
        Self::new()
    }
}

/// Decoder thread function
fn decode_loop(
    mut format: Box<dyn symphonia::core::formats::FormatReader>,
    mut decoder: Box<dyn symphonia::core::codecs::Decoder>,
    track_id: u32,
    sample_rate: u32,
    output_sample_rate: u32,
    source_channels: usize,
    output_channels: usize,
    ring_buffer: Arc<Mutex<RingBuffer>>,
    stop_signal: Arc<AtomicBool>,
    state: Arc<AtomicU8>,
    finished: Arc<AtomicBool>,
    error_msg: Arc<Mutex<Option<String>>>,
) {
    let mut prebuffered = false;
    let mut decode_errors = 0;

    let needs_resample = sample_rate != output_sample_rate;
    let resample_ratio = output_sample_rate as f64 / sample_rate as f64;

    let mut resampler: Option<SincFixedIn<f32>> = if needs_resample {
        let params = SincInterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.925,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 128,
            window: WindowFunction::Blackman,
        };
        SincFixedIn::new(resample_ratio, 2.0, params, 512, source_channels)
            .or_else(|_| {
                let simple_params = SincInterpolationParameters {
                    sinc_len: 32,
                    f_cutoff: 0.9,
                    interpolation: SincInterpolationType::Nearest,
                    oversampling_factor: 64,
                    window: WindowFunction::Hann,
                };
                SincFixedIn::new(resample_ratio, 2.0, simple_params, 256, source_channels)
            })
            .ok()
    } else {
        None
    };

    if needs_resample && resampler.is_none() {
        if let Ok(mut err) = error_msg.lock() {
            *err = Some(format!(
                "Resampler init failed ({}Hz -> {}Hz)",
                sample_rate, output_sample_rate
            ));
        }
        state.store(STATE_ERROR, Ordering::SeqCst);
        finished.store(true, Ordering::SeqCst);
        return;
    }

    let mut resample_in: Vec<Vec<f32>> = vec![Vec::new(); source_channels];

    loop {
        if stop_signal.load(Ordering::SeqCst) {
            break;
        }

        let current_state = state.load(Ordering::SeqCst);
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
                if err_str.contains("end of stream")
                    || e.kind() == std::io::ErrorKind::UnexpectedEof
                {
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

        // Resample if needed
        let resampled_samples: Vec<f32> = if let Some(ref mut resampler) = resampler {
            for (i, &sample) in interleaved_samples.iter().enumerate() {
                resample_in[i % source_channels].push(sample);
            }

            let chunk_size = resampler.input_frames_max();
            let mut all_output = Vec::new();

            while resample_in[0].len() >= chunk_size {
                let chunk_in: Vec<Vec<f32>> = resample_in
                    .iter_mut()
                    .map(|ch| ch.drain(..chunk_size).collect())
                    .collect();

                if let Ok(resampled) = resampler.process(&chunk_in, None) {
                    if !resampled.is_empty() && !resampled[0].is_empty() {
                        let out_len = resampled[0].len();
                        for i in 0..out_len {
                            for ch in 0..source_channels {
                                all_output.push(resampled[ch][i]);
                            }
                        }
                    }
                }
            }
            all_output
        } else {
            interleaved_samples.to_vec()
        };

        // Channel conversion
        let output_samples: Vec<f32> = if source_channels == output_channels {
            resampled_samples
        } else if source_channels == 1 && output_channels == 2 {
            resampled_samples.iter().flat_map(|&s| [s, s]).collect()
        } else if source_channels == 2 && output_channels == 1 {
            resampled_samples
                .chunks(2)
                .map(|pair| (pair.get(0).unwrap_or(&0.0) + pair.get(1).unwrap_or(&0.0)) * 0.5)
                .collect()
        } else {
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
}

