use rustfft::{num_complex::Complex, FftPlanner};

pub const NUM_BARS: usize = 16;
const FFT_SIZE: usize = 1024;
const BAR_CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// Real-time spectrum analyzer using FFT
pub struct SpectrumAnalyzer {
    bands: [f32; NUM_BARS],
    smoothed: [f32; NUM_BARS],
    sample_buffer: Vec<f32>,
    fft_planner: FftPlanner<f32>,
}

impl SpectrumAnalyzer {
    pub fn new() -> Self {
        Self {
            bands: [0.0; NUM_BARS],
            smoothed: [0.0; NUM_BARS],
            sample_buffer: Vec::with_capacity(FFT_SIZE),
            fft_planner: FftPlanner::new(),
        }
    }

    pub fn push_samples(&mut self, samples: &[f32]) {
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

    pub fn update_smooth(&mut self, dt: f32, playing: bool) {
        self.analyze();

        for i in 0..NUM_BARS {
            let target = if playing { self.bands[i] } else { 0.0 };
            let speed = if playing { 20.0 } else { 10.0 };
            self.smoothed[i] += (target - self.smoothed[i]) * speed * dt;
            self.smoothed[i] = self.smoothed[i].clamp(0.0, 1.0);
        }
    }

    pub fn render(&self) -> String {
        self.smoothed
            .iter()
            .map(|&v| {
                let idx = ((v * 7.0) as usize).min(7);
                BAR_CHARS[idx]
            })
            .collect()
    }

    pub fn clear(&mut self) {
        self.sample_buffer.clear();
        self.bands = [0.0; NUM_BARS];
    }
}

impl Default for SpectrumAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

