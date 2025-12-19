/// Ring buffer for audio sample streaming
pub struct RingBuffer {
    data: Vec<f32>,
    write_pos: usize,
    read_pos: usize,
    len: usize,
}

impl RingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![0.0; capacity],
            write_pos: 0,
            read_pos: 0,
            len: 0,
        }
    }

    pub fn push(&mut self, samples: &[f32]) -> usize {
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

    pub fn pop(&mut self, out: &mut [f32]) -> usize {
        let to_read = out.len().min(self.len);
        let capacity = self.data.len();

        for sample in out.iter_mut().take(to_read) {
            *sample = self.data[self.read_pos];
            self.read_pos = (self.read_pos + 1) % capacity;
        }
        self.len -= to_read;
        to_read
    }

    pub fn available(&self) -> usize {
        self.len
    }

    pub fn clear(&mut self) {
        self.write_pos = 0;
        self.read_pos = 0;
        self.len = 0;
    }
}

