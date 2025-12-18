# autotui 🎵

A minimal, single-line terminal music player with real-time spectrum visualization.

```
♪ ▶ ▃▅▇█▆▄▃▅▆█▇▅▃▂▁▃ Adoring Light, Coveting the Dark │ 50% │ 1/5
```

## Features

- **Single-line UI** — stays out of your way, lives in your terminal
- **Real-time spectrum analyzer** — 16-band FFT visualization that responds to actual audio frequencies
- **Streaming playback** — instant start, no loading delay
- **Format support** — MP3, WAV, FLAC, OGG
- **Network drive support** — play from mapped drives, NAS, or UNC paths
- **JSON playlists** — load custom playlists from JSON files

## Installation

### From Source

```bash
git clone https://github.com/yourusername/autotui
cd autotui
cargo build --release
./target/release/autotui
```

### Pre-built Binaries

Download from [Releases](https://github.com/yourusername/autotui/releases):
- `autotui-macos-arm64.tar.gz` — macOS Apple Silicon
- `autotui-macos-x64.tar.gz` — macOS Intel
- `autotui-linux-x64.tar.gz` — Linux x64
- `autotui-windows-x64.zip` — Windows

## Usage

Run `autotui` in a directory containing audio files. It will automatically scan for tracks and start playing.

```bash
# Play from current directory
autotui

# Play from a specific folder (supports network paths)
autotui --folder ~/Music
autotui -f "\\server\share\music"      # Windows network path
autotui -f "/Volumes/NAS/Music"         # macOS network mount

# Play from a JSON playlist
autotui --playlist my_playlist.json
autotui -p party_mix.json
```

### JSON Playlist Format

Create a `.json` file with your track list:

```json
{
  "name": "My Playlist",
  "tracks": [
    "/path/to/song1.mp3",
    "/path/to/song2.flac",
    { "path": "\\\\server\\share\\song3.mp3" }
  ]
}
```

Tracks can be simple path strings or objects with a `path` field. Both local and network paths are supported.

## Controls

| Key | Action |
|-----|--------|
| `Space` | Play / Pause |
| `Enter` | Play selected track |
| `s` | Stop |
| `n` | Next track |
| `p` | Previous track |
| `←` `h` | Select previous |
| `→` `l` | Select next |
| `↑` `+` | Volume up |
| `↓` `-` | Volume down |
| `o` | Rescan current directory |
| `q` `Esc` | Quit |

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  symphonia  │───▶│ Ring Buffer │───▶│    cpal     │
│  (decoder)  │    │  (samples)  │    │  (output)   │
└─────────────┘    └──────┬──────┘    └─────────────┘
                         │
                         ▼
                  ┌─────────────┐
                  │   rustfft   │
                  │ (spectrum)  │
                  └─────────────┘
```

- **symphonia** — Decodes MP3/FLAC/WAV/OGG in a background thread
- **cpal** — Low-level audio output with direct sample access
- **rustfft** — 1024-point FFT with Hann windowing for spectrum analysis
- **Ring buffer** — Lock-free sample transfer between decoder and audio callback

## Building

### Debug
```bash
cargo build
cargo run
```

### Release
```bash
cargo build --release
./target/release/autotui
```

### Cross-compile for Windows (from macOS)
```bash
rustup target add x86_64-pc-windows-gnu
brew install mingw-w64
cargo build --release --target x86_64-pc-windows-gnu
```

## Releasing

Create a new release with one command:

```bash
./release.sh 0.2.0
# or
make release V=0.2.0
```

This will:
1. Update version in `Cargo.toml`
2. Commit the version bump
3. Create and push a git tag
4. Trigger GitHub Actions to build binaries for all platforms

## Dependencies

| Crate | Purpose |
|-------|---------|
| [symphonia](https://github.com/pdeljanov/Symphonia) | Audio decoding |
| [cpal](https://github.com/RustAudio/cpal) | Cross-platform audio output |
| [rustfft](https://github.com/ejmahler/RustFFT) | FFT for spectrum analysis |
| [crossterm](https://github.com/crossterm-rs/crossterm) | Terminal manipulation |
| [clap](https://github.com/clap-rs/clap) | CLI argument parsing |
| [serde](https://github.com/serde-rs/serde) | JSON playlist support |
| [walkdir](https://github.com/BurntSushi/walkdir) | Directory traversal |
| [anyhow](https://github.com/dtolnay/anyhow) | Error handling |

## License

MIT
