# autotui 🎵

A minimal, single-line terminal music player with real-time spectrum visualization.

```
♪ ▶ ▃▅▇█▆▄▃▅▆█▇▅▃▂▁▃ Adoring Light 2:34/4:12 │ 75% │ 3/12 🔀
```

## Features

- **Single-line UI** — stays out of your way, lives in your terminal
- **Real-time spectrum analyzer** — 16-band FFT visualization
- **Streaming playback** — instant start, handles network latency
- **Format support** — MP3, WAV, FLAC, OGG
- **Network drive support** — plays from NAS, mapped drives, UNC paths
- **Shuffle mode** — toggle on/off during playback
- **Playlist creator** — save current track list to JSON

## Installation

### From Source

```bash
git clone https://github.com/eliranbenishai/autotui
cd autotui
cargo build --release
./target/release/autotui
```

### Pre-built Binaries

Download from [Releases](https://github.com/eliranbenishai/autotui/releases):
- `autotui-macos-arm64.tar.gz` — macOS Apple Silicon
- `autotui-macos-x64.tar.gz` — macOS Intel
- `autotui-linux-x64.tar.gz` — Linux x64
- `autotui-windows-x64.zip` — Windows

## Usage

```bash
autotui                              # Play current directory
autotui ~/Music                      # Play folder
autotui ~/Music -r                   # Play folder recursively
autotui ~/Music -s                   # Play folder shuffled
autotui ~/Music -rs                  # Recursive + shuffled
autotui playlist.json                # Play JSON playlist
autotui song.mp3                     # Play single track
autotui "\\server\share\music"       # Windows network path
autotui /Volumes/NAS/Music           # macOS network mount
```

### Creating Playlists

```bash
# Scan folder and save to playlist (without playing)
autotui ~/Music -w playlist.json

# Scan recursively and save shuffled
autotui ~/Music -rs -w shuffled.json

# Play saved playlist
autotui playlist.json
```

You can also press `w` during playback to save the current track list.

### JSON Playlist Format

```json
{
  "tracks": [
    "/path/to/song1.mp3",
    "/path/to/song2.flac",
    "\\\\server\\share\\song3.mp3"
  ]
}
```

## CLI Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `<PATH>` | | Folder, playlist (.json), or audio file |
| `--shuffle` | `-s` | Shuffle tracks |
| `--recursive` | `-r` | Scan subdirectories |
| `--write <FILE>` | `-w` | Save playlist to file and exit |
| `--help` | `-h` | Show help |
| `--version` | `-V` | Show version |

Path type is auto-detected:
- **Directory** → scans for audio files
- **`.json` file** → loads as playlist
- **Audio file** → plays single track

## Controls

| Key | Action |
|-----|--------|
| `Space` | Play / Pause |
| `Enter` | Play selected track |
| `←` `→` | Previous / Next track |
| `h` `l` | Previous / Next track (vim-style) |
| `↑` `↓` | Volume up / down |
| `+` `-` | Volume up / down |
| `n` | Next track |
| `p` | Previous track |
| `s` | Toggle shuffle 🔀 |
| `r` | Rescan directory |
| `w` | Save playlist to `playlist.json` |
| `q` `Esc` | Quit |

## Building

```bash
# Debug
cargo build && cargo run

# Release
cargo build --release
./target/release/autotui

# Cross-compile for Windows (from macOS)
rustup target add x86_64-pc-windows-gnu
brew install mingw-w64
cargo build --release --target x86_64-pc-windows-gnu
```

## License

MIT
