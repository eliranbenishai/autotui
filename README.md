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
git clone https://github.com/eliranbenishai/autotui
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

## CLI Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `<PATH>` | | Folder, playlist (.json), or audio file to play |
| `--shuffle` | `-s` | Shuffle the playlist or folder of files |
| `--recursive` | `-r` | Scan directories recursively |
| `--write <FILE>` | `-w` | Write playlist to file and exit |
| `--help` | `-h` | Print help information |
| `--version` | `-V` | Print version |

The path type is automatically detected:
- **Directory** → scans for audio files (use `-r` for subdirectories)
- **`.json` file** → loads as playlist
- **Audio file** → plays single track

**Examples:**

```bash
autotui                              # Scan current directory
autotui ~/Music                      # Play tracks in folder
autotui ~/Music -r                   # Play tracks recursively
autotui playlist.json                # Load JSON playlist
autotui song.mp3                     # Play single track
autotui ~/Music -s                   # Play folder shuffled
autotui ~/Music -rs                  # Recursive + shuffled
autotui -r ~/Music -w library.json   # Save recursive scan to playlist
```

## Controls

| Key | Action |
|-----|--------|
| `Space` | Play / Pause |
| `Enter` | Play selected track |
| `s` | Toggle shuffle 🔀 |
| `n` | Next track |
| `p` | Previous track |
| `←` `h` | Select previous |
| `→` `l` | Select next |
| `↑` `+` | Volume up |
| `↓` `-` | Volume down |
| `r` | Rescan current directory |
| `w` | Save playlist to `playlist.json` |
| `q` `Esc` | Quit |

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

## License

MIT
