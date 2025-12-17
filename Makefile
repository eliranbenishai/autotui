.PHONY: build release run clean

# Build debug version
build:
	cargo build

# Build release version
release-build:
	cargo build --release

# Run the player
run:
	cargo run

# Clean build artifacts
clean:
	cargo clean

# Create a new release (usage: make release V=0.2.0)
release:
	@if [ -z "$(V)" ]; then \
		./release.sh; \
	else \
		./release.sh $(V); \
	fi

# Show help
help:
	@echo "Available commands:"
	@echo "  make build         - Build debug version"
	@echo "  make release-build - Build optimized release"
	@echo "  make run           - Run the player"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make release V=X.Y.Z - Create and push a new release"

