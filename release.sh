#!/bin/bash
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get version from argument or prompt
VERSION=${1:-}

if [ -z "$VERSION" ]; then
    # Show current tags
    echo -e "${YELLOW}Recent tags:${NC}"
    git tag --sort=-version:refname | head -5
    echo ""
    read -p "Enter new version (e.g., 0.1.0): " VERSION
fi

# Ensure version starts with 'v'
if [[ ! "$VERSION" =~ ^v ]]; then
    VERSION="v$VERSION"
fi

echo -e "${YELLOW}Preparing release ${VERSION}...${NC}"

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo -e "${RED}Error: You have uncommitted changes. Please commit or stash them first.${NC}"
    exit 1
fi

# Check if tag already exists
if git rev-parse "$VERSION" >/dev/null 2>&1; then
    echo -e "${RED}Error: Tag ${VERSION} already exists.${NC}"
    echo "To delete and recreate: git tag -d ${VERSION} && git push origin :refs/tags/${VERSION}"
    exit 1
fi

# Update version in Cargo.toml
CARGO_VERSION="${VERSION#v}"
sed -i '' "s/^version = \".*\"/version = \"${CARGO_VERSION}\"/" Cargo.toml
echo -e "${GREEN}✓ Updated Cargo.toml to version ${CARGO_VERSION}${NC}"

# Commit version bump
git add Cargo.toml
git commit -m "Release ${VERSION}"
echo -e "${GREEN}✓ Committed version bump${NC}"

# Create and push tag
git tag -a "$VERSION" -m "Release ${VERSION}"
echo -e "${GREEN}✓ Created tag ${VERSION}${NC}"

# Push commit and tag
git push origin main
git push origin "$VERSION"
echo -e "${GREEN}✓ Pushed to origin${NC}"

echo ""
echo -e "${GREEN}🎉 Release ${VERSION} created!${NC}"
echo -e "GitHub Actions will now build and publish the release."
echo -e "Watch progress: ${YELLOW}https://github.com/$(git remote get-url origin | sed 's/.*github.com[:/]\(.*\)\.git/\1/')/actions${NC}"

