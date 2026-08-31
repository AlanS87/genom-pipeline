#!/usr/bin/env bash
#
# Fetches the official LogMap standalone distribution (Apache License 2.0,
# https://github.com/ernestojimenezruiz/logmap-matcher) instead of vendoring
# the jar in this git repository.
#
# Why not commit the jar to git:
#   - it's a large third-party binary (tens of MB) that git can't diff --
#     every future change to it doubles repo history size forever
#   - it has its own release cycle; pinning a download URL keeps the exact
#     version explicit and reproducible without bloating this repo
#   - the license permits redistribution, but the Python-ecosystem convention
#     for third-party binaries (same as HF model weights) is "fetch on setup",
#     not "vendor in source control"
#
# Usage:
#   scripts/download_logmap.sh [destination_dir] [asset_url]
#
#   destination_dir  default: .cache/logmap  (already covered by .gitignore)
#   asset_url        default: the July 2021 standalone release below
#
# NOTE: the default ASSET_URL below points at the "logmap-matcher-july-2021"
# release tag, which matches the standalone distribution this repo was
# developed against. GitHub's release-asset listing is JS-rendered and could
# not be resolved automatically while writing this script -- if the download
# 404s, open the release page, right-click the standalone zip/jar asset,
# "copy link", and pass it as the second argument (or edit ASSET_URL below):
#   https://github.com/ernestojimenezruiz/logmap-matcher/releases/tag/logmap-matcher-july-2021

set -euo pipefail

RELEASE_PAGE="https://github.com/ernestojimenezruiz/logmap-matcher/releases/tag/logmap-matcher-july-2021"
ASSET_URL="${2:-}"
DEST_DIR="${1:-.cache/logmap}"

mkdir -p "$DEST_DIR"

if [[ -z "$ASSET_URL" ]]; then
    echo "No asset URL given/hard-coded yet." >&2
    echo "Open $RELEASE_PAGE, copy the standalone distribution asset's download link, and re-run:" >&2
    echo "  scripts/download_logmap.sh $DEST_DIR <asset_url>" >&2
    exit 1
fi

ARCHIVE_NAME="$(basename "$ASSET_URL")"
ARCHIVE_PATH="$DEST_DIR/$ARCHIVE_NAME"

echo "Downloading $ASSET_URL -> $ARCHIVE_PATH ..."
curl -fL "$ASSET_URL" -o "$ARCHIVE_PATH"

if [[ "$ARCHIVE_NAME" == *.zip ]]; then
    echo "Extracting $ARCHIVE_PATH ..."
    unzip -o "$ARCHIVE_PATH" -d "$DEST_DIR"
fi

JAR_PATH="$(find "$DEST_DIR" -iname 'logmap-matcher*.jar' | head -n 1 || true)"

if [[ -n "$JAR_PATH" ]]; then
    echo ""
    echo "Done. Found jar at: $JAR_PATH"
    echo "Set it before running the pipeline:"
    echo "  export LOGMAP_JAR_PATH=\"$(cd "$(dirname "$JAR_PATH")" && pwd)/$(basename "$JAR_PATH")\""
else
    echo ""
    echo "Downloaded $ARCHIVE_PATH but could not find a logmap-matcher*.jar inside $DEST_DIR."
    echo "Check the archive contents and set LOGMAP_JAR_PATH to the jar's path manually."
fi
