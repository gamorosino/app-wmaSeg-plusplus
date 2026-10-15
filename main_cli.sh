#! /bin/bash

#########################################################################################################################
#########################################################################################################################
###################                                                                                   ###################
###################     title:             Local launcher — build config.json and run main            ###################
###################                                                                                   ###################
###################     usage:             main_cli.sh --track <path> --freesurfer <path>             ###################
###################                                                                                   ###################
###################     autor: gamorosino                                                             ###################
#########################################################################################################################
#########################################################################################################################

SCRIPT=$(realpath -s "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")

usage() {
    echo "Usage: $(basename $0) --track <path/to/track.tck> --freesurfer <path/to/freesurfer/>"
    exit 1
}

# ── defaults ──────────────────────────────────────────────────────────────────
track=""
freesurfer=""

# ── parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --track)       track=$(realpath "$2");       shift 2 ;;
        --freesurfer)  freesurfer=$(realpath "$2");  shift 2 ;;
        -h|--help)     usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

# ── validate required fields ──────────────────────────────────────────────────
errors=0

if [ -z "$track" ]; then
    echo "Error: --track is required" >&2
    errors=1
elif [ ! -f "$track" ]; then
    echo "Error: --track file not found: $track" >&2
    errors=1
fi

if [ -z "$freesurfer" ]; then
    echo "Error: --freesurfer is required" >&2
    errors=1
elif [ ! -d "$freesurfer" ]; then
    echo "Error: --freesurfer directory not found: $freesurfer" >&2
    errors=1
fi

[ $errors -ne 0 ] && exit 1

# ── write config.json ─────────────────────────────────────────────────────────
cat > config.json <<EOF
{
  "track":      $(jq -n --arg v "$track" '$v'),
  "freesurfer": $(jq -n --arg v "$freesurfer" '$v')
}
EOF

echo "config.json written:"
cat config.json

# ── run main ──────────────────────────────────────────────────────────────────
echo ""
echo "Running: bash ${SCRIPT_DIR}/main"
bash "${SCRIPT_DIR}/main"
