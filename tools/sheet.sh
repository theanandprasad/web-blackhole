#!/bin/sh
# Contact sheet of 4 renders in a 2x2 grid, each labelled with its parameters.
#   tools/sheet.sh out.png "params1" "params2" "params3" "params4"
# Each params string is a URL query (e.g. "cam=26,81,-12&exposure=1").
cd "$(dirname "$0")/.." || exit 1
out=$1; shift
tmp=$(mktemp -d)
i=0
for q in "$@"; do
  node tools/shoot.mjs "?shot=30&ui=0&scale=1&$q" "$tmp/$i.png" --w=1600 --h=900 | grep -E "page" || true
  i=$((i + 1))
done
FONT=/System/Library/Fonts/Supplemental/Arial.ttf
args=""; filt=""
i=0
for q in "$@"; do
  args="$args -i $tmp/$i.png"
  label=$(printf '%s' "$q" | sed "s/[:']/ /g")
  filt="$filt[$i]scale=800:450,drawtext=fontfile=$FONT:text='$label':x=10:y=10:fontsize=16:fontcolor=white:box=1:boxcolor=black@0.5[v$i];"
  i=$((i + 1))
done
ffmpeg -loglevel error -y $args -filter_complex "${filt}[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0" "$out"
rm -rf "$tmp"
echo "wrote $out"
