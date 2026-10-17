#!/bin/sh
# Renders the README images into docs/ as JPEG (photo mode: 64 accumulated samples).
cd "$(dirname "$0")/.." || exit 1
mkdir -p docs
shot() {
  node tools/shoot.mjs "?shot=10&accum=64&ui=0&scale=1&$2" "docs/$1.png" --w=1600 --h=900 --wait=60000 | grep -E "page|warning" || true
  ffmpeg -loglevel error -y -i "docs/$1.png" -q:v 3 "docs/$1.jpg" && rm "docs/$1.png"
}
shot hero     "a=1"
shot edge     "cam=40,89.5,4&fov=36"
shot close    "cam=11,86,15&fov=70"
shot redshift "view=1"
shot flight   "fly=14,-14,2.5,135,-6&fov=75"
ls -la docs
