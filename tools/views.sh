#!/bin/sh
# Renders the standard review views: tools/views.sh <tag> [extra url params]
# Output: shots/<tag>_<view>.png
cd "$(dirname "$0")/.." || exit 1
tag=$1; extra=$2
shoot() { node tools/shoot.mjs "?shot=30&ui=0&scale=1&$2&$extra" "shots/${tag}_$1.png" --w=1600 --h=900 | grep -E "page|warning" || true; }
shoot hero  "a=1"
shoot high  "cam=30,62,35"
shoot close "cam=11,86,15&fov=70"
shoot top   "cam=34,12,0"
shoot edge  "cam=40,89.5,0&fov=36"
