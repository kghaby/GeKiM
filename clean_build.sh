#!/bin/bash
set -e 

# remove previous build artifacts
rm -rf build/ dist/
find . -name '*.egg-info' -exec rm -rf {} +

# build
python -m build