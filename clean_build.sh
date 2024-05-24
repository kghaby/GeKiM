#!/bin/bash
set -e 

# get reqs (may need to check these in requirements.txt, and check their lowest compat python version)
#pipreqs . --force
#pip-missing-reqs .

# find lowest compat python version
#vermin .

# remove previous build artifacts
rm -rf build/ dist/
find . -name '*.egg-info' -exec rm -rf {} +

# build
python -m build

# upload
# twine upload dist/*