# OpenCV G-API (Fluid), standalone edition

This subtree hosts sources of G-API - a new OpenCV module for
efficient image processing. G-API serves as a preprocessing vehicle
for Inference Engine. At the moment, only Fluid (CPU) backend is used.

The sources are taken from OpenCV's [main repository](https://github.com/opencv).

There are supplementary scripts which ease and verify the update
process.

## Usage

Updating to the latest `master`:

    ./update.sh

Updating to a particular revision:

    ./update.sh COMMIT_HASH

During update, this script checks if the source tree was modified
after the latest update. If it was, update fails -- we want to avoid
any diverge in the source so _no changes_ should be committed ever to
this copy of G-API.

One can check manually if sources were diverged from its last "valid"
copy by running

    ./check.sh

An error message and non-zero exit code indicate possible inconsitency
with this source copy.

One updated, all changes will be automatically staged.

## Files

In addition to the source tree, the above two scripts maintain two
files:
- `revision.txt` -- the OpenCV's revision used to produce this source
  copy. If the code was taken from `master`, a timestamp is stored
  otherwise.
- `checksum.txt` -- latest valid copy's check sum. Don't update this
  file manually.
