#!/usr/bin/env bash

REVISION=""
TARGET_DIR=$(pwd)

case "$#" in
    "0") echo "Using latest master..."
         REVISION="master"
         ;;
    "1") REVISION=$1
         echo "Using revision ${REVISION}..."
         ;;
    *) echo "Usage: ${0} [REVISION]

    Update Fluid to OpenCV source tree at the given REVISION.
    If no revision specified, the most recent 'master' commit is used.
"
       exit 1 ;;
esac

# Before doing anything, check if this snapshot was not modified
./check.sh
if [ $? -ne 0 ]; then
    echo "Consistency check failed, please reset this subtree to its initial state first!"
    exit 1
fi

# Download the stuff...
URL="https://github.com/opencv/opencv/archive/${REVISION}.zip"
wget -c ${URL}
if [ $? -ne 0 ]; then
    echo "Failed to download ${URL}!"
    exit 1
fi

unzip -qq ${REVISION}.zip

# Remove current files
if [ -f modules ]; then
    find modules -type f | xargs git rm
    find modules -type f | xargs rm
    rm -vd modules
fi

# Put a new copy. Extend this section if needed
# BOM thing might help here, probably
pushd "opencv-${REVISION}"
cp -rv --parent modules/gapi ${TARGET_DIR}
popd
# Note "-f" is used to add files like doc/ which are omitted
# now by IE's current .gitignore - it breaks checksum otherwise.
git add -f modules/gapi

# Clean-up files
rm -rf "opencv-${REVISION}"
rm "${REVISION}.zip"

# Calculate and store checksum
./checksum.sh > checksum.txt
git add checksum.txt

# Store revision
if [ ${REVISION} == "master" ]; then
    REVISION="${REVISION} / $(date +%F)"
fi
echo ${REVISION} > revision.txt
git add revision.txt

# Display status
git status

# Fin
echo "Done!"
