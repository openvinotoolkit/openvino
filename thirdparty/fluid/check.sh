#!/usr/bin/env bash

if [ ! -f modules ] && [ ! -f checksum.txt ]; then
    exit 0
fi

THIS_HASH=$(./checksum.sh)
OLD_HASH=$(cat checksum.txt)

if [ $THIS_HASH != $OLD_HASH ]; then
    echo "Invalid checksum -- any changes were done to the source tree here?"
    exit 1
fi

echo "Check done."
