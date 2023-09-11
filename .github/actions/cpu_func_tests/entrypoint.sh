#!/bin/sh -l

echo "Hello $1"
time=$(date)
echo $(whoami)
echo $(ls -la ${GITHUB_OUTPUT})
echo "time=$time" >> $GITHUB_OUTPUT