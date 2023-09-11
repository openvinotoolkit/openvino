#!/bin/sh -l

echo "Hello $1"
time=$(date)
echo $(whoami)
echo "time=$time" >> $GITHUB_OUTPUT