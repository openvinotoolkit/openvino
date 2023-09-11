#!/bin/sh -l

echo "Hello $1"
time=$(date)
echo "user=$(whoami)" >> $GITHUB_OUTPUT
echo "time=$time" >> $GITHUB_OUTPUT