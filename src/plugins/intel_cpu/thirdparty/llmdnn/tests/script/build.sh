#!/bin/bash

pip uninstall -y llmdnn
cd ../../../../../../../build/ && make -j 20 llmdnnlib
cd -
cd ext
python setup.py clean --all install
