#!/bin/bash

func() {
    if $1; then
        python3 -m pip install numpy
    else
        python3 -m pip install openvino==2021.4
    fi
}

if ! func false && func true; then
    echo "Success"
else
    echo "Error"
fi

