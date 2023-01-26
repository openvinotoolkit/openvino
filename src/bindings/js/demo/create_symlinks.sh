#!/bin/bash

ln -svf ../../../../../bin/ia32/Release/openvino_wasm.js ./dist
ln -svf ../../../../../bin/ia32/Release/openvino_wasm.wasm ./dist

ln -svf ../dist ./browser
ln -svf ../assets ./browser
