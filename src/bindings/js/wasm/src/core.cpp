// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdio.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#ifdef __EMSCRIPTEN__
#    include <emscripten/bind.h>
#endif

#include "../include/session.h"
#include "../include/shape_lite.h"
#include "../include/tensor_lite.h"
#include "openvino/openvino.hpp"

#ifdef __EMSCRIPTEN__
using namespace emscripten;
#endif

std::string getVersionString() {
    ov::Version version = ov::get_openvino_version();
    std::string str;

    return str.assign(version.buildNumber);
}

std::string getDescriptionString() {
    ov::Version version = ov::get_openvino_version();
    std::string str;

    return str.assign(version.description);
}

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_BINDINGS(my_module) {
    function("getVersionString", &getVersionString);
    function("getDescriptionString", &getDescriptionString);

    class_<Session>("Session").constructor<std::string, std::string, ShapeLite*, std::string>().function(
        "infer",
        &Session::infer,
        allow_raw_pointers());

    class_<ShapeLite>("Shape")
        .constructor<uintptr_t, int>()
        .function("getDim", &ShapeLite::get_dim)
        .function("shapeSize", &ShapeLite::shape_size)
        .function("getData", &ShapeLite::get_data);

    class_<TensorLite>("Tensor")
        .constructor<std::string, uintptr_t, ShapeLite*>()
        .function("getShape", &TensorLite::get_shape, allow_raw_pointers())
        .function("getData", &TensorLite::get_data)
        .function("getPrecision", &TensorLite::get_precision);
}
#endif
