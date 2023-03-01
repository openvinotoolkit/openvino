// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>

#include "openvino/openvino.hpp"
#include "./tensor_lite.h"
#include "./shape_lite.h"

class Session {
    private:
        ov::CompiledModel model;
    public:
        Session(std::string xml_path, std::string bin_path, ShapeLite* shape, std::string layout);
        TensorLite infer(TensorLite* tensor);
};
