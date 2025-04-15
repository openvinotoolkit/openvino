// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/onnx/frontend.hpp"

inline void test_load() {
    ov::frontend::onnx::FrontEnd fe;
    fe.get_name();
}
