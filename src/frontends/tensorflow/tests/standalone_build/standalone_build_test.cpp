// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/frontend/tensorflow/frontend.hpp>

inline void test_load() {
    ov::frontend::tensorflow::FrontEnd fe;
    fe.get_name();
}
