// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/frontend/paddle/frontend.hpp>

inline void test_load() {
    ov::frontend::paddle::FrontEnd fe;
    fe.get_name();
}
