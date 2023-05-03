// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// clang-format off
#include "openvino/openvino.hpp"

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/classification_results.h"
#include "samples/slog.hpp"
#include "format_reader_ptr.h"

#include <openvino/core/preprocess/pre_post_process.hpp>
// clang-format on

int main() {
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model("/home/vitaliy/Downloads/preprocessing_api/v3-small_224_1.0_float.xml");
    ov::preprocess::PrePostProcessor ppp(model);

    ppp.input().tensor().set_element_type(ov::element::u8).set_layout(ov::Layout("HWC")).set_spatial_dynamic_shape();
    ppp.input().model().set_layout("NHWC");

    std::cout << ppp << std::endl;
    
    model = ppp.build();
}