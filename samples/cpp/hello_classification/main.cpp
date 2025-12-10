// Copyright (C) 2018-2025 Intel Corporation
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

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/add.hpp"
// clang-format on

/**
 * @brief Main with support Unicode paths, wide strings
 */
int tmain(int argc, tchar* argv[]) {
    // auto constant = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{3}, std::vector<uint8_t>{1, 2, 3});
    // auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::Shape{3});
    // auto add = std::make_shared<ov::op::v1::Add>(constant, parameter);
    // auto model = std::make_shared<ov::Model>(add, ov::ParameterVector{parameter}, "simple_add");
    // ov::pass::Serialize("simple_add.xml", "simple_add.bin").run_on_model(model);


    ov::Core core;
    ov::AnyMap config = {ov::enable_mmap(false)};
    auto compiled_model = core.compile_model("simple_add.xml", "GPU", config);
    return EXIT_SUCCESS;
}
