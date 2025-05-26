// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/ov_tensor_utils.hpp"
#include "functional_test_utils/common_utils.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/preprocess/color_format.hpp"

namespace ov {
namespace test {
namespace utils {

std::vector<uint8_t> color_test_image(size_t height, size_t width, int b_step, ov::preprocess::ColorFormat format);

using InputsMap = std::map<ov::NodeTypeInfo, std::function<ov::Tensor(
        const std::shared_ptr<ov::Node>& node,
        size_t port,
        const ov::element::Type& elemType,
        const ov::Shape& targetShape,
        std::shared_ptr<InputGenerateData> inGenRangeData)>>;

InputsMap getInputMap();

} // namespace utils
} // namespace test
} // namespace ov
