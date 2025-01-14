// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_constant(const ov::element::Type& type,
                                        const ov::Shape& shape,
                                        InputGenerateData in_data = InputGenerateData(1, 9, 1000, 1));

template <class T = float>
std::shared_ptr<ov::Node> make_constant(const ov::element::Type& type,
                                        const ov::Shape& shape,
                                        const std::vector<T>& data) {
    if (data.empty()) {
        return make_constant(type, shape);
    } else {
        return std::make_shared<ov::op::v0::Constant>(type, shape, data);
    }
}
}  // namespace utils
}  // namespace test
}  // namespace ov
