// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#endif

#include <memory>
#include <vector>

#include "common_test_utils/test_enums.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ngraph {
namespace helpers {
std::vector<std::pair<ov::element::Type, std::vector<std::uint8_t>>> interpreterFunction(
    const std::shared_ptr<ov::Model>& function,
    const std::vector<std::vector<std::uint8_t>>& inputs,
    const std::vector<ov::element::Type>& inputTypes = {});

std::vector<ov::Tensor> interpretFunction(const std::shared_ptr<ov::Model>& function,
                                          const std::map<std::shared_ptr<ov::Node>, ov::Tensor>& inputs);

}  // namespace helpers
}  // namespace ngraph
