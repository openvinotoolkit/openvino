// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/stft.hpp"

#include "default_opset.hpp"
#include "onnx_import/core/null_node.hpp"
#include "utils/common.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_17 {

OutputVector stft(const Node& node) {
    const auto tensor_iterator = std::make_shared<default_opset::TensorIterator>();
    const auto signal_param = std::make_shared<default_opset::Parameter>();

    // tensor_iterator->set_sliced_input()
    return {};
}

}  // namespace set_17

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
