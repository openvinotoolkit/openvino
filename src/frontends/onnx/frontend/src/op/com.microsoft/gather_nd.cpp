// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/com.microsoft/gather_nd.hpp"

#include <memory>

#include "default_opset.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/gather_nd.hpp"
#include "utils/common.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {

OutputVector gather_nd(const Node& node) {
    const OutputVector ng_inputs{node.get_ng_inputs()};
    const auto data = ng_inputs.at(0);
    const auto indices = ng_inputs.at(1);

    const auto indices_int64 = std::make_shared<default_opset::Convert>(indices, ov::element::i64);

    return {std::make_shared<default_opset::GatherND>(data, indices_int64)};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
