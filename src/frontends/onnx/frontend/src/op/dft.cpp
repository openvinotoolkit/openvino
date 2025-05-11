// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/dft.hpp"

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "utils/common.hpp"
namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector dft(const ov::frontend::onnx::Node& node) {
    const ov::OutputVector ng_inputs{node.get_ov_inputs()};
    const ov::Output<ov::Node> data = ng_inputs.at(0);

    const auto dft_length_provided = ng_inputs.size() > 1 && !ov::op::util::is_null(ng_inputs[1]);
    const auto axis = node.get_attribute_value<int64_t>("axis", 1);
    const auto inverse = node.get_attribute_value<int64_t>("inverse", 0);
    const auto onesided = node.get_attribute_value<int64_t>("onesided", 0);

    return {dft::make_dft(data,
                          dft_length_provided ? ng_inputs.at(1) : std::make_shared<NullNode>(),
                          axis,
                          inverse == 1,
                          onesided == 1)};
}

ONNX_OP("DFT", OPSET_SINCE(1), ai_onnx::opset_1::dft);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
