// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/mod.hpp"

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/floor_mod.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector mod(const ov::frontend::onnx::Node& node) {
    ov::Output<ov::Node> dividend{node.get_ov_inputs().at(0)};
    ov::Output<ov::Node> divisor{node.get_ov_inputs().at(1)};

    std::int64_t fmod = node.get_attribute_value<std::int64_t>("fmod", 0);
    ov::OutputVector output;
    if (fmod == 1) {
        output = {std::make_shared<v1::Mod>(dividend, divisor)};
    } else if (fmod == 0) {
        FRONT_END_GENERAL_CHECK(dividend.get_element_type().is_integral() && divisor.get_element_type().is_integral(),
                                "If the input type is floating point, then `fmod` attribute "
                                "must be set to 1.");
        output = {std::make_shared<v1::FloorMod>(dividend, divisor)};
    } else {
        OPENVINO_THROW("Unsupported value of 'fmod' attribute (should be: 0 or 1)");
    }
    return output;
}

ONNX_OP("Mod", OPSET_SINCE(1), ai_onnx::opset_1::mod);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
