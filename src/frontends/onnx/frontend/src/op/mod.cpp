// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/mod.hpp"

#include <memory>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "op/mod.hpp"

namespace ov {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector mod(const Node& node) {
    Output<ov::Node> dividend{node.get_ng_inputs().at(0)};
    Output<ov::Node> divisor{node.get_ng_inputs().at(1)};

    std::int64_t fmod = node.get_attribute_value<std::int64_t>("fmod", 0);
    OutputVector output;
    if (fmod == 1) {
        output = {std::make_shared<default_opset::Mod>(dividend, divisor)};
    } else if (fmod == 0) {
        OPENVINO_ASSERT(dividend.get_element_type().is_integral() && divisor.get_element_type().is_integral(),
                     "If the input type is floating point, then `fmod` attribute "
                     "must be set to 1.");
        output = {std::make_shared<default_opset::FloorMod>(dividend, divisor)};
    } else {
        throw ov::Exception("Unsupported value of 'fmod' attribute (should be: 0 or 1)");
    }
    return output;
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ov
