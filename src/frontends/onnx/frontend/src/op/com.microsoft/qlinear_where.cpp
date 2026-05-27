// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/decompositions/low_precision_dequantize.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/select.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace opset_1 {

ov::OutputVector qlinear_where(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 9);

    auto condition = node.get_ov_inputs().at(0);
    auto x = node.get_ov_inputs().at(1);
    auto x_scale = node.get_ov_inputs().at(2);
    auto x_zero_point = node.get_ov_inputs().at(3);
    auto y = node.get_ov_inputs().at(4);
    auto y_scale = node.get_ov_inputs().at(5);
    auto y_zero_point = node.get_ov_inputs().at(6);
    auto z_scale = node.get_ov_inputs().at(7);
    auto z_zero_point = node.get_ov_inputs().at(8);

    auto x_dequant = ov::decomposition::low_precision_dequantize(x, x_scale, x_zero_point);
    auto y_dequant = ov::decomposition::low_precision_dequantize(y, y_scale, y_zero_point);

    auto selected = std::make_shared<v1::Select>(condition, x_dequant, y_dequant);

    auto scaled_result = std::make_shared<v1::Divide>(selected, z_scale);
    auto quantise_result = std::make_shared<v0::Convert>(scaled_result, x.get_element_type());
    auto final_output = std::make_shared<v1::Add>(quantise_result, z_zero_point);

    return {final_output};
}

ONNX_OP("QLinearWhere", OPSET_SINCE(1), com_microsoft::opset_1::qlinear_where, MICROSOFT_DOMAIN);

}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
