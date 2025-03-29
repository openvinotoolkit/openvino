// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_fake_quant_aux_op(const NodeContext& node,
                                         const Output<Node>& inputs,
                                         const Output<Node>& min,
                                         const Output<Node>& max) {
    // retrieve attributes
    auto narrow_range = node.get_attribute<bool>("narrow_range", false);
    auto num_bits = node.get_attribute<int64_t>("num_bits", 8);

    size_t levels = static_cast<size_t>(pow(2, num_bits));
    levels = narrow_range ? levels - 1 : levels;

    // compute real min and max values
    Output<Node> minimum = make_shared<v1::Minimum>(min, max);
    Output<Node> maximum = make_shared<v1::Maximum>(min, max);

    // adjust min and max so that min <= 0
    auto zero = create_same_type_const_scalar<float>(min, 0);
    auto min_greater_zero = make_shared<v1::Greater>(minimum, zero);
    Output<Node> max_minus_min = make_shared<v1::Subtract>(maximum, minimum);
    minimum = make_shared<v1::Select>(min_greater_zero, zero, minimum);
    maximum = make_shared<v1::Select>(min_greater_zero, max_minus_min, maximum);

    // adjust min and max so that 0 <= max
    auto max_less_zero = make_shared<v1::Less>(maximum, zero);
    auto min_minus_max = make_shared<v1::Subtract>(minimum, maximum);
    minimum = make_shared<v1::Select>(max_less_zero, min_minus_max, minimum);
    maximum = make_shared<v1::Select>(max_less_zero, zero, maximum);

    // adjust min and max so that scale = (max - min) / (2^num_bits - 1),
    // min_adj = scale * round(min / scale) and max_adj = max + min_adj - min
    max_minus_min = make_shared<v1::Subtract>(maximum, minimum);
    auto const_levels = make_shared<v0::Constant>(element::f32, Shape{}, static_cast<float>(levels - 1));
    auto scale = make_shared<v1::Divide>(max_minus_min, const_levels);
    auto descaled_min = make_shared<v1::Divide>(minimum, scale);
    auto rounded_descaled_min = make_shared<v5::Round>(descaled_min, v5::Round::RoundMode::HALF_TO_EVEN);
    auto min_adj = make_shared<v1::Multiply>(scale, rounded_descaled_min);
    auto adjustment = make_shared<v1::Subtract>(min_adj, minimum);
    auto max_adj = make_shared<v1::Add>(maximum, adjustment);

    auto fake_quantize = make_shared<v0::FakeQuantize>(inputs, min_adj, max_adj, min_adj, max_adj, levels);
    set_node_name(node.get_name(), fake_quantize);
    return {fake_quantize};
}

OutputVector translate_fake_quant_op(const NodeContext& node) {
    default_op_checks(node, 2, {"FakeQuantWithMinMaxVars", "FakeQuantWithMinMaxVarsPerChannel"});
    auto inputs = node.get_input(0);
    auto min = node.get_input(1);
    auto max = node.get_input(2);

    return translate_fake_quant_aux_op(node, inputs, min, max);
}

OutputVector translate_fake_quant_with_min_max_args(const NodeContext& node) {
    default_op_checks(node, 1, {"FakeQuantWithMinMaxArgs"});
    auto inputs = node.get_input(0);
    auto min_val = node.get_attribute<float>("min", -6.0f);
    auto max_val = node.get_attribute<float>("max", 6.0f);
    auto min = make_shared<v0::Constant>(element::f32, Shape{}, min_val);
    auto max = make_shared<v0::Constant>(element::f32, Shape{}, max_val);

    return translate_fake_quant_aux_op(node, inputs, min, max);
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
