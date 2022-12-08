// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_fake_quant_op(const NodeContext& node) {
    default_op_checks(node, 2, {"FakeQuantWithMinMaxVars", "FakeQuantWithMinMaxVarsPerChannel"});
    auto inputs = node.get_input(0);
    auto min = node.get_input(1);
    auto max = node.get_input(2);

    auto narrow_range = node.get_attribute<bool>("narrow_range");
    auto num_bits = node.get_attribute<int64_t>("num_bits");

    size_t levels = static_cast<size_t>(std::pow(2, num_bits) - int(narrow_range));
    auto min_less_max = make_shared<Less>(min, max);
    auto minimum = make_shared<Select>(min_less_max, min, max);
    auto maximum = make_shared<Select>(min_less_max, max, min);

    auto zero = make_shared<Constant>(min.get_element_type(), Shape{}, std::vector<int>({0}));

    auto min_greater_zero = make_shared<Greater>(minimum, zero);
    auto max_minus_min = make_shared<Subtract>(maximum, minimum);
    minimum = make_shared<Select>(min_greater_zero, zero, minimum);
    maximum = make_shared<Select>(min_greater_zero, max_minus_min, maximum);

    auto max_less_zero = make_shared<Less>(maximum, zero);
    auto min_minus_max = make_shared<Subtract>(minimum, maximum);
    minimum = make_shared<Select>(max_less_zero, min_minus_max, minimum);
    maximum = make_shared<Select>(max_less_zero, zero, maximum);

    auto float_range = make_shared<Subtract>(maximum, minimum);
    auto quant_min_value = int(narrow_range);
    auto quant_max_value = std::pow(2, num_bits) - 1;
    auto value = static_cast<float>(quant_max_value - quant_min_value);
    auto int_range = make_shared<Constant>(element::f32, Shape{}, std::vector<float>({value}));
    auto scale = make_shared<Divide>(float_range, int_range);
    auto descaled_min = make_shared<Divide>(minimum, scale);
    auto rounded_descaled_min = make_shared<Round>(descaled_min, Round::RoundMode::HALF_TO_EVEN);
    auto min_adj = make_shared<Multiply>(scale, rounded_descaled_min);
    auto adjustment = make_shared<Subtract>(min_adj, minimum);
    auto max_adj = make_shared<Add>(maximum, adjustment);

    auto res = make_shared<FakeQuantize>(inputs, min_adj, max_adj, min_adj, max_adj, levels);
    set_node_name(node.get_name(), res);
    return {res};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
