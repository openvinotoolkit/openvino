// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ngraph::opset8;

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateFakeQuantWithMinMaxVarsOp(const NodeContext& node) {
    auto ng_input = node.get_ng_input(0);
    auto ng_min = node.get_ng_input(1);
    auto ng_max = node.get_ng_input(2);

    auto narrow_range = node.get_attribute<bool>("narrow_range");
    auto num_bits = node.get_attribute<int64_t>("num_bits");

    auto levels = std::pow(2, num_bits) - int(narrow_range);
    auto min_less_max = ConstructNgNode<Less>(node.get_name() + "/if_min_less_max", ng_min, ng_max);
    auto minimum = ConstructNgNode<Select>(node.get_name() + "/minimum", min_less_max, ng_min, ng_max);
    auto maximum = ConstructNgNode<Select>(node.get_name() + "/maximum", min_less_max, ng_max, ng_min);

    auto zero = ConstructNgNode<Constant>(node.get_name(), ng_min.get_element_type(), Shape{}, std::vector<int>({0}));

    auto min_greater_zero = ConstructNgNode<Greater>(node.get_name() + "/if_minimum_greater_zero", minimum, zero);
    auto max_minus_min = ConstructNgNode<Subtract>(node.get_name() + "/max_minus_min", maximum, minimum);
    minimum = ConstructNgNode<Select>(node.get_name() + "/first_adj_min", min_greater_zero, zero, minimum);
    maximum = ConstructNgNode<Select>(node.get_name() + "/first_adj_max", min_greater_zero, max_minus_min, maximum);

    auto max_less_zero = ConstructNgNode<Less>(node.get_name() + "/if_max_less_zero", maximum, zero);
    auto min_minus_max = ConstructNgNode<Subtract>(node.get_name() + "/min_minus_max", minimum, maximum);
    minimum = ConstructNgNode<Select>(node.get_name() + "/second_adj_min", max_less_zero, min_minus_max, minimum);
    maximum = ConstructNgNode<Select>(node.get_name() + "/second_adj_max", max_less_zero, zero, maximum);

    auto float_range = ConstructNgNode<Subtract>(node.get_name() + "/float_range", maximum, minimum);
    auto quant_min_value = int(narrow_range);
    auto quant_max_value = std::pow(2, num_bits) - 1;
    auto value = static_cast<float>(quant_max_value - quant_min_value);
    auto int_range =
        ConstructNgNode<Constant>(node.get_name() + "/int_range", element::f32, Shape{}, std::vector<float>({value}));
    auto scale = ConstructNgNode<Divide>(node.get_name() + "/scale", float_range, int_range);
    auto descaled_min = ConstructNgNode<Divide>(node.get_name() + "/descaled_min", minimum, scale);
    auto rounded_descaled_min =
        ConstructNgNode<Round>(node.get_name() + "/rounded_descaled_min", descaled_min, Round::RoundMode::HALF_TO_EVEN);
    auto min_adj = ConstructNgNode<Multiply>(node.get_name() + "/min_adj", scale, rounded_descaled_min);
    auto adjustment = ConstructNgNode<Subtract>(node.get_name() + "/limits_adjustment", min_adj, minimum);
    auto max_adj = ConstructNgNode<Add>(node.get_name() + "/max_adj", maximum, adjustment);

    auto ng_input_shape = ng_input.get_shape();
    if (ng_input_shape.size() == 4)
        Transpose<0, 3, 1, 2>(ng_input);
    auto ng_output =
        ConstructNgNode<FakeQuantize>(node.get_name(), ng_input, min_adj, max_adj, min_adj, max_adj, levels);
    if (ng_input_shape.size() == 4)
        Transpose<0, 2, 3, 1>(ng_output);

    ng_output.get_node_shared_ptr()->set_friendly_name(node.get_name());
    return {ng_output};
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph