// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/fold_gelu.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov;
using namespace ov::op;
using namespace ov::opset12;

namespace {
using NodePtr = const std::shared_ptr<Node>&;

void fold_gelu(NodePtr input_node, NodePtr output_node) {
    auto gelu = std::make_shared<Gelu>(input_node, GeluApproximationMode::TANH);
    copy_runtime_info(input_node, gelu);
    replace_node(output_node, gelu);
    gelu->set_friendly_name(output_node->get_friendly_name());
}

class HasValue {
public:
    HasValue(float value, float threshold = 0.001) : _value(value), _threshold(threshold) {}
    bool operator()(const ov::Output<Node>& output) const {
        const Constant* const_node = dynamic_cast<const Constant*>(output.get_node());
        if (!const_node)
            return false;
        const auto value = const_node->cast_vector<float>();
        if (value.size() != 1)
            return false;
        return (std::abs(value[0] - _value) < _threshold);
    }

private:
    const float _value;
    const float _threshold;
};
}  // namespace

pass::FoldGelu::FoldGelu() {
    MATCHER_SCOPE(FoldGelu);
    auto input = pattern::any_input();

    auto const1 = pattern::wrap_type<Constant>(HasValue(0.044715));
    auto mul1 = pattern::wrap_type<Multiply>({input, const1}, pattern::consumers_count(1));

    auto mul2 = pattern::wrap_type<Multiply>({mul1, input}, pattern::consumers_count(1));

    auto const2 = pattern::wrap_type<Constant>(HasValue(1.0));
    auto add1 = pattern::wrap_type<Add>({const2, mul2}, pattern::consumers_count(1));

    auto const3 = pattern::wrap_type<Constant>(HasValue(0.797885));
    auto mul3 = pattern::wrap_type<Multiply>({input, const3}, pattern::consumers_count(1));

    auto mul4 = pattern::wrap_type<Multiply>({add1, mul3}, pattern::consumers_count(1));

    auto tan = pattern::wrap_type<Tanh>({mul4}, pattern::consumers_count(1));

    auto const4 = pattern::wrap_type<Constant>(HasValue(1.0));
    auto add2 = pattern::wrap_type<Add>({tan, const4}, pattern::consumers_count(1));

    auto const5 = pattern::wrap_type<Constant>(HasValue(0.5));
    auto mul5 = pattern::wrap_type<Multiply>({input, const5}, pattern::consumers_count(1));

    auto mul6 = pattern::wrap_type<Multiply>({add2, mul5}, pattern::consumers_count(1));

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_map();
        auto input_node = pattern_to_output.at(input);
        auto output_node = pattern_to_output.at(mul6);
        fold_gelu(input_node, output_node);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(mul6, matcher_name);
    this->register_matcher(m, callback);
}
