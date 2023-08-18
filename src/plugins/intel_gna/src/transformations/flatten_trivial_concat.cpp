// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "flatten_trivial_concat.hpp"

#include <ngraph/pattern/op/wrap_type.hpp>
#include <openvino/cc/ngraph/itt.hpp>
#include <openvino/opsets/opset12.hpp>

using namespace ov::opset12;

namespace ov {
namespace intel_gna {
namespace pass {

FlattenTrivialConcat::FlattenTrivialConcat() {
    MATCHER_SCOPE(FlattenTrivialConcat);
    auto concat = ngraph::pattern::wrap_type<Concat>();

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto concat_node = std::dynamic_pointer_cast<Concat>(pattern_map.at(concat).get_node_shared_ptr());
        if (!concat_node) {
            return false;
        }
        auto concat_inputs = concat_node->inputs();
        auto concat_axis = concat_node->get_concatenation_axis();

        // no need for doing transformation
        if (concat_inputs[0].get_shape().size() <= 2) {
            return false;
        }

        // check if concat can be flattened
        for (int i = 0; i < concat_axis; i++) {
            if (concat_inputs[0].get_shape()[i] != 1) {
                return false;
            }
        }

        // do transformation
        NodeVector reshapes_before_concat;
        for (auto& input : concat_inputs) {
            auto input_dims = input.get_shape();
            auto input_size = static_cast<int32_t>(
                std::accumulate(input_dims.begin(), input_dims.end(), size_t(1), std::multiplies<size_t>()));
            auto shape_pattern =
                std::make_shared<Constant>(element::i32, Shape{2}, std::vector<int32_t>{1, input_size});
            reshapes_before_concat.push_back(
                std::make_shared<Reshape>(input.get_source_output().get_node()->output(0), shape_pattern, false));
        }

        auto flattened_concat_node = std::make_shared<Concat>(reshapes_before_concat, 1);
        auto original_output_shape = concat_node->get_output_shape(0);
        // cast shape vector to i32, because GNA doesn't support u64 constant precision
        std::vector<int32_t> original_output_shape_casted;
        std::transform(original_output_shape.begin(),
                       original_output_shape.end(),
                       std::back_inserter(original_output_shape_casted),
                       [](size_t x) {
                           return static_cast<int32_t>(x);
                       });
        auto output_shape_pattern =
            std::make_shared<Constant>(element::i32, Shape{original_output_shape.size()}, original_output_shape_casted);
        auto reshape_after_concat = std::make_shared<Reshape>(flattened_concat_node, output_shape_pattern, false);
        reshape_after_concat->set_friendly_name(concat_node->get_friendly_name());
        ngraph::replace_node(concat_node, reshape_after_concat);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(concat, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
