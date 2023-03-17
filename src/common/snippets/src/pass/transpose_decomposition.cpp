// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/pass/transpose_decomposition.hpp>
#include <snippets/itt.hpp>
#include <snippets/snippets_isa.hpp>
#include <snippets/tensor_descriptor.hpp>
#include <ngraph/partial_shape.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pass/manager.hpp>
const std::set<std::vector<int>> ngraph::snippets::pass::TransposeDecomposition::supported_cases = {{0, 2, 3, 1}};
ngraph::snippets::pass::TransposeDecomposition::TransposeDecomposition() {
    MATCHER_SCOPE(TransposeDecomposition);
    // todo: we need a special transformation that detects and propagates data access pattern to Parameters and Results
    //  this is needed to communicate access pattern to the plugin node and op::Kernel
    // This is the reason we match only to Parameter, this limitation could be relaxed if we propagate access pattern
    // to the appropriate parameter
    auto match_data = ngraph::pattern::wrap_type<opset1::Parameter>();
    auto match_order = ngraph::pattern::wrap_type<opset1::Constant>();
    auto match_transpose = ngraph::pattern::wrap_type<ngraph::opset1::Transpose>({match_data, match_order});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::TransposeDecomposition")
        auto& pattern_to_output = m.get_pattern_value_map();
        const auto transpose = ov::as_type_ptr<ngraph::opset1::Transpose>(
                                                            pattern_to_output.at(match_transpose).get_node_shared_ptr());

        const auto order = ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(match_order).get_node_shared_ptr());
        if (transformation_callback(transpose) || transpose->is_dynamic())
            return false;

        auto order_value = order->cast_vector<int>();
        if (supported_cases.count(order_value) == 0)
            return false;

        auto data_input = pattern_to_output.at(match_data);
        const std::vector<size_t>& tensor_shape {data_input.get_shape()};
        // number of elements that can be processed on every iteration. For 0,1,2,3 -> 0,2,3,1 we can guarantee only scalar access
        const std::vector<size_t> subtensor_shape {1};
        const auto& layout = order->cast_vector<size_t>();
        // todo: LoadReshape used here is essentially Load + an easy way to maintain correct shape propagation
        //  fix this in future and develop a more consistent shape propagation approach.
        auto load = std::make_shared<snippets::op::LoadReshape>(data_input, subtensor_shape[0], 0, layout);
        auto store = std::make_shared<snippets::op::Store>(load, subtensor_shape[0]);
        ngraph::snippets::set_tensor_descriptor_ptr(load->output(0), std::make_shared<TensorDescriptor>(tensor_shape, subtensor_shape, layout));
        ngraph::snippets::set_tensor_descriptor_ptr(store->output(0),
                                                    std::make_shared<TensorDescriptor>(store->get_output_shape(0),
                                                                                             std::vector<size_t>{},
                                                                                             std::vector<size_t>{}));

        for (auto& input : transpose->output(0).get_target_inputs()) {
            input.replace_source_output(store->output(0));
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(match_transpose, matcher_name);
    register_matcher(m, callback);
}
