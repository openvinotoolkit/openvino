// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/pass/transpose_decomposition.hpp>
#include <snippets/itt.hpp>
#include <snippets/snippets_isa.hpp>
#include <snippets/pass/loop_helpers.hpp>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/partial_shape.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pass/manager.hpp>
#include <numeric>
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
        const auto& data_node = pattern_to_output.at(match_data).get_node_shared_ptr();
        auto &param_rt = data_node->get_rt_info();
        // Note: store and usage inside emitters as size_t is more convenient, so static_cast here
        const auto& access_pattern = order->cast_vector<size_t>();
        param_rt["Layout"] = access_pattern;

        // The line below is Ok, since we ensured that transpose is static above
        auto data_shape = data_input.get_shape();
        // dim indexes with respect to SRC
        const auto dim_C_idx = data_shape.size() - 3;
        const auto dim_H_idx = data_shape.size() - 2;
        const auto dim_W_idx = data_shape.size() - 1;
        const auto size_C = static_cast<int64_t>(data_shape[dim_C_idx]);
        const auto size_W = static_cast<int64_t>(data_shape[dim_W_idx]);
        const auto size_H = static_cast<int64_t>(data_shape[dim_H_idx]);

        auto loop_W_begin = std::make_shared<op::LoopBegin>(OutputVector{data_input});
        auto loop_C_begin = std::make_shared<op::LoopBegin>(OutputVector{loop_W_begin->output(0)});
        // todo: LoadReshape used here is essentially Load + an easy way to maintain correct shape propagation
        //  fix this in future and develop a more consistent shape propagation approach.
        auto load = std::make_shared<snippets::op::LoadReshape>(loop_C_begin->output(0), 1, 0, access_pattern);
        auto store = std::make_shared<snippets::op::Store>(load, 1);
        const std::vector<int64_t> ptr_increments_C {size_H * size_W, 1};
        const std::vector<int64_t> finalization_offsets_C {1 - size_H * size_W * size_C, 0};
        auto loop_C_end = std::make_shared<op::LoopEnd>(OutputVector{store->output(0), loop_C_begin->output(1)},
                                                        size_C, 1, ptr_increments_C, finalization_offsets_C);
        auto loop_W_end = std::make_shared<op::LoopEnd>(OutputVector{loop_C_end->output(0), loop_W_begin->output(1)},
                                                        size_W, 1, std::vector<int64_t>{0, 0}, std::vector<int64_t>{0, 0});

        for (auto& input : transpose->output(0).get_target_inputs()) {
            input.replace_source_output(loop_W_end->output(0));
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(match_transpose, matcher_name);
    register_matcher(m, callback);
}
