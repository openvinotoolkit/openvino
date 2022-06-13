// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/skip_gather_before_transpose_and_reshape.hpp"

#include <memory>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/opsets/opset8.hpp>
#include <vector>

#include "itt.hpp"
#include "transformations/utils/utils.hpp"

ngraph::pass::SkipGatherBeforeTransposeAndReshape::SkipGatherBeforeTransposeAndReshape() {
    MATCHER_SCOPE(SkipGatherBeforeTransposeAndReshape);

    auto input_m = ngraph::pattern::any_input(ngraph::pattern::has_static_dim(0));

    auto indices_m = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto axis_m = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto gather_m = ngraph::pattern::wrap_type<ngraph::op::util::GatherBase>({input_m, indices_m, axis_m});

    auto transpose_const_m = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto transpose_m = ngraph::pattern::wrap_type<ngraph::opset8::Transpose>({gather_m, transpose_const_m});

    auto reshape_const_m = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto reshape_m = ngraph::pattern::wrap_type<ngraph::opset8::Reshape>({transpose_m, reshape_const_m});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& input = pattern_map.at(input_m);
        if (input.get_partial_shape()[0] != 1) {
            return false;
        }

        const auto gather = pattern_map.at(gather_m).get_node_shared_ptr();
        const auto indices = as_type_ptr<ngraph::opset8::Constant>(pattern_map.at(indices_m).get_node_shared_ptr());
        const auto axis = as_type_ptr<ngraph::opset8::Constant>(pattern_map.at(axis_m).get_node_shared_ptr());
        if (!indices || !axis) {
            return false;
        }

        const std::vector<std::int64_t> expected_gather_value{0};
        if (indices->cast_vector<std::int64_t>() != expected_gather_value ||
            axis->cast_vector<std::int64_t>() != expected_gather_value) {
            return false;
        }

        const auto transpose = pattern_map.at(transpose_m).get_node_shared_ptr();
        const auto transpose_const =
            as_type_ptr<ngraph::opset8::Constant>(pattern_map.at(transpose_const_m).get_node_shared_ptr());
        if (!transpose_const) {
            return false;
        }

        const auto reshape_const =
            as_type_ptr<ngraph::opset8::Constant>(pattern_map.at(reshape_const_m).get_node_shared_ptr());
        if (!reshape_const) {
            return false;
        }

        const auto reshape_vals = reshape_const->cast_vector<std::int64_t>();
        if (std::any_of(reshape_vals.begin(), reshape_vals.end(), [](const std::int64_t x) {
                return x == 0;
            })) {
            return false;
        }

        const auto transpose_vals = transpose_const->cast_vector<std::int64_t>();
        std::vector<std::int64_t> new_transpose_vals{0};
        // update the transpose const to compensate for the removal of Gather
        for (auto elem : transpose_vals) {
            new_transpose_vals.push_back(++elem);
        }

        const auto new_transpose_const = ngraph::opset8::Constant::create(transpose_const->get_element_type(),
                                                                          {new_transpose_vals.size()},
                                                                          new_transpose_vals);
        const auto new_transpose = transpose->clone_with_new_inputs({input, new_transpose_const});
        new_transpose->set_friendly_name(transpose->get_friendly_name());
        ngraph::copy_runtime_info({transpose, gather}, new_transpose);
        ngraph::replace_node(transpose, new_transpose);
        MATCHER_SCOPE_ENABLE(SkipGatherBeforeTransposeAndReshape);
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape_m, matcher_name);
    register_matcher(m, callback);
}
