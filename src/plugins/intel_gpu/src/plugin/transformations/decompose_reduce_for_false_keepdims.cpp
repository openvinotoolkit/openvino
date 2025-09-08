// Copyright (C) 2018-2025 Intel Corporationc
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "transformations/utils/utils.hpp"

#include "decompose_reduce_for_false_keepdims.hpp"

#include <algorithm>
#include <cassert>
#include <memory>
#include <vector>

namespace ov::intel_gpu {

DecomposeReduceForFalseKeepDims::DecomposeReduceForFalseKeepDims() {
    // Get one MatcherPass for all modes
    auto reduce_pattern = ov::pass::pattern::wrap_type<ov::op::v1::ReduceSum,
                                                       ov::op::v1::ReduceMean,
                                                       ov::op::v1::ReduceProd,
                                                       ov::op::v1::ReduceMin,
                                                       ov::op::v1::ReduceMax>(
        {ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape()),
         ov::pass::pattern::wrap_type<ov::op::v0::Constant>()},
        ov::pass::pattern::has_static_shape());

    // register callback
    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto reduce =
            as_type_ptr<op::util::ArithmeticReductionKeepDims>(pattern_map.at(reduce_pattern).get_node_shared_ptr());
        if (!reduce)
            return false;

        auto input = reduce->input_value(0);
        const auto input_shape = input.get_shape();
        const auto reduce_shape = reduce->output(0).get_shape();
        const auto input_rank = input.get_partial_shape().rank().get_length();

        auto axes_vector = reduce->get_reduction_axes().to_vector();
        std::sort(axes_vector.begin(), axes_vector.end());

        if (!reduce->get_keep_dims() &&
            need_transformation_for_reordered_axes(axes_vector, input_rank, (input_rank - 2)) &&
            input_shape.size() < 6) {
            ov::NodeVector new_ops;

            // Reduce
            auto reduce_const =
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{axes_vector.size()}, axes_vector);

            // Add each reduce mode supported by oneDNN
            if (ov::is_type<ov::op::v1::ReduceSum>(reduce))
                input = std::make_shared<ov::op::v1::ReduceSum>(input, reduce_const, true);
            else if (ov::is_type<ov::op::v1::ReduceMean>(reduce))
                input = std::make_shared<ov::op::v1::ReduceMean>(input, reduce_const, true);
            else if (ov::is_type<ov::op::v1::ReduceMin>(reduce))
                input = std::make_shared<ov::op::v1::ReduceMin>(input, reduce_const, true);
            else if (ov::is_type<ov::op::v1::ReduceMax>(reduce))
                input = std::make_shared<ov::op::v1::ReduceMax>(input, reduce_const, true);
            else if (ov::is_type<ov::op::v1::ReduceProd>(reduce))
                input = std::make_shared<ov::op::v1::ReduceProd>(input, reduce_const, true);
            else
                return false;

            input.get_node_shared_ptr()->set_friendly_name(reduce->get_friendly_name());
            new_ops.push_back(input.get_node_shared_ptr());

            // Reshape
            auto reshape_shape = ov::Shape((input_rank - axes_vector.size()), 1);
            // Expected that a feature axis is only un-reduced unless a new case for this decomposition is added.
            assert(reshape_shape.size() == 1);
            reshape_shape[0] = reduce_shape[0];
            input = std::make_shared<ov::op::v1::Reshape>(
                input,
                ov::op::v0::Constant::create(ov::element::i64,
                                              ov::Shape{reshape_shape.size()},
                                              reshape_shape),
                false);

            input.get_node_shared_ptr()->set_friendly_name(reduce->get_friendly_name() + "_reshape_false_keepdims");
            new_ops.push_back(input.get_node_shared_ptr());

            ov::copy_runtime_info(reduce, new_ops);
            reduce->output(0).replace(input);
            return true;
        }

        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(reduce_pattern, "DecomposeReduceForFalseKeepDims");
    register_matcher(m, callback);
}

bool DecomposeReduceForFalseKeepDims::need_transformation_for_reordered_axes(std::vector<int64_t> reduce_axes,
                                                                             size_t num_dim,
                                                                             size_t num_spatial) {
    bool feature_axis_is_only_remaining = false;
    // Case to reduce batch axis and spatial axes
    if (reduce_axes.size() > 1 && count(reduce_axes.begin(), reduce_axes.end(), 0) != 0 &&
        count(reduce_axes.begin(), reduce_axes.end(), 1) == 0) {
        feature_axis_is_only_remaining = true;
        // Check if it reduces all spatial axes
        for (size_t idx_spatial = (num_dim - num_spatial); idx_spatial < num_dim; idx_spatial++) {
            if (count(reduce_axes.begin(), reduce_axes.end(), idx_spatial) == 0) {
                feature_axis_is_only_remaining = false;
                break;
            }
        }
    }

    return feature_axis_is_only_remaining;
}

}  // namespace ov::intel_gpu
