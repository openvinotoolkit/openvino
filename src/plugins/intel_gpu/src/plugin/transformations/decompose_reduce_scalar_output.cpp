// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "decompose_reduce_scalar_output.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

#define CREATE_REDUCE(input, reduce_const, keep_dims)                                          \
    if (ov::is_type<ov::op::v1::ReduceSum>(reduce_orig))                                       \
        reduce_new = std::make_shared<ov::op::v1::ReduceSum>(input, reduce_const, keep_dims);  \
    else if (ov::is_type<ov::op::v1::ReduceMin>(reduce_orig))                                  \
        reduce_new = std::make_shared<ov::op::v1::ReduceMin>(input, reduce_const, keep_dims);  \
    else if (ov::is_type<ov::op::v1::ReduceMax>(reduce_orig))                                  \
        reduce_new = std::make_shared<ov::op::v1::ReduceMax>(input, reduce_const, keep_dims);  \
    else if (ov::is_type<ov::op::v1::ReduceProd>(reduce_orig))                                 \
        reduce_new = std::make_shared<ov::op::v1::ReduceProd>(input, reduce_const, keep_dims); \
    else                                                                                       \
        return false;

ov::intel_gpu::DecomposeReduceForScalarOutput::DecomposeReduceForScalarOutput() {
    auto check_reduce_shape = [=](Output<Node> output) -> bool {
        const auto reduce = ov::as_type_ptr<op::util::ArithmeticReductionKeepDims>(output.get_node_shared_ptr());
        auto& input_shape = reduce->input_value(0).get_partial_shape();
        auto& reduce_shape = reduce->input_value(1).get_partial_shape();
        if (reduce_shape.is_dynamic() || reduce_shape.size() != 1) {
            return false;
        } else if (reduce_shape.to_shape()[0] <= 1 || reduce_shape.to_shape()[0] != input_shape.size()) {
            return false;
        }
        auto& output_shape = reduce->get_output_partial_shape(0);
        if (output_shape.is_static() && input_shape.is_static()) {
            // Output size decides at most how many EU threads can be used for this node execution,
            // less than 4 EU threads to execute a primitive will lead to poor performance.
            if (ov::shape_size(output_shape.to_shape()) > 4) {
                return false;
            }
            // Input shape is too small, 1 EU thread should be enough.
            const auto input_static_shape = input_shape.to_shape();
            if (ov::shape_size(input_static_shape) < 64) {
                return false;
            }
        }
        return true;
    };

    auto reduce_pattern = ov::pass::pattern::
        wrap_type<ov::op::v1::ReduceSum, ov::op::v1::ReduceProd, ov::op::v1::ReduceMin, ov::op::v1::ReduceMax>(
            {ov::pass::pattern::any_input(), ov::pass::pattern::wrap_type<ov::op::v0::Constant>()},
            check_reduce_shape);

    // register callback
    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto reduce_orig =
            as_type_ptr<op::util::ArithmeticReductionKeepDims>(pattern_map.at(reduce_pattern).get_node_shared_ptr());
        if (!reduce_orig || transformation_callback(reduce_orig))
            return false;

        auto& input_shape = reduce_orig->input_value(0).get_partial_shape();
        auto& output_shape = reduce_orig->get_output_partial_shape(0);
        bool dynamic_shape = input_shape.is_dynamic() || output_shape.is_dynamic();
        std::shared_ptr<ov::op::util::ArithmeticReductionKeepDims> reduce_new = nullptr;
        if (!dynamic_shape) {
            // Find out the the most length dimension
            const auto input_static_shape = input_shape.to_shape();
            size_t max_dim = std::distance(input_static_shape.begin(),
                                           std::max_element(input_static_shape.begin(), input_static_shape.end()));
            if (input_static_shape[max_dim] == ov::shape_size(input_static_shape)) {
                return false;
            }
            CREATE_REDUCE(reduce_orig->input_value(0),
                          ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {max_dim}),
                          true);

        } else if (input_shape.rank().is_static()) {
            // Dynamic shape and output shape is [0], which will lead to 1 EU thread to do all work.
            auto input = reduce_orig->input_value(0);
            for (size_t i = input_shape.size() - 1; i > 0; i--) {
                // Reduce one dimension by one dimension to avoid 1 EU thread do all work.
                if (input_shape[i].is_dynamic() || (input_shape[i].is_static() && input_shape[i].get_length() >= 4)) {
                    CREATE_REDUCE(input, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {i}), true);
                    input = reduce_new->get_default_output();
                }
            }
        }
        if (!reduce_new)
            return false;

        CREATE_REDUCE(reduce_new->get_default_output(), reduce_orig->input_value(1), reduce_orig->get_keep_dims());
        reduce_new->set_friendly_name(reduce_orig->get_friendly_name());
        copy_runtime_info(reduce_orig, reduce_new);
        replace_node(reduce_orig, reduce_new);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(reduce_pattern, "DecomposeReduceForScalarOutput");
    register_matcher(m, callback);
}
