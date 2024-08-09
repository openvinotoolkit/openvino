// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_reducemax_scalar_output.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include "openvino/op/reduce_max.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/core/rt_info.hpp"

ov::intel_gpu::ConvertReduceMaxScalarOutput::ConvertReduceMaxScalarOutput() {
    // Check all Reduce nodes
    auto m = std::make_shared<ov::pass::pattern::Matcher>(ov::pass::pattern::wrap_type<ov::op::v1::ReduceMax>(),
                                                          "ConvertReduceMaxScalarOutput");
    register_matcher(m, [&](ov::pass::pattern::Matcher& m) {
        auto reduce_max = std::dynamic_pointer_cast<ov::op::v1::ReduceMax>(m.get_match_root());
        if (!reduce_max || transformation_callback(reduce_max)) {
            return false;
        }

        const auto input_shape = reduce_max->input_value(0).get_partial_shape();
        auto reduce_shape = reduce_max->input_value(1).get_partial_shape();
        if (reduce_shape.is_dynamic() || reduce_shape.size() != 1 || reduce_shape.to_shape()[0] != input_shape.size() ||
            reduce_shape.to_shape()[0] <= 1) {
            return false;
        }

        auto dynamic_shape = false;
        const auto output_shape = reduce_max->get_output_partial_shape(0);
        if (input_shape.is_dynamic() || output_shape.is_dynamic()) {
            dynamic_shape = true;
        }

        std::shared_ptr<ov::op::v1::ReduceMax> reduce_ = nullptr, reduce = nullptr;
        if (dynamic_shape == false) {
            // Output size decides at most how many EUs can be used for this node execution,
            // less than 4 EUs to execute a primitive will lead to poor performance.
            if (ov::shape_size(output_shape.to_shape()) > 4) {
                return false;
            }
            // Input shape is too small, 1 EU should be enough.
            const auto input_static_shape = input_shape.to_shape();
            if (ov::shape_size(input_static_shape) < 64) {
                return false;
            }

            // Find out the the most length dimension
            size_t max_dim = std::distance(input_static_shape.begin(),
                                           std::max_element(input_static_shape.begin(), input_static_shape.end()));
            if (input_static_shape[max_dim] == ov::shape_size(input_static_shape)) {
                return false;
            }

            reduce_ = std::make_shared<ov::op::v1::ReduceMax>(
                reduce_max->input_value(0),
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {max_dim}),
                true);
        } else if (input_shape.rank().is_static()) {
            // Dynamic shape and output shape is [0], which will lead to 1 EU to do all work
            for (size_t i = 0; i < input_shape.size() - 1; i++) {
                // Reduce one dimension by one dimension to avoid 1 EU do all work.
                if (input_shape[i].is_dynamic() || (input_shape[i].is_static() && input_shape[i].get_length() >= 4)) {
                    if (!reduce_)
                        reduce_ = std::make_shared<ov::op::v1::ReduceMax>(
                            reduce_max->input_value(0),
                            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {i}),
                            true);
                    else
                        reduce_ = std::make_shared<ov::op::v1::ReduceMax>(
                            reduce_->get_default_output(),
                            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {i}),
                            true);
                }
            }
        }

        if (!reduce_)
            return false;
        reduce = std::make_shared<ov::op::v1::ReduceMax>(reduce_->get_default_output(),
                                                         reduce_max->input_value(1),
                                                         reduce_max->get_keep_dims());
        reduce->set_friendly_name(reduce_max->get_friendly_name());
        copy_runtime_info(reduce_max, reduce);
        replace_node(reduce_max, reduce);
        return true;
    });
}
