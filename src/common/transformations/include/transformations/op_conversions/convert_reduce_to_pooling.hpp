// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertReduceToPooling;
class TRANSFORMATIONS_API ConvertReduceMeanToPooling;
class TRANSFORMATIONS_API ConvertReduceMaxToPooling;
class TRANSFORMATIONS_API ConvertReduceSumToPooling;

}  // namespace pass
}  // namespace ov

class ConvertReduceBase : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertReduceBase");

    template <class T>
    ov::matcher_pass_callback convert_reduce_to_pooling();
};

class ov::pass::ConvertReduceMeanToPooling : public ConvertReduceBase {
public:
    OPENVINO_RTTI("ConvertReduceMeanToPooling", "0", ConvertReduceBase);
    ConvertReduceMeanToPooling();
};

class ov::pass::ConvertReduceMaxToPooling : public ConvertReduceBase {
public:
    OPENVINO_RTTI("ConvertReduceMaxToPooling", "0", ConvertReduceBase);
    ConvertReduceMaxToPooling();
};

class ov::pass::ConvertReduceSumToPooling : public ConvertReduceBase {
public:
    OPENVINO_RTTI("ConvertReduceSumToPooling", "0", ConvertReduceBase);
    ConvertReduceSumToPooling();
};

class ov::pass::ConvertReduceToPooling : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ConvertReduceToPooling");
    ConvertReduceToPooling() {
        add_matcher<ConvertReduceMeanToPooling>();
        add_matcher<ConvertReduceMaxToPooling>();
        add_matcher<ConvertReduceSumToPooling>();
    }
};

template <class T>
ov::matcher_pass_callback ConvertReduceBase::convert_reduce_to_pooling() {
    return [&](ov::pass::pattern::Matcher& m) {
        auto reduce = ov::as_type_ptr<T>(m.get_match_root());

        if (!reduce || transformation_callback(reduce) || ov::shape_size(reduce->input_value(0).get_shape()) == 0) {
            return false;
        }

        auto input = reduce->input_value(0);

        auto axes_node = ov::as_type_ptr<ov::op::v0::Constant>(reduce->input_value(1).get_node_shared_ptr());
        if (!axes_node) {
            return false;
        }

        auto axes_vector = axes_node->template cast_vector<int64_t>();
        const auto input_rank = input.get_partial_shape().rank().get_length();
        // Transform negative axes into non-negative ones
        for (size_t i = 0; i < axes_vector.size(); ++i) {
            if (axes_vector[i] < 0) {
                axes_vector[i] += input_rank;
            }
        }
        std::sort(axes_vector.begin(), axes_vector.end());

        // If axes are empty we just remove Reduction operation
        if (axes_vector.empty()) {
            return replace_output_update_name(reduce->output(0), input);
        }

        auto input_shape = input.get_shape();

        // If Reduce op reduces only 1 dims we replace it with Reshape
        if (std::all_of(axes_vector.begin(), axes_vector.end(), [&input_shape](const int64_t& axis) {
                return input_shape[axis] == 1;
            })) {
            const auto reshape_shape = reduce->output(0).get_shape();
            auto reshape = std::make_shared<ov::op::v1::Reshape>(
                input,
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{reshape_shape.size()}, reshape_shape),
                true);

            reshape->set_friendly_name(reduce->get_friendly_name());
            copy_runtime_info(reduce, reshape);
            replace_node(reduce, reshape);
            return true;
        }

        // Check that axes are consecutive otherwise this transformation is not applicable
        for (size_t i = 1; i < axes_vector.size(); ++i) {
            if (axes_vector[i] - axes_vector[i - 1] != 1) {
                return false;
            }
        }

        // Check either reduction applies to spatial dimensions or not
        bool spatial_dims_reduction(true);
        size_t reduction_dims_count = 1;
        for (auto& axis : axes_vector) {
            reduction_dims_count *= input_shape[axis];
            if (axis <= 1) {
                spatial_dims_reduction = false;
            }
        }

        /*
         * Prepare default attributes for Pooling operation
         *      pads_begin/pads_end - should be zeros as we don't need any padding
         *      stride - should be filled with ones
         *      kernel - depends on Reduction operation axes
         *
         * Also here we decide should we use Reshapes before and after Pooling
         *      shape_begin - if not empty indicates that we need a Reshape before Pooling
         *      shape_end   - if not empty indicates that we need a Reshape after Pooling
         */

        ov::Strides strides;
        ov::Shape pads_begin, pads_end, kernel, shape_begin, shape_end;

        if (!spatial_dims_reduction || input_shape.size() != 4) {
            // In case if reduction applies not to spatial dimensions
            // we have to fit it into 4D Pooling
            size_t dims_prod = 1, dims_begin = 1, dims_end = 1;
            for (int64_t i = 0; static_cast<size_t>(i) < input_shape.size(); ++i) {
                if (i < *axes_vector.begin()) {
                    dims_begin *= input_shape[i];
                } else if (i >= axes_vector.front() && i <= axes_vector.back()) {
                    dims_prod *= input_shape[i];
                } else {
                    dims_end *= input_shape[i];
                }
            }
            // The batch dimenstion is repositioned in the shape
            // only in case of batch dimension reduction
            shape_begin.assign({dims_begin, 1, dims_prod, dims_end});
            shape_end = reduce->output(0).get_shape();
            strides.assign({1, 1});
            pads_begin.assign({0, 0});
            pads_end.assign({0, 0});
            kernel.assign({dims_prod, 1});
        } else {
            for (size_t i = 0; i < input_shape.size() - 2; ++i) {
                strides.push_back(1);
                pads_begin.push_back(0);
                pads_end.push_back(0);
                kernel.push_back(1);
            }
            for (auto& axis : axes_vector) {
                kernel[axis - 2] = input_shape[axis];
            }
            shape_end = reduce->output(0).get_shape();
        }

        /*
         *  ReduceMean => AvgPool
         *                AvgPool->Reshape (in case if keep_dims=False)
         *                Reshape->AvgPool->Reshape (in case if axes doesn't match spatial dims)

         *  ReduceMax  => MaxPool
         *                MaxPool->Reshape (in case if keep_dims=False)
         *                Reshape->MaxPool->Reshape (in case if axes doesn't match spatial dims)
         *
         *  ReduceSum  => AvgPool->Multiply
         *                AvgPool->Multiply->Reshape (in case if keep_dims=False)
         *                Reshape->AvgPool->Multiply->Reshape (in case if axes doesn't match spatial dims)
         *
         *  Note: some of reshape nodes can be optimized if they do nothing.
         */
        ov::NodeVector new_ops;

        if (!shape_begin.empty() && shape_begin != input.get_shape()) {
            input = std::make_shared<ov::op::v1::Reshape>(
                input,
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{shape_begin.size()}, shape_begin),
                true);
            input.get_node_shared_ptr()->set_friendly_name(reduce->get_friendly_name() + "/reshape_begin");
            new_ops.push_back(input.get_node_shared_ptr());
        }

        if (std::is_same<T, ov::op::v1::ReduceMean>()) {
            input = std::make_shared<ov::op::v1::AvgPool>(input,
                                                          strides,
                                                          pads_begin,
                                                          pads_end,
                                                          kernel,
                                                          true,
                                                          ov::op::RoundingType::FLOOR);

            input.get_node_shared_ptr()->set_friendly_name(reduce->get_friendly_name() + "/pool");
            new_ops.push_back(input.get_node_shared_ptr());
        } else if (std::is_same<T, ov::op::v1::ReduceMax>()) {
            input = std::make_shared<ov::op::v1::MaxPool>(input,
                                                          strides,
                                                          pads_begin,
                                                          pads_end,
                                                          kernel,
                                                          ov::op::RoundingType::FLOOR);

            input.get_node_shared_ptr()->set_friendly_name(reduce->get_friendly_name() + "/pool");
            new_ops.push_back(input.get_node_shared_ptr());
        } else if (std::is_same<T, ov::op::v1::ReduceSum>()) {
            // Fallback to real type because of potential data loss in case of integer AVG Pool
            bool fallback_to_real = input.get_element_type().is_integral();

            if (fallback_to_real) {
                input = std::make_shared<ov::op::v0::Convert>(input, ov::element::f32);
                new_ops.push_back(input.get_node_shared_ptr());
            }

            input = std::make_shared<ov::op::v1::AvgPool>(input,
                                                          strides,
                                                          pads_begin,
                                                          pads_end,
                                                          kernel,
                                                          true,
                                                          ov::op::RoundingType::FLOOR);

            input.get_node_shared_ptr()->set_friendly_name(reduce->get_friendly_name() + "/pool");
            new_ops.push_back(input.get_node_shared_ptr());

            input = std::make_shared<ov::op::v1::Multiply>(
                input,
                ov::op::v0::Constant::create(input.get_element_type(), ov::Shape{1}, {reduction_dims_count}));
            input.get_node_shared_ptr()->set_friendly_name(reduce->get_friendly_name() + "/mul");
            new_ops.push_back(input.get_node_shared_ptr());

            if (fallback_to_real) {
                input = std::make_shared<ov::op::v0::Convert>(input, reduce->output(0).get_element_type());
                new_ops.push_back(input.get_node_shared_ptr());
            }
        } else {
            return false;
        }

        if (shape_end != input.get_shape()) {
            input = std::make_shared<ov::op::v1::Reshape>(
                input,
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{shape_end.size()}, shape_end),
                true);
            new_ops.push_back(input.get_node_shared_ptr());
        }
        input.get_node_shared_ptr()->set_friendly_name(reduce->get_friendly_name());
        copy_runtime_info(reduce, new_ops);
        reduce->output(0).replace(input);
        return true;
    };
}
