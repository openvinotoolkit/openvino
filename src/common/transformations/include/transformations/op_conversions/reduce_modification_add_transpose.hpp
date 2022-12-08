// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <openvino/core/rt_info.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ov {
namespace pass {

// Modify reduce if keep_dims is false and it reduces batch and spatial axes
class TRANSFORMATIONS_API ModifyReduceToAddTranspose;
class TRANSFORMATIONS_API ModifyReduceMeanToAddTranspose;
class TRANSFORMATIONS_API ModifyReduceSumToAddTranspose;
class TRANSFORMATIONS_API ModifyReduceMaxToAddTranspose;
class TRANSFORMATIONS_API ModifyReduceMinToAddTranspose;
class TRANSFORMATIONS_API ModifyReduceLogicalAndToAddTranspose;
class TRANSFORMATIONS_API ModifyReduceLogicalOrToAddTranspose;

}  // namespace pass
}  // namespace ov

// Add transposes before and after Reduce if it has reduction of feature and spatial axes and remains 1 spatial axis
// Reduction which remains single spatial axis generally shows a huge pef drop on oneDNN execution.
// So a transpose changes an input tensor to reduce all spatial axes by moving un-reduced axis to feature axis.
class ModifyReduceBaseTranspose : public ov::pass::MatcherPass {
public:
    template <class T>
    ov::matcher_pass_callback add_transpose_to_reduce();

    // Get original order of transposed tensor
    std::vector<uint16_t> get_reverse_order(std::vector<uint16_t>& transposed_order);

    // Return true if it reduces spatial and feature axes and remains one spatial axis
    // An output 'order' is valid if return is true. It refers to an order of transposed input tensor for Reduce
    bool is_reduce_single_spatial_axis(const std::vector<int64_t>& reduce_axes,
                                       size_t num_dim,
                                       size_t num_spatial,
                                       std::vector<uint16_t>& order);
};

class ov::pass::ModifyReduceMeanToAddTranspose : public ModifyReduceBaseTranspose {
public:
    OPENVINO_RTTI("ModifyReduceMeanToAddTranspose", "0");
    ModifyReduceMeanToAddTranspose();
};

class ov::pass::ModifyReduceSumToAddTranspose : public ModifyReduceBaseTranspose {
public:
    OPENVINO_RTTI("ModifyReduceSumToAddTranspose", "0");
    ModifyReduceSumToAddTranspose();
};

class ov::pass::ModifyReduceMaxToAddTranspose : public ModifyReduceBaseTranspose {
public:
    OPENVINO_RTTI("ModifyReduceMaxToAddTranspose", "0");
    ModifyReduceMaxToAddTranspose();
};

class ov::pass::ModifyReduceMinToAddTranspose : public ModifyReduceBaseTranspose {
public:
    OPENVINO_RTTI("ModifyReduceMinToAddTranspose", "0");
    ModifyReduceMinToAddTranspose();
};

class ov::pass::ModifyReduceLogicalAndToAddTranspose : public ModifyReduceBaseTranspose {
public:
    OPENVINO_RTTI("ModifyReduceLogicalAndToAddTranspose", "0");
    ModifyReduceLogicalAndToAddTranspose();
};

class ov::pass::ModifyReduceLogicalOrToAddTranspose : public ModifyReduceBaseTranspose {
public:
    OPENVINO_RTTI("ModifyReduceLogicalOrToAddTranspose", "0");
    ModifyReduceLogicalOrToAddTranspose();
};

class ov::pass::ModifyReduceToAddTranspose : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("ModifyReduceToAddTranspose", "0");
    ModifyReduceToAddTranspose() {
        add_matcher<ModifyReduceMeanToAddTranspose>();
        add_matcher<ModifyReduceSumToAddTranspose>();
        add_matcher<ModifyReduceMaxToAddTranspose>();
        add_matcher<ModifyReduceMinToAddTranspose>();
        add_matcher<ModifyReduceLogicalAndToAddTranspose>();
        add_matcher<ModifyReduceLogicalOrToAddTranspose>();
    }
};

template <class T>
ov::matcher_pass_callback ModifyReduceBaseTranspose::add_transpose_to_reduce() {
    return [&](ov::pass::pattern::Matcher& m) {
        auto reduce = std::dynamic_pointer_cast<T>(m.get_match_root());
        if (!reduce)
            return false;

        auto input = reduce->input_value(0);
        const auto input_shape = input.get_shape();
        const auto reduce_shape = reduce->output(0).get_shape();
        auto axes_node = std::dynamic_pointer_cast<ov::opset1::Constant>(reduce->input_value(1).get_node_shared_ptr());
        if (!axes_node)
            return false;

        auto axes_vector = axes_node->template cast_vector<int64_t>();
        const auto input_rank = input.get_partial_shape().rank().get_length();
        // Transform negative axes into non-negative ones
        for (size_t i = 0; i < axes_vector.size(); ++i) {
            if (axes_vector[i] < 0) {
                axes_vector[i] += input_rank;
            }
        }
        std::sort(axes_vector.begin(), axes_vector.end());

        std::vector<uint16_t> order;
        if (is_reduce_single_spatial_axis(axes_vector, input_rank, (input_rank - 2), order)) {
            ngraph::NodeVector new_ops;

            auto create_constant = [](std::vector<uint16_t>& vec) -> std::shared_ptr<ov::opset1::Constant> {
                return ov::opset1::Constant::create(ov::element::i64, ov::Shape{vec.size()}, vec);
            };

            // Transpose
            input = std::make_shared<ov::opset1::Transpose>(input, create_constant(order));
            input.get_node_shared_ptr()->set_friendly_name(reduce->get_friendly_name() + "_input_transposed");
            new_ops.push_back(input.get_node_shared_ptr());

            // Get axes for reducing on transposed tensor from the first transpose
            std::vector<int64_t> transposed_axes;
            for (auto axis : axes_vector) {
                for (size_t idx = 0; idx < order.size(); idx++) {
                    if (axis == order[idx])
                        transposed_axes.push_back(idx);
                }
            }

            // Reduce
            auto reduce_const =
                ov::opset1::Constant::create(ov::element::i64, ov::Shape{transposed_axes.size()}, transposed_axes);
            input = std::make_shared<T>(input, reduce_const, true);
            input.get_node_shared_ptr()->set_friendly_name(reduce->get_friendly_name());
            new_ops.push_back(input.get_node_shared_ptr());

            // Transpose
            auto order_for_revert = get_reverse_order(order);
            input = std::make_shared<ov::opset1::Transpose>(input, create_constant(order_for_revert));
            input.get_node_shared_ptr()->set_friendly_name(reduce->get_friendly_name() + "_output_transposed");
            new_ops.push_back(input.get_node_shared_ptr());

            copy_runtime_info(reduce, new_ops);
            reduce->output(0).replace(input);
            return true;
        }

        return false;
    };
}

namespace pass {
using ov::pass::ModifyReduceLogicalAndToAddTranspose;
using ov::pass::ModifyReduceLogicalOrToAddTranspose;
using ov::pass::ModifyReduceMaxToAddTranspose;
using ov::pass::ModifyReduceMeanToAddTranspose;
using ov::pass::ModifyReduceMinToAddTranspose;
using ov::pass::ModifyReduceSumToAddTranspose;
using ov::pass::ModifyReduceToAddTranspose;
}  // namespace pass
