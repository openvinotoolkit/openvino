// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <openvino/core/rt_info.hpp>
#include <openvino/opsets/opset10.hpp>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ov {
namespace pass {

// Modify reduce if keep_dims is false and it reduces batch and spatial axes
class TRANSFORMATIONS_API ModifyReduceForFalseKeepDims;
class TRANSFORMATIONS_API ModifyReduceMeanForFalseKeepDims;
class TRANSFORMATIONS_API ModifyReduceSumForFalseKeepDims;
class TRANSFORMATIONS_API ModifyReduceMaxForFalseKeepDims;
class TRANSFORMATIONS_API ModifyReduceMinForFalseKeepDims;
class TRANSFORMATIONS_API ModifyReduceLogicalAndForFalseKeepDims;
class TRANSFORMATIONS_API ModifyReduceLogicalOrForFalseKeepDims;

}  // namespace pass
}  // namespace ov

// Add Reshape to modify output of Reduce and update keep_dims to true : reduce-reshape
// A clDNN Reduce reorders un-reduced axes of its output tensor to b-f and spatial order when keep_dims is false.
// oneDNN reduction does not allow this. And clDNN execution shows a hug perf drop for blocked formats.
class ModifyReduceBase : public ov::pass::MatcherPass {
public:
    template <class T>
    ov::matcher_pass_callback modify_reduce_for_false_keepdims();

    bool is_unsupported_reordered_axes(std::vector<int64_t> reduce_axes, size_t num_dim, size_t num_spatial);
};

class ov::pass::ModifyReduceMeanForFalseKeepDims : public ModifyReduceBase {
public:
    OPENVINO_RTTI("ModifyReduceMeanForFalseKeepDims", "0");
    ModifyReduceMeanForFalseKeepDims();
};

class ov::pass::ModifyReduceSumForFalseKeepDims : public ModifyReduceBase {
public:
    OPENVINO_RTTI("ModifyReduceSumForFalseKeepDims", "0");
    ModifyReduceSumForFalseKeepDims();
};

class ov::pass::ModifyReduceMaxForFalseKeepDims : public ModifyReduceBase {
public:
    OPENVINO_RTTI("ModifyReduceMaxForFalseKeepDims", "0");
    ModifyReduceMaxForFalseKeepDims();
};

class ov::pass::ModifyReduceMinForFalseKeepDims : public ModifyReduceBase {
public:
    OPENVINO_RTTI("ModifyReduceMinForFalseKeepDims", "0");
    ModifyReduceMinForFalseKeepDims();
};

class ov::pass::ModifyReduceLogicalAndForFalseKeepDims : public ModifyReduceBase {
public:
    OPENVINO_RTTI("ModifyReduceLogicalAndForFalseKeepDims", "0");
    ModifyReduceLogicalAndForFalseKeepDims();
};

class ov::pass::ModifyReduceLogicalOrForFalseKeepDims : public ModifyReduceBase {
public:
    OPENVINO_RTTI("ModifyReduceLogicalOrForFalseKeepDims", "0");
    ModifyReduceLogicalOrForFalseKeepDims();
};

class ov::pass::ModifyReduceForFalseKeepDims : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("ModifyReduceForFalseKeepDims", "0");
    ModifyReduceForFalseKeepDims() {
        add_matcher<ModifyReduceMeanForFalseKeepDims>();
        add_matcher<ModifyReduceSumForFalseKeepDims>();
        add_matcher<ModifyReduceMaxForFalseKeepDims>();
        add_matcher<ModifyReduceMinForFalseKeepDims>();
        add_matcher<ModifyReduceLogicalAndForFalseKeepDims>();
        add_matcher<ModifyReduceLogicalOrForFalseKeepDims>();
    }
};

template <class T>
ov::matcher_pass_callback ModifyReduceBase::modify_reduce_for_false_keepdims() {
    return [&](ov::pass::pattern::Matcher& m) {
        auto reduce = std::dynamic_pointer_cast<T>(m.get_match_root());
        if (!reduce)
            return false;

        auto input = reduce->input_value(0);
        const auto input_shape = input.get_shape();
        const auto reduce_shape = reduce->output(0).get_shape();
        const auto input_rank = input.get_partial_shape().rank().get_length();

        auto axes_vector = reduce->get_reduction_axes().to_vector();
        std::sort(axes_vector.begin(), axes_vector.end());

        if (is_unsupported_reordered_axes(axes_vector, input_rank, (input_rank - 2)) && input_shape.size() < 6) {
            ngraph::NodeVector new_ops;

            // Reduce
            auto reduce_const =
                ov::opset10::Constant::create(ov::element::i64, ov::Shape{axes_vector.size()}, axes_vector);
            input = std::make_shared<T>(input, reduce_const, true);
            input.get_node_shared_ptr()->set_friendly_name(reduce->get_friendly_name());
            new_ops.push_back(input.get_node_shared_ptr());

            // Reshape
            auto reshape_shape = ov::Shape(input_shape.size(), 1);
            // Expected that a feature axis is only un-reduced.
            reshape_shape[0] = reduce_shape[0];
            input = std::make_shared<ov::opset10::Reshape>(
                input,
                ov::opset10::Constant::create(ov::element::i64, ov::Shape{reshape_shape.size()}, reshape_shape),
                false);

            input.get_node_shared_ptr()->set_friendly_name(reduce->get_friendly_name() + "_reshape_false_keepdims");
            new_ops.push_back(input.get_node_shared_ptr());

            copy_runtime_info(reduce, new_ops);
            reduce->output(0).replace(input);
            return true;
        }

        return false;
    };
}

namespace pass {
using ov::pass::ModifyReduceForFalseKeepDims;
using ov::pass::ModifyReduceLogicalAndForFalseKeepDims;
using ov::pass::ModifyReduceLogicalOrForFalseKeepDims;
using ov::pass::ModifyReduceMaxForFalseKeepDims;
using ov::pass::ModifyReduceMeanForFalseKeepDims;
using ov::pass::ModifyReduceMinForFalseKeepDims;
using ov::pass::ModifyReduceSumForFalseKeepDims;
}  // namespace pass
