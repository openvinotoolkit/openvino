// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

// Convert redundant reduce to reshape : expected to be optimized out or modified
class TRANSFORMATIONS_API ConvertReduceToReshape;
class TRANSFORMATIONS_API ConvertReduceMeanToReshape;
class TRANSFORMATIONS_API ConvertReduceSumToReshape;
class TRANSFORMATIONS_API ConvertReduceProdToReshape;
class TRANSFORMATIONS_API ConvertReduceMaxToReshape;
class TRANSFORMATIONS_API ConvertReduceMinToReshape;
class TRANSFORMATIONS_API ConvertReduceLogicalAndToReshape;
class TRANSFORMATIONS_API ConvertReduceLogicalOrToReshape;

}  // namespace pass
}  // namespace ov

class CvtReduceBase : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("CvtReduceBase");

    template <class T>
    ov::matcher_pass_callback convert_reduce_to_reshape();

    bool is_redundant(ov::Shape input, ov::Shape output);
};

class ov::pass::ConvertReduceMeanToReshape : public CvtReduceBase {
public:
    OPENVINO_RTTI("ConvertReduceMeanToReshape", "0", CvtReduceBase);
    ConvertReduceMeanToReshape();
};

class ov::pass::ConvertReduceSumToReshape : public CvtReduceBase {
public:
    OPENVINO_RTTI("ConvertReduceSumToReshape", "0", CvtReduceBase);
    ConvertReduceSumToReshape();
};

class ov::pass::ConvertReduceProdToReshape : public CvtReduceBase {
public:
    OPENVINO_RTTI("ConvertReduceProdToReshape", "0", CvtReduceBase);
    ConvertReduceProdToReshape();
};

class ov::pass::ConvertReduceMaxToReshape : public CvtReduceBase {
public:
    OPENVINO_RTTI("ConvertReduceMaxToReshape", "0", CvtReduceBase);
    ConvertReduceMaxToReshape();
};

class ov::pass::ConvertReduceMinToReshape : public CvtReduceBase {
public:
    OPENVINO_RTTI("ConvertReduceMinToReshape", "0", CvtReduceBase);
    ConvertReduceMinToReshape();
};

class ov::pass::ConvertReduceLogicalAndToReshape : public CvtReduceBase {
public:
    OPENVINO_RTTI("ConvertReduceLogicalAndToReshape", "0", CvtReduceBase);
    ConvertReduceLogicalAndToReshape();
};

class ov::pass::ConvertReduceLogicalOrToReshape : public CvtReduceBase {
public:
    OPENVINO_RTTI("ConvertReduceLogicalOrToReshape", "0", CvtReduceBase);
    ConvertReduceLogicalOrToReshape();
};

class ov::pass::ConvertReduceToReshape : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ConvertReduceToReshape");
    // Handling reduce if it can be converted to reshape (check input/output tensor)
    ConvertReduceToReshape() {
        // Redundant reduce based on its mode
        add_matcher<ConvertReduceMeanToReshape>();
        add_matcher<ConvertReduceSumToReshape>();
        add_matcher<ConvertReduceProdToReshape>();
        add_matcher<ConvertReduceMaxToReshape>();
        add_matcher<ConvertReduceMinToReshape>();
        add_matcher<ConvertReduceLogicalAndToReshape>();
        add_matcher<ConvertReduceLogicalOrToReshape>();
    }
};

template <class T>
ov::matcher_pass_callback CvtReduceBase::convert_reduce_to_reshape() {
    return [&](ov::pass::pattern::Matcher& m) {
        auto reduce = ov::as_type_ptr<T>(m.get_match_root());
        if (!reduce)
            return false;

        auto input = reduce->input_value(0);
        const auto input_shape = input.get_shape();
        const auto reduce_shape = reduce->output(0).get_shape();

        // convert redundant reduce to reshape if the input shape is supported by reshape
        if (is_redundant(input_shape, reduce_shape) && input_shape.size() < 6) {
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

        return false;
    };
}
