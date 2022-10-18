// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/concat_reduce_fusion.hpp"

#include <memory>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <vector>

#include "itt.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/utils/utils.hpp"

namespace {
enum class ReduceType { NONE, MAX, MIN };

ReduceType get_reduce_type(const std::shared_ptr<ov::Node>& reduce_node) {
    if (ov::is_type<ngraph::opset8::ReduceMax>(reduce_node)) {
        return ReduceType::MAX;
    } else if (ov::is_type<ngraph::opset8::ReduceMin>(reduce_node)) {
        return ReduceType::MIN;
    } else {
        return ReduceType::NONE;
    }
}
}  // namespace

ngraph::pass::PullSqueezeThroughEltwise::PullSqueezeThroughEltwise() {
    MATCHER_SCOPE(PullSqueezeThroughEltwise);
    auto eltwise_pattern = pattern::wrap_type<op::util::BinaryElementwiseArithmetic>();

    auto squeeze_axes_pattern = pattern::wrap_type<opset8::Constant>();
    auto squeeze_pattern = pattern::wrap_type<opset8::Squeeze>({eltwise_pattern, squeeze_axes_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_map();
        const auto& eltwise = pattern_map.at(eltwise_pattern);
        const auto& squeeze = pattern_map.at(squeeze_pattern);

        size_t eltwise_inputs_size = eltwise->get_input_size();
        for (size_t input_index = 0; input_index < eltwise_inputs_size; ++input_index) {
            const auto input_node = eltwise->get_input_node_shared_ptr(input_index);
            // check that we will able to fuse propagated squeeze in NopElimination pass
            if (!is_type<opset8::Constant>(input_node) && !is_type<opset8::Unsqueeze>(input_node) &&
                !is_type<opset8::Reshape>(input_node)) {
                return false;
            }
        }

        ngraph::OutputVector eltwise_inputs;
        for (size_t input_index = 0; input_index < eltwise_inputs_size; ++input_index) {
            const auto eltwise_input = eltwise->input_value(input_index);
            const auto new_input_node =
                ngraph::op::util::clone_try_fold(squeeze, {eltwise_input, squeeze->input_value(1)});

            if (!is_type<opset8::Constant>(new_input_node))
                register_new_node(as_type_ptr<opset8::Squeeze>(new_input_node));

            ngraph::copy_runtime_info(squeeze, new_input_node);
            eltwise_inputs.push_back(new_input_node);
        }

        const auto new_eltwise = ngraph::op::util::clone_try_fold(eltwise, eltwise_inputs);
        new_eltwise->set_friendly_name(squeeze->get_friendly_name());
        ngraph::copy_runtime_info({eltwise, squeeze}, new_eltwise);
        ngraph::replace_node(squeeze, new_eltwise);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(squeeze_pattern, matcher_name);
    this->register_matcher(m, callback);
}

ngraph::pass::ReplaceConcatReduceByMinOrMax::ReplaceConcatReduceByMinOrMax() {
    MATCHER_SCOPE(ReplaceConcatReduceByMinOrMax);

    auto concat_pattern = ngraph::pattern::wrap_type<opset8::Concat>({pattern::any_input(), pattern::any_input()});
    auto reduce_axes_pattern = ngraph::pattern::wrap_type<opset8::Constant>();
    auto reduce_pattern =
        ngraph::pattern::wrap_type<opset8::ReduceMin, opset8::ReduceMax>({concat_pattern, reduce_axes_pattern});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto concat = as_type_ptr<opset8::Concat>(pattern_map.at(concat_pattern).get_node_shared_ptr());
        auto reduce =
            as_type_ptr<op::util::ArithmeticReductionKeepDims>(pattern_map.at(reduce_pattern).get_node_shared_ptr());
        if (!reduce || !concat)
            return false;

        const auto& reduction_axes = reduce->get_reduction_axes();
        if (reduction_axes.size() != 1 || concat->get_axis() != static_cast<int64_t>(*reduction_axes.begin())) {
            return false;
        }

        ReduceType reduce_type = get_reduce_type(reduce);
        std::shared_ptr<ngraph::Node> result_node;
        switch (reduce_type) {
        case ReduceType::MAX:
            result_node = register_new_node<opset8::Maximum>(concat->input_value(0), concat->input_value(1));
            break;
        case ReduceType::MIN:
            result_node = register_new_node<opset8::Minimum>(concat->input_value(0), concat->input_value(1));
            break;
        default:
            return false;
        }

        copy_runtime_info({concat, reduce}, result_node);

        if (!reduce->get_keep_dims()) {
            const auto squeeze_axis_node =
                ngraph::opset8::Constant::create(ngraph::element::i64, {}, {*reduction_axes.begin()});
            result_node = register_new_node<ngraph::opset8::Squeeze>(result_node, squeeze_axis_node);
            copy_runtime_info({concat, reduce}, result_node);
        }

        result_node->set_friendly_name(reduce->get_friendly_name());
        replace_node(reduce, result_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reduce_pattern, matcher_name);
    register_matcher(m, callback);
}

ngraph::pass::ConcatReduceFusion::ConcatReduceFusion() {
    add_matcher<ReplaceConcatReduceByMinOrMax>();
    add_matcher<PullSqueezeThroughEltwise>();
    add_matcher<EliminateSqueeze>();
}
