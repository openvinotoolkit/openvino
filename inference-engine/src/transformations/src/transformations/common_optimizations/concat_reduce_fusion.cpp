// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/concat_reduce_fusion.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"

#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include "itt.hpp"


NGRAPH_RTTI_DEFINITION(ngraph::pass::ConcatReduceFusionWithoutFolding, "ConcatReduceFusionWithoutFolding", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::PullSqueezeThroughEltwise, "PullSqueezeThroughEltwise", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::ConcatReduceFusion, "ConcatReduceFusion", 0);


namespace {

enum class ReduceType {NONE, MAX, MIN};

ReduceType get_reduce_type(const std::shared_ptr<ov::Node>& reduce_node) {
    if (std::dynamic_pointer_cast<ngraph::opset8::ReduceMax>(reduce_node) != nullptr) {
        return ReduceType::MAX;
    }
    if (std::dynamic_pointer_cast<ngraph::opset8::ReduceMin>(reduce_node) != nullptr) {
        return ReduceType::MIN;
    }
    return ReduceType::NONE;
}

} // namespace

ngraph::pass::PullSqueezeThroughEltwise::PullSqueezeThroughEltwise() {
    MATCHER_SCOPE(PullSqueezeThroughEltwise);
    auto eltwise_pattern = pattern::wrap_type<op::util::BinaryElementwiseArithmetic>();

    auto squeeze_axes_pattern = pattern::wrap_type<opset8::Constant>();
    auto squeeze_pattern = pattern::wrap_type<opset8::Squeeze>({eltwise_pattern, squeeze_axes_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto & pattern_map = m.get_pattern_map();
        auto eltwise = pattern_map.at(eltwise_pattern);

        if (!eltwise) {
            return false;
        }

        size_t eltwise_inputs_size = eltwise->get_input_size();
        for (size_t input_index = 0; input_index < eltwise_inputs_size; ++input_index) {
            auto input_node = eltwise->get_input_node_shared_ptr(input_index);
            if (std::dynamic_pointer_cast<opset8::Constant>(input_node) == nullptr &&
                std::dynamic_pointer_cast<opset8::Unsqueeze>(input_node) == nullptr) {
                    return false;
            }
        }

        auto squeeze = pattern_map.at(squeeze_pattern);
        if (!squeeze) {
            return false;
        }

        ngraph::OutputVector eltwise_inputs;
        for (size_t input_index = 0; input_index < eltwise_inputs_size; ++input_index) {
            auto eltwise_input = eltwise->input_value(input_index);
            std::shared_ptr<Node> new_input_node = squeeze->clone_with_new_inputs({eltwise_input, squeeze->input_value(1)});
            ngraph::copy_runtime_info(squeeze, new_input_node);
            eltwise_inputs.push_back(new_input_node);
        }

        auto new_eltwise = eltwise->copy_with_new_inputs(eltwise_inputs);
        new_eltwise->set_friendly_name(eltwise->get_friendly_name());
        ngraph::copy_runtime_info({eltwise, squeeze}, new_eltwise);
        ngraph::replace_node(squeeze, new_eltwise);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(squeeze_pattern, matcher_name);
    this->register_matcher(m, callback);
}

ngraph::pass::ConcatReduceFusionWithoutFolding::ConcatReduceFusionWithoutFolding() {
    MATCHER_SCOPE(ConcatReduceFusionWithoutFolding);

    auto concat_pattern = ngraph::pattern::wrap_type<opset8::Concat>({pattern::any_input(), pattern::any_input()});
    auto reduce_axes_pattern = ngraph::pattern::wrap_type<opset8::Constant>();
    auto reduce_pattern = ngraph::pattern::wrap_type<opset8::ReduceMin, opset8::ReduceMax>({concat_pattern, reduce_axes_pattern});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto concat = std::dynamic_pointer_cast<opset8::Concat>(pattern_map.at(concat_pattern).get_node_shared_ptr());

        ReduceType reduce_type = get_reduce_type(pattern_map.at(reduce_pattern).get_node_shared_ptr());
        if (reduce_type == ReduceType::NONE) {
            return false;
        }

        auto reduce = std::dynamic_pointer_cast<op::util::ArithmeticReductionKeepDims>(pattern_map.at(reduce_pattern).get_node_shared_ptr());

        if (!reduce || !concat) {
             return false;
        }

        const auto& reduction_axes = reduce->get_reduction_axes();
        if (reduction_axes.size() != 1 || concat->get_axis() != static_cast<int64_t>(*reduction_axes.begin())) {
            return false;
        }

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
            auto squeeze_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {}, {*reduction_axes.begin()});
            result_node = register_new_node<ngraph::opset8::Squeeze>(result_node, squeeze_axis_node);
        }

        result_node->set_friendly_name(reduce->get_friendly_name());

        replace_node(reduce, result_node);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reduce_pattern, matcher_name);
    register_matcher(m, callback);
}

bool ngraph::pass::ConcatReduceFusion::run_on_function(std::shared_ptr<ngraph::Function> f) {
    RUN_ON_FUNCTION_SCOPE(ConcatReduceFusion);
    ngraph::pass::Manager manager;
    manager.register_pass<ConcatReduceFusionWithoutFolding>();
    manager.register_pass<PullSqueezeThroughEltwise>();
    manager.register_pass<ngraph::pass::NopElimination>();
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.run_passes(f);
    return true;
}
