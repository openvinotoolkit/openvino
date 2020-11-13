// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "legacy/transformations/convert_opset1_to_legacy/ssd_pattern_before_topk_fusion.hpp"

#include <memory>
#include <vector>

#include <ngraph/graph_util.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <legacy/ngraph_ops/gather_ie.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/runtime/host_tensor.hpp>
#include <transformations/utils/utils.hpp>

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ngraph::pass::PatternBeforeTopKFusion, "PatternBeforeTopKFusion", 0);

ngraph::pass::PatternBeforeTopKFusion::PatternBeforeTopKFusion() {
    auto m_data = ngraph::pattern::any_input();
    auto m_shape_of = ngraph::pattern::wrap_type<opset5::ShapeOf>({m_data}, pattern::consumers_count(1));
    auto m_indices = ngraph::pattern::wrap_type<opset5::Constant>();
    auto m_gather_axis = ngraph::pattern::wrap_type<opset5::Constant>();
    auto m_gather_ie = ngraph::pattern::wrap_type<op::GatherIE>({m_shape_of, m_indices, m_gather_axis}, pattern::consumers_count(1));
    auto m_fst_unsqueeze_axis = ngraph::pattern::wrap_type<opset5::Constant>();
    auto m_fst_unsqueeze = ngraph::pattern::wrap_type<opset5::Unsqueeze>({m_gather_ie, m_fst_unsqueeze_axis}, pattern::consumers_count(1));
    auto m_concat_const = ngraph::pattern::wrap_type<opset5::Constant>();
    auto m_concat = ngraph::pattern::wrap_type<opset5::Concat>({m_concat_const, m_fst_unsqueeze}, pattern::consumers_count(1));
    auto m_fst_convert = ngraph::pattern::wrap_type<opset5::Convert>({m_concat}, pattern::consumers_count(1));
    auto m_reduce_axes = ngraph::pattern::wrap_type<opset5::Constant>();
    auto m_reduce =  ngraph::pattern::wrap_type<opset5::ReduceMin>({m_fst_convert, m_reduce_axes}, pattern::consumers_count(1));
    auto m_snd_convert = ngraph::pattern::wrap_type<opset5::Convert>({m_reduce}, pattern::consumers_count(1));
    auto m_snd_unsqueeze_axis = ngraph::pattern::wrap_type<opset5::Constant>();
    auto m_snd_unsqueeze = ngraph::pattern::wrap_type<opset5::Unsqueeze>({m_snd_convert, m_snd_unsqueeze_axis});

    auto is_applicable = [=](ngraph::pattern::Matcher & m) -> bool {
        auto & label_to_output = m.get_pattern_value_map();

        auto shape_of = label_to_output[m_shape_of].get_node_shared_ptr();
        auto data_shape = shape_of->get_input_partial_shape(0);

        if (data_shape.rank().is_dynamic() || data_shape.is_dynamic()) {
            return false;
        }

        auto indices_node = label_to_output[m_indices];
        auto indices_const = std::dynamic_pointer_cast<ngraph::opset5::Constant>(indices_node.get_node_shared_ptr());
        if (!indices_const) {
            return false;
        }
        auto indices_vector = indices_const->cast_vector<int64_t>();
        if (indices_vector.size() != 1 || indices_vector[0] != 0) {
            return false;
        }

        auto gather_axis_node = label_to_output[m_gather_axis];
        auto gather_axis_const = std::dynamic_pointer_cast<ngraph::opset5::Constant>(gather_axis_node.get_node_shared_ptr());
        if (!gather_axis_const) {
            return false;
        }
        auto gather_axis_vector = gather_axis_const->cast_vector<int64_t>();
        if (gather_axis_vector.size() != 1 || gather_axis_vector[0] != 0) {
            return false;
        }

        auto fst_unsqueeze_axis_node = label_to_output[m_fst_unsqueeze_axis];
        auto fst_unsqueeze_axis_const = std::dynamic_pointer_cast<ngraph::opset5::Constant>(fst_unsqueeze_axis_node.get_node_shared_ptr());
        if (!fst_unsqueeze_axis_const) {
            return false;
        }
        auto fst_unsqueeze_axis_vector = fst_unsqueeze_axis_const->cast_vector<int64_t>();
        if (fst_unsqueeze_axis_vector.size() != 1 || fst_unsqueeze_axis_vector[0] != 0) {
            return false;
        }

        auto snd_unsqueeze_axis_node = label_to_output[m_snd_unsqueeze_axis];
        auto snd_unsqueeze_axis_const = std::dynamic_pointer_cast<ngraph::opset5::Constant>(snd_unsqueeze_axis_node.get_node_shared_ptr());
        if (!snd_unsqueeze_axis_const) {
            return false;
        }
        auto snd_unsqueeze_axis_vector = snd_unsqueeze_axis_const->cast_vector<int64_t>();
        if (snd_unsqueeze_axis_vector.size() != 1 || snd_unsqueeze_axis_vector[0] != 0) {
            return false;
        }

        auto concat_const_node = label_to_output[m_concat_const];
        auto concat_const = std::dynamic_pointer_cast<ngraph::opset5::Constant>(concat_const_node.get_node_shared_ptr());
        if (!concat_const) {
            return false;
        }
        auto concat_const_vector = concat_const->cast_vector<int64_t>();
        if (concat_const_vector.size() != 1) {
            return false;
        }

        auto concat_node = label_to_output[m_concat];
        auto concat = std::dynamic_pointer_cast<ngraph::opset5::Concat>(concat_node.get_node_shared_ptr());
        if (!concat) {
            return false;
        }
        int64_t concat_axis = concat->get_axis();
        if (concat_axis != 0) {
            return false;
        }

        return true;
    };

    auto folding = [=](ngraph::pattern::Matcher & m) -> Output<Node> {
        auto & label_to_output = m.get_pattern_value_map();

        auto shape_of = label_to_output[m_shape_of].get_node_shared_ptr();
        Shape data_shape = shape_of->get_input_shape(0);
        Output<Node> gather_input = opset5::Constant::create(element::i64, shape_of->get_output_shape(0), data_shape);

        auto indices = label_to_output[m_indices];
        auto gather_axis = label_to_output[m_gather_axis];
        auto gather = std::make_shared<opset5::Gather>(gather_input, indices, gather_axis);
        OutputVector gather_output(gather->get_output_size());
        if (!gather->constant_fold(gather_output, {gather_input, indices, gather_axis})) {
            throw ngraph_error("Can not constant fold Gather node");
        }

        auto fst_unsqueeze_axis_node = label_to_output[m_fst_unsqueeze_axis];
        auto fst_unsqueeze = std::make_shared<opset5::Unsqueeze>(gather_output[0], fst_unsqueeze_axis_node);
        OutputVector fst_unsqueeze_output(fst_unsqueeze->get_output_size());
        if (!fst_unsqueeze->constant_fold(fst_unsqueeze_output, {gather_output[0], fst_unsqueeze_axis_node})) {
            throw ngraph_error("Can not constant the first Unsqueeze node");
        }

        auto concat_const_node = label_to_output[m_concat_const];
        auto concat_node = label_to_output[m_concat];
        auto concat = std::dynamic_pointer_cast<ngraph::opset5::Concat>(concat_node.get_node_shared_ptr());
        if (!concat) {
            throw ngraph_error("Expected Concat node");
        }
        int64_t concat_axis = concat->get_axis();
        OutputVector new_concat_args = {concat_const_node, fst_unsqueeze_output[0]};
        auto new_concat = std::make_shared<opset5::Concat>(new_concat_args, concat_axis);
        auto concat_result = std::make_shared<HostTensor>();
        HostTensorVector concat_eval_args = {std::make_shared<HostTensor>(concat_const_node), std::make_shared<HostTensor>(fst_unsqueeze_output[0])};
        bool new_concat_eval_status = new_concat->evaluate({concat_result}, concat_eval_args);
        Output<Node> result;
        return result;
    };

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher & m) -> bool {
        if (!is_applicable(m)) {
            return false;
        }

        auto folded_constant = folding(m);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(m_snd_unsqueeze, "PatternBeforeTopKFusion");
    this->register_matcher(m, callback);
}
