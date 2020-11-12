// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "legacy/transformations/convert_opset1_to_legacy/ssd_pattern_before_topk_fusion.hpp"

#include <memory>
#include <vector>

#include <ngraph/graph_util.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <legacy/ngraph_ops/gather_ie.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
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
        auto data = label_to_output[m_data].get_node_shared_ptr();
        auto data_shape = shape_of->get_input_partial_shape(0);

        if (data_shape.rank().is_dynamic() || data_shape.is_dynamic()) {
            return false;
        }

        auto indices = label_to_output[m_indices].get_node_shared_ptr();
        auto indices_vector = indices->cast_vector<int64_t>();
        if (indices_vector.size() != 1 || indices_vector[0] != 0) {
            return false;
        }

        auto gather_axis = label_to_output[m_gather_axis].get_node_shared_ptr();
        auto gather_axis_vector = gather_axis->cast_vector<int64_t>();
        if (gather_axis_vector.size() != 1 || gather_axis_vector[0] != 0) {
            return false;
        }

        auto fst_unsqueeze_axis = label_to_output[m_fst_unsqueeze_axis].get_node_shared_ptr();
        auto fst_unsqueeze_axis_vector = fst_unsqueeze_axis->cast_vector<int64_t>();
        if (fst_unsqueeze_axis_vector.size() != 1 || fst_unsqueeze_axis_vector[0] != 0) {
            return false;
        }

        auto snd_unsqueeze_axis = label_to_output[m_snd_unsqueeze_axis].get_node_shared_ptr();
        auto snd_unsqueeze_axis_vector = snd_unsqueeze_axis->cast_vector<int64_t>();
        if (snd_unsqueeze_axis_vector.size() != 1 || snd_unsqueeze_axis_vector[0] != 0) {
            return false;
        }

        auto concat_const = label_to_output[m_concat_const].get_node_shared_ptr();
        auto concat_const_vector = concat_const->cast_vector<int64_t>();
        if (concat_const_vector.size() != 1) {
            return false;
        }

        return true;
    };

    auto folding = [=](ngraph::pattern::Matcher & m) -> Output<Node> {
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

    auto m = std::make_shared<ngraph::pattern::Matcher>(m_mul, "PatternBeforeTopKFusion");
    this->register_matcher(m, callback);
}
