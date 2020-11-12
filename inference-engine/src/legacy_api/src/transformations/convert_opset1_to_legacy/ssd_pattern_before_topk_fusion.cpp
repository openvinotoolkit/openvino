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
    auto m_shape_of = ngraph::pattern::wrap_type<opset5::ShapeOf>({data}, pattern::consumers_count(1));
    auto m_indices = ngraph::pattern::wrap_type<opset5::Constant>();
    auto m_gather_axis = ngraph::pattern::wrap_type<opset5::Constant>();
    auto m_gather_ie = ngraph::pattern::wrap_type<op::GatherIE>({shape_of, indices, gather_axis}, pattern::consumers_count(1));
    auto m_fst_unsqueeze_axis = ngraph::pattern::wrap_type<opset5::Constant>();
    auto m_fst_unsqueeze = ngraph::pattern::wrap_type<opset5::Unsqueeze>({gather_ie, fst_unsqueeze_axis}, pattern::consumers_count(1));
    auto m_concat_axis = ngraph::pattern::wrap_type<opset5::Constant>();
    auto m_concat = ngraph::pattern::wrap_type<opset5::Concat>({fst_unsqueeze, concat_axis}, pattern::consumers_count(1));
    auto m_fst_convert = ngraph::pattern::wrap_type<opset5::Convert>({concat}, pattern::consumers_count(1));
    auto m_reduce_axes = ngraph::pattern::wrap_type<opset5::Constant>();
    auto m_reduce =  ngraph::pattern::wrap_type<opset5::ReduceMin>({fst_convert, reduce_axes}, pattern::consumers_count(1));
    auto m_snd_convert = ngraph::pattern::wrap_type<opset5::Convert>({reduce}, pattern::consumers_count(1));
    auto m_snd_unsqueeze_axis = ngraph::pattern::wrap_type<opset5::Constant>();
    auto m_snd_unsqueeze = ngraph::pattern::wrap_type<opset5::Unsqueeze>({snd_convert, snd_unsqueeze_axis});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher & m) -> bool {
        auto & label_to_output = m.get_pattern_value_map();

        auto shape_of = label_to_output[m_shape_of].get_node_shared_ptr();
        auto data = label_to_output[m_data].get_node_shared_ptr();
        auto data_shape = data->get_partial_shape();

        if (data_shape.rank().is_dynamic() || data_shape.is_dynamic()) {
            return false;
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(m_mul, "PatternBeforeTopKFusion");
    this->register_matcher(m, callback);
}
