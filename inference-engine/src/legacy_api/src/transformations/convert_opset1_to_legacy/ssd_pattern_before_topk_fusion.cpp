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
    auto data = ngraph::pattern::any_input();
    auto shape_of = ngraph::pattern::wrap_type<opset5::ShapeOf>({data}, pattern::consumers_count(1));
    auto indices = ngraph::pattern::wrap_type<opset5::Constant>();
    auto gather_axis = ngraph::pattern::wrap_type<opset5::Constant>();
    auto gather_ie = ngraph::pattern::wrap_type<op::GatherIE>({shape_of, indices, gather_axis}, pattern::consumers_count(1));
    auto fst_unsqueeze_axis = ngraph::pattern::wrap_type<opset5::Constant>();
    auto fst_unsqueeze = ngraph::pattern::wrap_type<opset5::Unsqueeze>({gather_ie, fst_unsqueeze_axis}, pattern::consumers_count(1));
    auto concat_axis = ngraph::pattern::wrap_type<opset5::Constant>();
    auto concat = ngraph::pattern::wrap_type<opset5::Concat>({fst_unsqueeze, concat_axis}, pattern::consumers_count(1));
    auto fst_convert = ngraph::pattern::wrap_type<opset5::Convert>({concat}, pattern::consumers_count(1));
    auto reduce_axes = ngraph::pattern::wrap_type<opset5::Constant>();
    auto reduce =  ngraph::pattern::wrap_type<opset5::ReduceMin>({fst_convert, reduce_axes}, pattern::consumers_count(1));
    auto snd_convert = ngraph::pattern::wrap_type<opset5::Convert>({reduce}, pattern::consumers_count(1));
    auto snd_unsqueeze_axis = ngraph::pattern::wrap_type<opset5::Constant>();
    auto snd_unsqueeze = ngraph::pattern::wrap_type<opset5::Unsqueeze>({snd_convert, snd_unsqueeze_axis});
}
