// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_power_sum_add_rsqrt_to_normalizel2.hpp"

#include <memory>
#include <vector>
#include <ngraph/ngraph.hpp>

bool ngraph::pass::ConvertPowerSumAddRsqrtToNormalizeL2::is_applicable(pattern::Matcher& m) {
    auto mul = std::dynamic_pointer_cast<ngraph::opset1::Multiply>(m.get_match_root());
    if (!mul) {
        return false;
    }

    auto inv_node = mul->input(0).get_source_output().get_node_shared_ptr();
    auto inv      = ngraph::as_type_ptr<ngraph::opset1::Power>(inv_node);
    if (!inv) {
        return false;
    }

    auto sqrt_node = inv->input(0).get_source_output().get_node_shared_ptr();
    auto sqrt      = ngraph::as_type_ptr<ngraph::opset1::Power>(sqrt_node);
    if (!sqrt) {
        return false;
    }

    auto add_node = sqrt->input(0).get_source_output().get_node_shared_ptr();
    auto add      = ngraph::as_type_ptr<ngraph::opset1::Add>(add_node);
    if (!add) {
        return false;
    }

    auto reduce_sum_node = add->input(0).get_source_output().get_node_shared_ptr();
    auto reduce_sum      = ngraph::as_type_ptr<ngraph::opset1::ReduceSum>(reduce_sum_node);
    if (!reduce_sum) {
        return false;
    }

    auto square_node = reduce_sum->input(0).get_source_output().get_node_shared_ptr();
    auto square      = ngraph::as_type_ptr<ngraph::opset1::Power>(square_node);
    if (!square) {
        return false;
    }

    auto minus_one = inv->input_value(1).get_node_shared_ptr();
    if (!minus_one->is_constant()) {
        return false;
    }

    auto half = sqrt->input_value(1).get_node_shared_ptr();
    if (!half->is_constant()) {
        return false;
    }

    auto epsilon = add->input_value(1).get_node_shared_ptr();
    if (!epsilon->is_constant()) {
        return false;
    }

    auto axes = reduce_sum->input_value(1).get_node_shared_ptr();
    if (!axes->is_constant()) {
        return false;
    }

    auto two = square->input_value(1).get_node_shared_ptr();
    if (!two->is_constant()) {
        return false;
    }

    if (!reduce_sum->get_keep_dims()) {
        return false;
    }
    return true;
}

void ngraph::pass::ConvertPowerSumAddRsqrtToNormalizeL2::convert_to_normalize_l2() {
    auto data        = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto square_data = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto square      = std::make_shared<ngraph::opset1::Power>(data, square_data);
    auto axes        = std::make_shared<pattern::op::Label>(element::i64, Shape{4});
    auto reduce_sum  = std::make_shared<ngraph::opset1::ReduceSum>(square, axes);
    auto epsilon     = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto add         = std::make_shared<ngraph::opset1::Add>(reduce_sum, epsilon);
    auto sqrt_data   = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto sqrt        = std::make_shared<ngraph::opset1::Power>(add, sqrt_data);
    auto minus_one   = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto inv         = std::make_shared<ngraph::opset1::Power>(sqrt, minus_one);
    auto mul         = std::make_shared<ngraph::opset1::Multiply>(data, inv);

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
//        auto mul = std::dynamic_pointer_cast<ngraph::opset1::Multiply>(m.get_match_root());
//        if (!mul) {
//            return false;
//        }
//
//        auto inv_node = mul->input(0).get_source_output().get_node_shared_ptr();
//        auto inv      = ngraph::as_type_ptr<ngraph::opset1::Power>(inv_node);
//        if (!inv) {
//            return false;
//        }
//
//        auto sqrt_node = inv->input(0).get_source_output().get_node_shared_ptr();
//        auto sqrt      = ngraph::as_type_ptr<ngraph::opset1::Power>(sqrt_node);
//        if (!sqrt) {
//            return false;
//        }
//
//        auto add_node = sqrt->input(0).get_source_output().get_node_shared_ptr();
//        auto add      = ngraph::as_type_ptr<ngraph::opset1::Add>(add_node);
//        if (!add) {
//            return false;
//        }
//
//        auto reduce_sum_node = add->input(0).get_source_output().get_node_shared_ptr();
//        auto reduce_sum      = ngraph::as_type_ptr<ngraph::opset1::ReduceSum>(reduce_sum_node);
//        if (!reduce_sum) {
//            return false;
//        }
//        return true;
        bool applicability = is_applicable(m);
        if (!applicability) {
            return false;
        }

        return true;
   };
}
