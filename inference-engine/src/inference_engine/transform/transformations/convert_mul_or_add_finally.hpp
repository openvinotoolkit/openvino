// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ngraph/pass/graph_rewrite.hpp>

#include <ngraph/op/add.hpp>
#include <ngraph/op/multiply.hpp>
#include "ngraph/op/constant.hpp"
#include <ngraph/op/experimental/dyn_broadcast.hpp>

#include <ngraph_ops/scaleshift.hpp>
#include <ngraph_ops/eltwise.hpp>
#include <ngraph_ops/power.hpp>

#include "convert_mul_add_to_scaleshift_or_power.hpp"


namespace ngraph {
namespace pass {

class ConvertMulOrAddFinally;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertMulOrAddFinally: public ngraph::pass::GraphRewrite {
public:
    // This pass finally converts single Multiply and Add operations to ScaleShift or Power operation
    ConvertMulOrAddFinally() : GraphRewrite() {
        convert_mul_or_add_finally<ngraph::op::Add>();
        convert_mul_or_add_finally<ngraph::op::Multiply>();
    }

private:
    template<typename T>
    void convert_mul_or_add_finally();
};

template <typename T>
bool convert_to_eltwise(std::shared_ptr<T> & node,
                        std::shared_ptr<ngraph::Node> data1,
                        std::shared_ptr<ngraph::Node> data2) {
    ELTWISE_TYPE et;
    if (std::is_same<T, ngraph::op::Multiply>()) {
        et = ELTWISE_TYPE::Prod;
    } else if (std::is_same<T, ngraph::op::Add>()) {
        et = ELTWISE_TYPE::Sum;
    } else {
        return false;
    }

    auto eltwise = std::make_shared<ngraph::op::Eltwise>(data1, data2, et);
    eltwise->set_friendly_name(node->get_friendly_name());
    ngraph::replace_node(node, std::dynamic_pointer_cast<ngraph::Node>(eltwise));
    return true;
}

template <typename T>
ngraph::graph_rewrite_callback get_callback() {
    ngraph::graph_rewrite_callback callback = [](ngraph::pattern::Matcher& m) {
        static_assert(std::is_same<T, ngraph::op::Add>() || std::is_same<T, ngraph::op::Multiply>(),
                      "Unsupported template parameter. Only Add or Multiply allowed!");

        auto lin_op = std::dynamic_pointer_cast<T> (m.get_match_root());
        if (!lin_op) {
            return false;
        }

        // Until Eltwise has no broadcast support we are checking DynBroadcast instead of Constant
        auto broadcast1 = std::dynamic_pointer_cast<ngraph::op::DynBroadcast> (lin_op->get_argument(0));
        auto broadcast2 = std::dynamic_pointer_cast<ngraph::op::DynBroadcast> (lin_op->get_argument(1));

        if (!broadcast1 && !broadcast2) {
            return convert_to_eltwise<T>(lin_op,
                                         lin_op->get_argument(0),
                                         lin_op->get_argument(1));
        }

        // In case of two broadcasts we expect only one with constant input
        auto res1 = check_dyn_broadcast(broadcast1);
        auto res2 = check_dyn_broadcast(broadcast2);

        if (broadcast1 && broadcast2) {
            if ((res1 == CONVERSION_RESULT::NONE && res2 == CONVERSION_RESULT::NONE) ||
                (res1 != CONVERSION_RESULT::NONE && res2 != CONVERSION_RESULT::NONE)) {
                // In case if both are NONE or both can be converted to smth it will be Eltwise
                return convert_to_eltwise<T>(lin_op,
                                             broadcast1->get_inputs()[0].get_output().get_node(),
                                             broadcast2->get_inputs()[0].get_output().get_node());
            }
        }

        if (broadcast1 && !broadcast2) {
            if (res1 == CONVERSION_RESULT::NONE) {
                return convert_to_eltwise<T>(lin_op,
                                             broadcast1->get_inputs()[0].get_output().get_node(),
                                             lin_op->get_argument(1));
            }
        }

        if (!broadcast1 && broadcast2) {
            if (res2 == CONVERSION_RESULT::NONE) {
                return convert_to_eltwise<T>(lin_op,
                                             lin_op->get_argument(0),
                                             broadcast2->get_inputs()[0].get_output().get_node());
            }
        }

        std::shared_ptr<ngraph::op::Constant> constant;
        std::shared_ptr<ngraph::Node> data;
        CONVERSION_RESULT final_result = CONVERSION_RESULT::NONE;

        if (res1 != CONVERSION_RESULT::NONE) {
            data = lin_op->get_argument(1);
            constant = std::dynamic_pointer_cast<ngraph::op::Constant>(broadcast1->get_inputs()[0].get_output().get_node());
            final_result = res1;
        }

        if (res2 != CONVERSION_RESULT::NONE) {
            data = lin_op->get_argument(0);
            constant = std::dynamic_pointer_cast<ngraph::op::Constant>(broadcast2->get_inputs()[0].get_output().get_node());
            final_result = res2;
        }

        if (final_result == CONVERSION_RESULT::SCALE_SHIFT) {
            auto weights_et = constant->get_element_type();
            auto weights_shape = constant->get_shape();

            // TODO: Currently supports only FP32
            if (weights_et != ngraph::element::f32) return false;

            //  Fill weights with fake values
            std::vector<float> weights_values = constant->get_vector<float>();

            // In case of Add we create fake weights with 1, in case of Multiply we create fake bias with 0
            std::shared_ptr<ngraph::op::ScaleShiftIE> scaleshift;
            if (std::is_same<T, ngraph::op::Add>()) {
                std::fill(weights_values.begin(), weights_values.end(), 1.);
                auto weights = std::make_shared<ngraph::op::Constant>(weights_et, weights_shape, weights_values);
                scaleshift = std::make_shared<ngraph::op::ScaleShiftIE>(data, normalize_constant(weights, lin_op->get_shape()),
                                                                              normalize_constant(constant, lin_op->get_shape()));
            } else {
                std::fill(weights_values.begin(), weights_values.end(), 0.);
                auto bias = std::make_shared<ngraph::op::Constant>(weights_et, weights_shape, weights_values);
                scaleshift = std::make_shared<ngraph::op::ScaleShiftIE>(data, normalize_constant(constant, lin_op->get_shape()),
                                                                              normalize_constant(bias, lin_op->get_shape()));
            }

            scaleshift->set_friendly_name(lin_op->get_friendly_name());
            ngraph::replace_node(m.get_match_root(), std::dynamic_pointer_cast<ngraph::Node>(scaleshift));
        } else {
            // TODO: Currently supports only FP32
            if (constant->get_element_type() != ngraph::element::f32) return false;

            // In case Add we create fake scale equal to 1, in case of Multiply we create fake shift equal to 0
            std::shared_ptr<ngraph::op::PowerIE> power;
            if (std::is_same<T, ngraph::op::Add>()) {
                power = std::make_shared<ngraph::op::PowerIE>(data,
                                                            1.,
                                                            1.,
                                                            *constant->get_vector<float>().begin());
            } else {
                power = std::make_shared<ngraph::op::PowerIE>(data,
                                                            1.,
                                                            *constant->get_vector<float>().begin(),
                                                            0.);
            }
            power->set_friendly_name(lin_op->get_friendly_name());
            ngraph::replace_node(m.get_match_root(), std::dynamic_pointer_cast<ngraph::Node>(power));
        }

        return true;
    };
    return callback;
}

template <typename T>
void ngraph::pass::ConvertMulOrAddFinally::convert_mul_or_add_finally() {
    auto data_batch_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 2, 1, 1});
    auto data_batch_2 = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 2, 1, 1});

    auto lin_op = std::make_shared<T>(data_batch_1, data_batch_2);

    auto m = std::make_shared<ngraph::pattern::Matcher>(lin_op);
    this->add_matcher(m, get_callback<T>(), PassProperty::CHANGE_DYNAMIC_STATE);
}
