// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

#include "ngraph_ops/scaleshift.hpp"
#include "ngraph_ops/eltwise.hpp"
#include "ngraph_ops/power.hpp"

#include "transformations/utils/utils.hpp"

#include "transformations/convert_opset1_to_legacy/convert_mul_add_to_scaleshift_or_power.hpp"

namespace ngraph {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertMulOrAddFinally);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertMulOrAddFinally: public ngraph::pass::GraphRewrite {
public:
    // This pass finally converts single Multiply and Add operations to ScaleShift or Power operation
    ConvertMulOrAddFinally() : GraphRewrite() {
        convert_mul_or_add_finally<ngraph::opset1::Add>();
        convert_mul_or_add_finally<ngraph::opset1::Multiply>();
    }

private:
    template<typename T>
    void convert_mul_or_add_finally();
};

template <typename T>
bool convert_to_eltwise(std::shared_ptr<T> & node,
                        ngraph::Output<ngraph::Node> data1,
                        ngraph::Output<ngraph::Node> data2) {
    ELTWISE_TYPE et;
    if (std::is_same<T, ngraph::opset1::Multiply>()) {
        et = ELTWISE_TYPE::Prod;
    } else if (std::is_same<T, ngraph::opset1::Add>()) {
        et = ELTWISE_TYPE::Sum;
    } else {
        return false;
    }

    auto eltwise = std::make_shared<ngraph::op::Eltwise>(data1, data2, et);
    eltwise->set_friendly_name(node->get_friendly_name());
    ngraph::copy_runtime_info(node, eltwise);
    ngraph::replace_node(node, eltwise);
    return true;
}

template <typename T>
ngraph::graph_rewrite_callback get_callback() {
    ngraph::graph_rewrite_callback callback = [](ngraph::pattern::Matcher& m) {
        static_assert(std::is_same<T, ngraph::opset1::Add>() || std::is_same<T, ngraph::opset1::Multiply>(),
                      "Unsupported template parameter. Only Add or Multiply allowed!");

        auto lin_op = std::dynamic_pointer_cast<T> (m.get_match_root());
        if (!lin_op || lin_op->output(0).get_partial_shape().rank().is_dynamic()) {
            return false;
        }

        const auto output_shape = lin_op->output(0).get_partial_shape();
        const auto output_shape_rank = output_shape.rank().get_length();

        if (!lin_op->get_element_type().is_real()) {
            return convert_to_eltwise<T>(lin_op,
                                         lin_op->input(0).get_source_output(),
                                         lin_op->input(1).get_source_output());
        }

        std::shared_ptr<ngraph::opset1::Constant> const_node = std::dynamic_pointer_cast<ngraph::opset1::Constant>(
                lin_op->input(0).get_source_output().get_node_shared_ptr());
        auto data_node = lin_op->input(1).get_source_output();
        if (!const_node) {
            const_node = std::dynamic_pointer_cast<ngraph::opset1::Constant> (lin_op->input(1).get_source_output().get_node_shared_ptr());
            data_node = lin_op->input(0).get_source_output();
            if (!const_node) {
                return convert_to_eltwise<T>(lin_op,
                                             lin_op->input(0).get_source_output(),
                                             lin_op->input(1).get_source_output());
            }
        }

        auto constant_broadcast_output = [](ngraph::PartialShape data_pshape, ngraph::Shape const_shape) -> bool {
            if (data_pshape.rank().is_dynamic() || const_shape.size() > data_pshape.rank().get_length()) {
                return true;
            }

            std::vector<ngraph::Dimension> data_shape(data_pshape);

            auto const_shape_it = const_shape.rbegin();
            auto data_shape_it = data_shape.rbegin();

            while (const_shape_it != const_shape.rend()) {
                auto data_dim = *data_shape_it;
                auto const_dim = *const_shape_it;

                /* DATA DIM - CONST DIM - CONSTANT BROADCAST OUTPUT
                   DYN      - 64        - TRUE
                   DYN      - 1         - FALSE
                   64       - 1         - FALSE
                   1        - 64        - TRUE
                   64       - 64        - FALSE
                */
                if ((data_dim.is_dynamic() && const_dim != 1) ||
                    (data_dim.is_static() && data_dim.get_length() == 1 && const_dim != 1)) {
                    return true;
                }

                ++const_shape_it;
                ++data_shape_it;
            }

            return false;
        };

        // Check that eltwise is not useless and do not broadcast output otherwise we remove it
        if (((std::is_same<T, ngraph::opset1::Add>() && ngraph::op::util::constantIsEqualTo(const_node, 0)) ||
            (std::is_same<T, ngraph::opset1::Multiply>() && ngraph::op::util::constantIsEqualTo(const_node, 1))) &&
            !constant_broadcast_output(data_node.get_partial_shape(), const_node->get_shape())) {
            bool has_result_output = false;
            for (const auto & output : lin_op->output(0).get_target_inputs()) {
                if (dynamic_cast<ngraph::op::Result*>(output.get_node())) {
                    has_result_output = true;
                }
            }

            auto parent = data_node.get_node_shared_ptr();
            size_t consumers_count = 0;
            for (const auto &output : parent->outputs()) {
                consumers_count += output.get_target_inputs().size();
            }

            if (!has_result_output || consumers_count == 1) {
                if (!std::dynamic_pointer_cast<ngraph::op::Parameter>(parent)) {
                    parent->set_friendly_name(lin_op->get_friendly_name());
                }
                // TODO: due to ngraph::replace_node function limitations we have to reconnect output port consumers to the new input
                // using replace_source_output method
                for (auto &input : lin_op->output(0).get_target_inputs()) {
                    input.replace_source_output(data_node);
                }
                return true;
            }
        }


        auto res = check_constant(const_node, data_node.get_partial_shape());

        if (res == CONVERSION_RESULT::NONE || (res == CONVERSION_RESULT::SCALE_SHIFT && output_shape_rank < 4)) {
            return convert_to_eltwise<T>(lin_op,
                                         lin_op->input(0).get_source_output(),
                                         lin_op->input(1).get_source_output());
        }

        // TODO: if all values in Constant are equal the best way is to convert this Eltwise to Power
        if (res == CONVERSION_RESULT::SCALE_SHIFT) {
            auto weights_et = const_node->get_element_type();
            auto weights_shape = const_node->get_shape();

            // In case of Add we create fake weights with 1, in case of Multiply we create fake bias with 0
            std::shared_ptr<ngraph::op::ScaleShiftIE> scaleshift;
            if (std::is_same<T, ngraph::opset1::Add>()) {
                auto weights = ngraph::opset1::Constant::create(weights_et, weights_shape, {1});
                scaleshift = std::make_shared<ngraph::op::ScaleShiftIE>(data_node, ngraph::op::util::normalize_constant(weights, output_shape),
                                                                                   ngraph::op::util::normalize_constant(const_node, output_shape));
            } else {
                auto bias = ngraph::opset1::Constant::create(weights_et, weights_shape, {0});
                scaleshift = std::make_shared<ngraph::op::ScaleShiftIE>(data_node, ngraph::op::util::normalize_constant(const_node, output_shape),
                                                                                   ngraph::op::util::normalize_constant(bias, output_shape));
            }

            scaleshift->set_friendly_name(lin_op->get_friendly_name());
            ngraph::copy_runtime_info(m.get_match_root(), scaleshift);
            ngraph::replace_node(m.get_match_root(), scaleshift);
        } else {
            float value;
            if (!ngraph::op::util::get_single_value(const_node, value)) {
                return false;
            }

            // In case Add we create fake scale equal to 1, in case of Multiply we create fake shift equal to 0
            std::shared_ptr<ngraph::op::PowerIE> power;
            if (std::is_same<T, ngraph::opset1::Add>()) {
                power = std::make_shared<ngraph::op::PowerIE>(data_node, 1., 1., value);
            } else if (std::is_same<T, ngraph::opset1::Multiply>()) {
                power = std::make_shared<ngraph::op::PowerIE>(data_node, 1., value, 0.);
            } else {
                return false;
            }
            power->set_friendly_name(lin_op->get_friendly_name());
            ngraph::copy_runtime_info(m.get_match_root(), power);
            ngraph::replace_node(m.get_match_root(), power);
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
