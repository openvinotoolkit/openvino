// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/reverse_shape_and_type_infer.hpp"

#include <memory>
#include <ngraph/graph_util.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/validation_util.hpp>
#include <openvino/opsets/opset10.hpp>

#include "itt.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::opset10;

namespace {
bool inherit_output_shape(std::shared_ptr<ov::Node> node, std::vector<size_t> input_idxs) {
    auto is_changed = false;
    auto output_shape = node->get_output_partial_shape(0);

    for (auto idx : input_idxs) {
        if (node->get_input_partial_shape(idx).rank().is_dynamic()) {
            node->get_input_tensor(idx).set_partial_shape(output_shape);
            is_changed = true;
        }
    }
    return is_changed;
}

bool inherit_output_type(std::shared_ptr<ov::Node> node, std::vector<size_t> input_idxs) {
    auto is_changed = false;
    auto output_type = node->get_output_element_type(0);

    for (auto idx : input_idxs) {
        if (node->get_input_element_type(idx).is_dynamic()) {
            node->get_input_tensor(idx).set_element_type(output_type);
            is_changed = true;
        }
    }
    return is_changed;
}
}  // namespace

bool ov::pass::ReverseShapeAndTypeInfer::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(ReverseShapeAndTypeInfer);
    bool is_changed = false;
    auto ops = f->get_ordered_ops();
    for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
        const auto& op = *it;
        auto output_shape = op->get_output_partial_shape(0);
        auto output_type = op->get_output_element_type(0);
        if (std::dynamic_pointer_cast<Convolution>(op)) {
            if (op->get_input_partial_shape(0).rank().is_dynamic() && output_shape.rank().is_static()) {
                op->get_input_tensor(0).set_partial_shape(PartialShape::dynamic(output_shape.rank().get_length()));
                is_changed = true;
            }
            is_changed |= inherit_output_type(op, {0});
        } else if (std::dynamic_pointer_cast<Pad>(op)) {
            // Shape of pads_begin and pads_end must match rank of input
            if (op->get_input_partial_shape(0).rank().is_dynamic()) {
                auto pads_begin_shape = op->get_input_partial_shape(1);
                auto pads_end_shape = op->get_input_partial_shape(2);
                if (pads_begin_shape.is_static() && pads_begin_shape.size() > 0) {
                    op->get_input_tensor(0).set_partial_shape(PartialShape::dynamic(pads_begin_shape[0]));
                    is_changed = true;
                } else if (pads_end_shape.is_static() && pads_end_shape.size() > 0) {
                    op->get_input_tensor(0).set_partial_shape(PartialShape::dynamic(pads_end_shape[0]));
                    is_changed = true;
                }
            }
            is_changed |= inherit_output_type(op, {0});
        } else if (const auto& param = std::dynamic_pointer_cast<Parameter>(op)) {
            if (param->get_partial_shape().rank().is_dynamic()) {
                param->set_partial_shape(output_shape);
                is_changed = true;
            }
            if (param->get_element_type().is_dynamic()) {
                param->set_element_type(output_type);
                is_changed = true;
            }
        }
    }
    return is_changed;
}
