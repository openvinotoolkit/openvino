// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/reverse_shape_and_type_infer.hpp"

#include "openvino/opsets/opset10.hpp"

#include "itt.hpp"

using namespace ov::opset10;

namespace {
bool inherit_output_shape(std::shared_ptr<ov::Node> node, std::vector<size_t> input_idxs) {
    auto is_changed = false;
    auto output_shape = node->get_output_partial_shape(0);

    for (auto idx : input_idxs) {
        if (node->get_input_partial_shape(idx).rank().is_dynamic()) {
            OPENVINO_SUPPRESS_DEPRECATED_START
            node->get_input_tensor(idx).set_partial_shape(output_shape);
            OPENVINO_SUPPRESS_DEPRECATED_END
            is_changed = true;
        }
    }
    return is_changed;
}

bool inherit_output_rank(std::shared_ptr<ov::Node> node, std::vector<size_t> input_idxs) {
    auto is_changed = false;
    auto output_shape = node->get_output_partial_shape(0);

    for (auto idx : input_idxs) {
        if (idx < node->get_input_size() && node->get_input_partial_shape(idx).rank().is_dynamic()) {
            OPENVINO_SUPPRESS_DEPRECATED_START
            node->get_input_tensor(idx).set_partial_shape(ov::PartialShape::dynamic(output_shape.rank()));
            OPENVINO_SUPPRESS_DEPRECATED_END
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
            OPENVINO_SUPPRESS_DEPRECATED_START
            node->get_input_tensor(idx).set_element_type(output_type);
            OPENVINO_SUPPRESS_DEPRECATED_END
            is_changed = true;
        }
    }
    return is_changed;
}
}  // namespace

bool ov::pass::ReverseShapeAndTypeInfer::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(ReverseShapeAndTypeInfer);
    bool is_changed = false;
    auto ops = f->get_ordered_ops();
    OPENVINO_SUPPRESS_DEPRECATED_START
    for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
        const auto& op = *it;
        auto output_shape = op->get_output_partial_shape(0);
        auto output_type = op->get_output_element_type(0);
        std::cout << op->get_type_name() << std::endl;
        if (const auto& param = std::dynamic_pointer_cast<Parameter>(op)) {
            if (param->get_partial_shape().rank().is_dynamic()) {
                param->set_partial_shape(output_shape);
                is_changed = true;
            }
            if (param->get_element_type().is_dynamic()) {
                param->set_element_type(output_type);
                is_changed = true;
            }
        } else if (std::dynamic_pointer_cast<Convolution>(op) ||
                   std::dynamic_pointer_cast<GroupConvolutionBackpropData>(op) ||
                   std::dynamic_pointer_cast<ConvolutionBackpropData>(op) ||
                   std::dynamic_pointer_cast<GroupConvolution>(op)) {
            is_changed |= inherit_output_rank(op, {0, 1});
            is_changed |= inherit_output_type(op, {0, 1});
        } else if (std::dynamic_pointer_cast<DeformableConvolution>(op)) {
            is_changed |= inherit_output_rank(op, {0, 1, 2, 3});
            is_changed |= inherit_output_type(op, {0, 1, 2, 3});
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
        } else if (std::dynamic_pointer_cast<op::util::UnaryElementwiseArithmetic>(op)) {
            is_changed |= inherit_output_shape(op, {0});
            is_changed |= inherit_output_type(op, {0});
        }
    }
    OPENVINO_SUPPRESS_DEPRECATED_END
    return is_changed;
}
