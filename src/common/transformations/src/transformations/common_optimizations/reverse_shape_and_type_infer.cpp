// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/reverse_shape_and_type_infer.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/util/pad_base.hpp"
#include "openvino/opsets/opset10.hpp"

using namespace ov::opset10;

bool ov::pass::ReverseShapeAndTypeInfer::inherit_output_shape(const std::shared_ptr<ov::Node>& node,
                                                              const std::vector<size_t>& input_idxs) {
    auto is_changed = false;
    auto output_shape = node->get_output_partial_shape(0);

    for (auto idx : input_idxs) {
        if (idx < node->get_input_size() && node->get_input_partial_shape(idx).rank().is_dynamic()) {
            node->get_input_tensor(idx).m_partial_shape = output_shape;
            is_changed = true;
        }
    }
    return is_changed;
}

bool ov::pass::ReverseShapeAndTypeInfer::inherit_output_rank(const std::shared_ptr<ov::Node>& node,
                                                             const std::vector<size_t>& input_idxs) {
    auto is_changed = false;
    auto output_shape = node->get_output_partial_shape(0);

    for (auto idx : input_idxs) {
        if (idx < node->get_input_size() && node->get_input_partial_shape(idx).rank().is_dynamic()) {
            node->get_input_tensor(idx).m_partial_shape = ov::PartialShape::dynamic(output_shape.rank());
            is_changed = true;
        }
    }
    return is_changed;
}

bool ov::pass::ReverseShapeAndTypeInfer::inherit_output_type(const std::shared_ptr<ov::Node>& node,
                                                             const std::vector<size_t>& input_idxs) {
    auto is_changed = false;
    auto output_type = node->get_output_element_type(0);

    for (auto idx : input_idxs) {
        if (idx < node->get_input_size() && node->get_input_element_type(idx).is_dynamic()) {
            node->get_input_tensor(idx).m_element_type = output_type;
            is_changed = true;
        }
    }
    return is_changed;
}

bool ov::pass::ReverseShapeAndTypeInfer::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(ReverseShapeAndTypeInfer);
    bool is_changed = false;
    auto ops = f->get_ordered_ops();
    for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
        const auto& op = *it;
        auto output_shape = op->get_output_partial_shape(0);
        auto output_type = op->get_output_element_type(0);
        if (const auto& param = std::dynamic_pointer_cast<Parameter>(op)) {
            if (param->get_partial_shape().rank().is_dynamic()) {
                param->set_partial_shape(output_shape);
                is_changed = true;
            }
            if (param->get_element_type().is_dynamic()) {
                param->set_element_type(output_type);
                is_changed = true;
            }
        } else if (std::dynamic_pointer_cast<Convolution>(op)) {
            is_changed |= inherit_output_rank(op, {0, 1});
            // Inherit channels from weights
            const auto& weigths_pshape = op->get_input_partial_shape(1);
            if (weigths_pshape.rank().is_static() && op->get_input_partial_shape(1).rank().is_static() &&
                weigths_pshape[1] != 1) {
                op->get_input_tensor(0).m_partial_shape[1] = weigths_pshape[1];
            }
            is_changed |= inherit_output_type(op, {0, 1});
        } else if (std::dynamic_pointer_cast<GroupConvolution>(op)) {
            is_changed |= inherit_output_rank(op, {0, 1});
            // Inherit channels from weights
            const auto& weigths_pshape = op->get_input_partial_shape(1);
            if (weigths_pshape.rank().is_static() && op->get_input_partial_shape(1).rank().is_static() &&
                weigths_pshape[2] != 1) {
                op->get_input_tensor(0).m_partial_shape[1] = weigths_pshape[0] * weigths_pshape[2];
            }
            is_changed |= inherit_output_type(op, {0, 1});
        } else if (std::dynamic_pointer_cast<ConvolutionBackpropData>(op)) {
            is_changed |= inherit_output_rank(op, {0, 1});
            // Inherit channels from weights
            const auto& weigths_pshape = op->get_input_partial_shape(1);
            if (weigths_pshape.rank().is_static() && op->get_input_partial_shape(1).rank().is_static() &&
                weigths_pshape[0] != 1) {
                op->get_input_tensor(0).m_partial_shape[1] = weigths_pshape[0];
            }
            is_changed |= inherit_output_type(op, {0, 1});
        } else if (std::dynamic_pointer_cast<GroupConvolutionBackpropData>(op)) {
            is_changed |= inherit_output_rank(op, {0, 1});
            // Inherit channels from weights
            const auto& weigths_pshape = op->get_input_partial_shape(1);
            if (weigths_pshape.rank().is_static() && op->get_input_partial_shape(1).rank().is_static() &&
                weigths_pshape[1] != 1) {
                op->get_input_tensor(0).m_partial_shape[1] = weigths_pshape[0] * weigths_pshape[1];
            }
            is_changed |= inherit_output_type(op, {0, 1});
        } else if (std::dynamic_pointer_cast<DeformableConvolution>(op)) {
            is_changed |= inherit_output_rank(op, {0, 1, 2, 3});
            is_changed |= inherit_output_type(op, {0, 1, 2, 3});
        } else if (std::dynamic_pointer_cast<ov::op::util::PadBase>(op)) {
            // Shape of pads_begin and pads_end must match rank of input
            if (op->get_input_partial_shape(0).rank().is_dynamic()) {
                auto pads_begin_shape = op->get_input_partial_shape(1);
                auto pads_end_shape = op->get_input_partial_shape(2);
                if (pads_begin_shape.is_static() && pads_begin_shape.size() > 0) {
                    op->get_input_tensor(0).m_partial_shape = PartialShape::dynamic(pads_begin_shape[0]);
                    is_changed = true;
                } else if (pads_end_shape.is_static() && pads_end_shape.size() > 0) {
                    op->get_input_tensor(0).m_partial_shape = PartialShape::dynamic(pads_end_shape[0]);
                    is_changed = true;
                }
            }
            is_changed |= inherit_output_type(op, {0});
        } else if (std::dynamic_pointer_cast<op::util::UnaryElementwiseArithmetic>(op)) {
            is_changed |= inherit_output_shape(op, {0});
            is_changed |= inherit_output_type(op, {0});
        } else if (const auto& eltwise = std::dynamic_pointer_cast<op::util::BinaryElementwiseArithmetic>(op)) {
            if (output_shape.rank().is_static()) {
                auto in0_rank = op->get_input_partial_shape(0).rank();
                auto in1_rank = op->get_input_partial_shape(1).rank();
                if (in0_rank.is_dynamic() && in1_rank.is_static()) {
                    if (eltwise->get_autob() == ov::op::AutoBroadcastType::NONE)
                        op->get_input_tensor(0).m_partial_shape = output_shape;
                    else if (in1_rank.get_length() < output_shape.rank().get_length())
                        op->get_input_tensor(0).m_partial_shape = PartialShape::dynamic(output_shape.rank());
                } else if (in1_rank.is_dynamic() && in0_rank.is_static()) {
                    if (eltwise->get_autob() == ov::op::AutoBroadcastType::NONE)
                        op->get_input_tensor(1).m_partial_shape = output_shape;
                    else if (in0_rank.get_length() < output_shape.rank().get_length())
                        op->get_input_tensor(1).m_partial_shape = PartialShape::dynamic(output_shape.rank());
                }
            }
            is_changed |= inherit_output_type(op, {0, 1});
        } else if (const auto& concat = std::dynamic_pointer_cast<Concat>(op)) {
            std::vector<size_t> input_idxs(op->get_input_size());
            std::iota(input_idxs.begin(), input_idxs.end(), 0);

            auto axis = concat->get_axis();
            if (output_shape.rank().is_static()) {
                if (axis < 0) {
                    axis = output_shape.rank().get_length() + axis;
                }
                auto input_pshape = output_shape;
                input_pshape[axis] = Dimension::dynamic();
                for (auto idx : input_idxs) {
                    if (idx < op->get_input_size() && op->get_input_partial_shape(idx).rank().is_dynamic()) {
                        op->get_input_tensor(idx).m_partial_shape = input_pshape;
                        is_changed = true;
                    }
                }
            }
            is_changed |= inherit_output_type(op, input_idxs);
        } else if (std::dynamic_pointer_cast<Slice>(op)) {
            is_changed |= inherit_output_rank(op, {0});
            is_changed |= inherit_output_type(op, {0});
        } else if (std::dynamic_pointer_cast<Squeeze>(op)) {
            auto in0_rank = op->get_input_partial_shape(0).rank();
            if (output_shape.rank().is_static() && in0_rank.is_dynamic() && op->get_input_size() > 1) {
                auto in1_pshape = op->get_input_partial_shape(1);
                if (in1_pshape.is_static()) {
                    auto num_dims = in1_pshape.size() == 0 ? 1 : in1_pshape[0].get_length();
                    op->get_input_tensor(0).m_partial_shape =
                        PartialShape::dynamic(output_shape.rank().get_length() + num_dims);
                }
            }
            is_changed |= inherit_output_type(op, {0});
        } else if (std::dynamic_pointer_cast<Unsqueeze>(op)) {
            auto in0_rank = op->get_input_partial_shape(0).rank();
            auto in1_pshape = op->get_input_partial_shape(1);
            if (output_shape.rank().is_static() && in0_rank.is_dynamic() && in1_pshape.is_static()) {
                auto num_dims = in1_pshape.size() == 0 ? 1 : in1_pshape[0].get_length();
                op->get_input_tensor(0).m_partial_shape =
                    PartialShape::dynamic(output_shape.rank().get_length() - num_dims);
            }
            is_changed |= inherit_output_type(op, {0});
        } else if (const auto& if_op = std::dynamic_pointer_cast<If>(op)) {
            auto then_body = if_op->get_then_body();
            auto else_body = if_op->get_else_body();
            // First set types and shapes to Result nodes
            const auto& then_body_results = then_body->get_results();
            const auto& else_body_results = else_body->get_results();
            const auto& then_out_desc = if_op->get_output_descriptions(If::THEN_BODY_INDEX);
            const auto& else_out_desc = if_op->get_output_descriptions(If::ELSE_BODY_INDEX);
            for (const auto& out_desc : then_out_desc) {
                const auto& out_indx = out_desc->m_output_index;
                const auto& body_indx = out_desc->m_body_value_index;
                then_body_results[body_indx]->get_input_tensor(0).m_partial_shape =
                    if_op->get_output_partial_shape(out_indx);
                then_body_results[body_indx]->get_input_tensor(0).m_element_type =
                    if_op->get_output_element_type(out_indx);
            }
            for (const auto& out_desc : else_out_desc) {
                const auto& out_indx = out_desc->m_output_index;
                const auto& body_indx = out_desc->m_body_value_index;
                else_body_results[body_indx]->get_input_tensor(0).m_partial_shape =
                    if_op->get_output_partial_shape(out_indx);
                else_body_results[body_indx]->get_input_tensor(0).m_element_type =
                    if_op->get_output_element_type(out_indx);
            }
            is_changed |= run_on_model(then_body);
            is_changed |= run_on_model(else_body);
            auto then_body_params = then_body->get_parameters();
            auto else_body_params = else_body->get_parameters();
            const auto& then_in_desc = if_op->get_input_descriptions(If::THEN_BODY_INDEX);
            const auto& else_in_desc = if_op->get_input_descriptions(If::ELSE_BODY_INDEX);
            for (const auto& in_desc : then_in_desc) {
                const auto& in_indx = in_desc->m_input_index;
                const auto& body_indx = in_desc->m_body_parameter_index;
                if (if_op->get_input_tensor(in_indx).get_partial_shape().rank().is_dynamic()) {
                    if_op->get_input_tensor(in_indx).m_partial_shape =
                        then_body_params.at(body_indx)->get_partial_shape();
                    is_changed = true;
                }
                if (if_op->get_input_tensor(in_indx).get_element_type().is_dynamic()) {
                    if_op->get_input_tensor(in_indx).m_element_type =
                        then_body_params.at(body_indx)->get_element_type();
                    is_changed = true;
                }
            }
            for (const auto& in_desc : else_in_desc) {
                const auto& in_indx = in_desc->m_input_index;
                const auto& body_indx = in_desc->m_body_parameter_index;
                if (if_op->get_input_tensor(in_indx).get_partial_shape().rank().is_dynamic()) {
                    if_op->get_input_tensor(in_indx).m_partial_shape =
                        else_body_params.at(body_indx)->get_partial_shape();
                    is_changed = true;
                }
                if (if_op->get_input_tensor(in_indx).get_element_type().is_dynamic()) {
                    if_op->get_input_tensor(in_indx).m_element_type =
                        else_body_params.at(body_indx)->get_element_type();
                    is_changed = true;
                }
            }
            // Set type for If condition
            if (if_op->get_input_element_type(0).is_dynamic()) {
                if_op->get_input_tensor(0).m_element_type = element::boolean;
                is_changed = true;
            }
        } else if (std::dynamic_pointer_cast<ConvertLike>(op)) {
            is_changed |= inherit_output_shape(op, {0});
            is_changed |= inherit_output_type(op, {1});
        } else if (std::dynamic_pointer_cast<Transpose>(op)) {
            OPENVINO_SUPPRESS_DEPRECATED_START
            auto transpose_order = get_constant_from_source(op->input_value(1));
            OPENVINO_SUPPRESS_DEPRECATED_END
            if (output_shape.rank().is_static()) {
                if (transpose_order) {
                    // set more precise dimensions during reverse infer
                    // if transpose order is known
                    int64_t rank_length = output_shape.rank().get_length();
                    op->get_input_tensor(0).m_partial_shape = PartialShape::dynamic(output_shape.rank());
                    auto order_value = transpose_order->cast_vector<int64_t>();
                    OPENVINO_ASSERT(order_value.size() == static_cast<size_t>(rank_length),
                                    "The length of Transpose order and the input rank mismatch");
                    for (int64_t dim_idx = 0; dim_idx < rank_length; ++dim_idx) {
                        OPENVINO_ASSERT(0 <= order_value[dim_idx] && order_value[dim_idx] < rank_length,
                                        "Transpose order is out-of-range");
                        op->get_input_tensor(0).m_partial_shape[order_value[dim_idx]] = output_shape[dim_idx];
                    }
                    is_changed = true;
                } else {
                    is_changed |= inherit_output_rank(op, {0});
                }
            } else if (transpose_order) {
                auto order_value = transpose_order->cast_vector<int64_t>();
                op->get_input_tensor(0).m_partial_shape = PartialShape::dynamic(order_value.size());
                is_changed = true;
            }
            is_changed |= inherit_output_type(op, {0});
        }
    }
    return is_changed;
}
