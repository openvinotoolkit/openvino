// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/reverse_shape_and_type_infer.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/deformable_convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/pad_base.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"
#include "transformations/utils/utils.hpp"

namespace {

void set_source_output_type_shape(const ov::Node& node,
                                  const ov::element::Type& et,
                                  const ov::PartialShape& new_shape,
                                  const size_t port) {
    const auto source_output = node.get_input_source_output(port);
    source_output.get_node()->set_output_type(source_output.get_index(), et, new_shape);
}

void set_source_output_shape(const ov::Node& node, const ov::PartialShape& new_shape, const size_t port) {
    set_source_output_type_shape(node, node.get_input_element_type(port), new_shape, port);
}

void set_source_output_type(const ov::Node& node, const ov::element::Type& et, const size_t port) {
    set_source_output_type_shape(node, et, node.get_input_partial_shape(port), port);
}
}  // namespace

bool ov::pass::ReverseShapeAndTypeInfer::inherit_output_shape(const std::shared_ptr<ov::Node>& node,
                                                              const std::vector<size_t>& input_idxs) {
    auto is_changed = false;
    auto output_shape = node->get_output_partial_shape(0);

    for (auto idx : input_idxs) {
        if (idx < node->get_input_size() && node->get_input_partial_shape(idx).compatible(output_shape)) {
            auto new_shape = node->get_input_partial_shape(idx);
            PartialShape::merge_into(new_shape, output_shape);
            set_source_output_shape(*node, new_shape, idx);
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
            set_source_output_shape(*node, ov::PartialShape::dynamic(output_shape.rank()), idx);
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
            set_source_output_type(*node, output_type, idx);
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
        is_changed = ov::op::util::process_subgraph(*this, op) || is_changed;

        auto output_shape = op->get_output_partial_shape(0);
        auto output_type = op->get_output_element_type(0);
        if (const auto& param = ov::as_type_ptr<ov::op::v0::Parameter>(op)) {
            if (param->get_partial_shape().compatible(output_shape)) {
                auto shape = param->get_partial_shape();
                PartialShape::merge_into(shape, output_shape);
                param->set_partial_shape(shape);
                is_changed = true;
            }
            if (param->get_element_type().is_dynamic()) {
                param->set_element_type(output_type);
                is_changed = true;
            }
        } else if (ov::as_type_ptr<ov::op::v1::Convolution>(op)) {
            is_changed |= inherit_output_rank(op, {0, 1});
            // Inherit channels from weights
            const auto& weigths_pshape = op->get_input_partial_shape(1);
            if (weigths_pshape.rank().is_static() && op->get_input_partial_shape(1).rank().is_static() &&
                weigths_pshape[1] != 1) {
                auto new_shape = op->get_input_partial_shape(0);
                new_shape[1] = weigths_pshape[1];
                set_source_output_shape(*op, new_shape, 0);
            }
            is_changed |= inherit_output_type(op, {0, 1});
        } else if (ov::as_type_ptr<ov::op::v1::GroupConvolution>(op)) {
            is_changed |= inherit_output_rank(op, {0, 1});
            // Inherit channels from weights
            const auto& weigths_pshape = op->get_input_partial_shape(1);
            if (weigths_pshape.rank().is_static() && op->get_input_partial_shape(1).rank().is_static() &&
                weigths_pshape[2] != 1) {
                auto new_shape = op->get_input_partial_shape(0);
                new_shape[1] = weigths_pshape[0] * weigths_pshape[2];
                set_source_output_shape(*op, new_shape, 0);
            }
            is_changed |= inherit_output_type(op, {0, 1});
        } else if (ov::as_type_ptr<ov::op::v1::ConvolutionBackpropData>(op)) {
            is_changed |= inherit_output_rank(op, {0, 1});
            // Inherit channels from weights
            const auto& weigths_pshape = op->get_input_partial_shape(1);
            if (weigths_pshape.rank().is_static() && op->get_input_partial_shape(1).rank().is_static() &&
                weigths_pshape[0] != 1) {
                auto new_shape = op->get_input_partial_shape(0);
                new_shape[1] = weigths_pshape[0];
                set_source_output_shape(*op, new_shape, 0);
            }
            is_changed |= inherit_output_type(op, {0, 1});
        } else if (ov::as_type_ptr<ov::op::v1::GroupConvolutionBackpropData>(op)) {
            is_changed |= inherit_output_rank(op, {0, 1});
            // Inherit channels from weights
            const auto& weigths_pshape = op->get_input_partial_shape(1);
            if (weigths_pshape.rank().is_static() && op->get_input_partial_shape(1).rank().is_static() &&
                weigths_pshape[1] != 1) {
                auto new_shape = op->get_input_partial_shape(0);
                new_shape[1] = weigths_pshape[0] * weigths_pshape[1];
                set_source_output_shape(*op, new_shape, 0);
            }
            is_changed |= inherit_output_type(op, {0, 1});
        } else if (ov::as_type_ptr<ov::op::v8::DeformableConvolution>(op)) {
            is_changed |= inherit_output_rank(op, {0, 1, 2, 3});
            is_changed |= inherit_output_type(op, {0, 1, 2, 3});
        } else if (ov::as_type_ptr<ov::op::util::PadBase>(op)) {
            // Shape of pads_begin and pads_end must match rank of input
            if (op->get_input_partial_shape(0).rank().is_dynamic()) {
                auto pads_begin_shape = op->get_input_partial_shape(1);
                auto pads_end_shape = op->get_input_partial_shape(2);
                if (pads_begin_shape.is_static() && pads_begin_shape.size() > 0) {
                    set_source_output_shape(*op, PartialShape::dynamic(pads_begin_shape[0]), 0);
                    is_changed = true;
                } else if (pads_end_shape.is_static() && pads_end_shape.size() > 0) {
                    set_source_output_shape(*op, PartialShape::dynamic(pads_end_shape[0]), 0);
                    is_changed = true;
                }
            }
            is_changed |= inherit_output_type(op, {0});
        } else if (ov::as_type_ptr<op::util::UnaryElementwiseArithmetic>(op)) {
            is_changed |= inherit_output_shape(op, {0});
            is_changed |= inherit_output_type(op, {0});
        } else if (const auto& eltwise = ov::as_type_ptr<op::util::BinaryElementwiseArithmetic>(op)) {
            if (output_shape.rank().is_static()) {
                auto in0_rank = op->get_input_partial_shape(0).rank();
                auto in1_rank = op->get_input_partial_shape(1).rank();
                if (in0_rank.is_dynamic() && in1_rank.is_static()) {
                    if (eltwise->get_autob() == ov::op::AutoBroadcastType::NONE) {
                        set_source_output_shape(*op, output_shape, 0);
                    } else if (in1_rank.get_length() < output_shape.rank().get_length()) {
                        set_source_output_shape(*op, PartialShape::dynamic(output_shape.rank()), 0);
                    }
                } else if (in1_rank.is_dynamic() && in0_rank.is_static()) {
                    if (eltwise->get_autob() == ov::op::AutoBroadcastType::NONE) {
                        set_source_output_shape(*op, output_shape, 1);
                    } else if (in0_rank.get_length() < output_shape.rank().get_length()) {
                        set_source_output_shape(*op, PartialShape::dynamic(output_shape.rank()), 1);
                    }
                }
            }
            is_changed |= inherit_output_type(op, {0, 1});
        } else if (const auto& concat = ov::as_type_ptr<ov::op::v0::Concat>(op)) {
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
                    if (idx < op->get_input_size() && op->get_input_partial_shape(idx).compatible(input_pshape)) {
                        auto new_shape = op->get_input_partial_shape(idx);
                        PartialShape::merge_into(new_shape, input_pshape);
                        set_source_output_shape(*op, new_shape, idx);
                        is_changed = true;
                    }
                }
            }
            is_changed |= inherit_output_type(op, input_idxs);
        } else if (ov::as_type_ptr<ov::op::v8::Slice>(op)) {
            is_changed |= inherit_output_rank(op, {0});
            is_changed |= inherit_output_type(op, {0});
        } else if (ov::as_type_ptr<ov::op::v0::Squeeze>(op)) {
            auto in0_pshape = op->get_input_partial_shape(0);
            auto in0_rank = in0_pshape.rank();
            if (output_shape.rank().is_static()) {
                if (in0_rank.is_dynamic() && op->get_input_size() > 1) {
                    auto in1_pshape = op->get_input_partial_shape(1);
                    if (in1_pshape.is_static()) {
                        auto num_dims = in1_pshape.size() == 0 ? 1 : in1_pshape[0].get_length();
                        set_source_output_shape(*op,
                                                PartialShape::dynamic(output_shape.rank().get_length() + num_dims),
                                                0);
                    }
                } else if (in0_rank.is_static() && op->get_input_size() == 1) {
                    // attempt to create second input
                    std::vector<int64_t> in1_data;
                    for (size_t i = 0; i < in0_pshape.size(); i++) {
                        if (in0_pshape[i] == 1) {
                            in1_data.push_back(i);
                        }
                    }
                    int64_t num_ones = in1_data.size();
                    if (num_ones == in0_rank.get_length() - output_shape.rank().get_length()) {
                        auto axes = ov::op::v0::Constant::create(element::i64, Shape{in1_data.size()}, in1_data);
                        auto new_squeeze = std::make_shared<ov::op::v0::Squeeze>(op->get_input_source_output(0), axes);
                        op->output(0).replace(new_squeeze->output(0));
                        copy_runtime_info(op, new_squeeze);
                    }
                }
            }
            is_changed |= inherit_output_type(op, {0});
        } else if (ov::as_type_ptr<ov::op::v0::Unsqueeze>(op)) {
            auto in0_rank = op->get_input_partial_shape(0).rank();
            auto in1_pshape = op->get_input_partial_shape(1);
            if (output_shape.rank().is_static() && in0_rank.is_dynamic() && in1_pshape.is_static()) {
                auto num_dims = in1_pshape.size() == 0 ? 1 : in1_pshape[0].get_length();
                set_source_output_shape(*op, PartialShape::dynamic(output_shape.rank().get_length() - num_dims), 0);
            }
            is_changed |= inherit_output_type(op, {0});
        } else if (const auto& if_op = ov::as_type_ptr<ov::op::v8::If>(op)) {
            auto then_body = if_op->get_then_body();
            auto else_body = if_op->get_else_body();
            // First set types and shapes to Result nodes
            const auto& then_body_results = then_body->get_results();
            const auto& else_body_results = else_body->get_results();
            const auto& then_out_desc = if_op->get_output_descriptions(ov::op::v8::If::THEN_BODY_INDEX);
            const auto& else_out_desc = if_op->get_output_descriptions(ov::op::v8::If::ELSE_BODY_INDEX);

            for (const auto& out_desc : then_out_desc) {
                const auto& out_indx = out_desc->m_output_index;
                const auto& body_indx = out_desc->m_body_value_index;

                set_source_output_type_shape(*then_body_results[body_indx],
                                             if_op->get_output_element_type(out_indx),
                                             if_op->get_output_partial_shape(out_indx),
                                             0);
            }

            for (const auto& out_desc : else_out_desc) {
                const auto& out_indx = out_desc->m_output_index;
                const auto& body_indx = out_desc->m_body_value_index;
                set_source_output_type_shape(*else_body_results[body_indx],
                                             if_op->get_output_element_type(out_indx),
                                             if_op->get_output_partial_shape(out_indx),
                                             0);
            }
            is_changed |= run_on_model(then_body);
            is_changed |= run_on_model(else_body);
            auto then_body_params = then_body->get_parameters();
            auto else_body_params = else_body->get_parameters();
            const auto& then_in_desc = if_op->get_input_descriptions(ov::op::v8::If::THEN_BODY_INDEX);
            const auto& else_in_desc = if_op->get_input_descriptions(ov::op::v8::If::ELSE_BODY_INDEX);
            for (const auto& in_desc : then_in_desc) {
                const auto& in_indx = in_desc->m_input_index;
                const auto& body_indx = in_desc->m_body_parameter_index;
                if (if_op->get_input_tensor(in_indx).get_partial_shape().rank().is_dynamic() ||
                    if_op->get_input_tensor(in_indx).get_element_type().is_dynamic()) {
                    set_source_output_type_shape(*if_op,
                                                 then_body_params.at(body_indx)->get_element_type(),
                                                 then_body_params.at(body_indx)->get_partial_shape(),
                                                 in_indx);
                    is_changed = true;
                }
            }
            for (const auto& in_desc : else_in_desc) {
                const auto& in_indx = in_desc->m_input_index;
                const auto& body_indx = in_desc->m_body_parameter_index;
                if (if_op->get_input_tensor(in_indx).get_partial_shape().rank().is_dynamic() ||
                    if_op->get_input_tensor(in_indx).get_element_type().is_dynamic()) {
                    set_source_output_type_shape(*if_op,
                                                 then_body_params.at(body_indx)->get_element_type(),
                                                 then_body_params.at(body_indx)->get_partial_shape(),
                                                 in_indx);
                    is_changed = true;
                }
            }
            // Set type for If condition
            if (if_op->get_input_element_type(0).is_dynamic()) {
                set_source_output_type(*if_op, element::boolean, 0);
                is_changed = true;
            }

            // in case TensorFlow models, we can deduce predicate shape that must be a scalar
            // If operations created by fusing Switch-Merge sub-graph contain tf_switch_merge_if rt-info
            if (if_op->get_rt_info().count("tf_switch_merge_if") &&
                if_op->get_rt_info()["tf_switch_merge_if"].as<bool>() &&
                if_op->input_value(0).get_partial_shape().rank().is_dynamic()) {
                set_source_output_shape(*if_op, PartialShape{}, 0);
                is_changed = true;
            }
        } else if (ov::as_type_ptr<ov::op::v1::ConvertLike>(op)) {
            is_changed |= inherit_output_shape(op, {0});
            is_changed |= inherit_output_type(op, {1});
        } else if (ov::as_type_ptr<ov::op::v1::Transpose>(op)) {
            auto transpose_order = ov::util::get_constant_from_source(op->input_value(1));
            if (output_shape.rank().is_static()) {
                if (transpose_order) {
                    // set more precise dimensions during reverse infer
                    // if transpose order is known
                    int64_t rank_length = output_shape.rank().get_length();
                    auto new_shape = op->get_input_partial_shape(0);
                    PartialShape::merge_into(new_shape, PartialShape::dynamic(output_shape.rank()));
                    auto order_value = transpose_order->cast_vector<int64_t>();
                    OPENVINO_ASSERT(order_value.size() == static_cast<size_t>(rank_length),
                                    "The length of Transpose order and the input rank mismatch");
                    for (int64_t dim_idx = 0; dim_idx < rank_length; ++dim_idx) {
                        OPENVINO_ASSERT(0 <= order_value[dim_idx] && order_value[dim_idx] < rank_length,
                                        "Transpose order is out-of-range");
                        new_shape[order_value[dim_idx]] = output_shape[dim_idx];
                    }
                    set_source_output_shape(*op, new_shape, 0);
                    is_changed = true;
                } else {
                    is_changed |= inherit_output_rank(op, {0});
                }
            } else if (transpose_order) {
                auto order_value = transpose_order->cast_vector<int64_t>();
                set_source_output_shape(*op, PartialShape::dynamic(order_value.size()), 0);
                is_changed = true;
            }
            is_changed |= inherit_output_type(op, {0});
        }
    }
    return is_changed;
}
