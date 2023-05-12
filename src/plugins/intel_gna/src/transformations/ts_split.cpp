// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/ts_split.hpp"

#include <openvino/cc/ngraph/itt.hpp>

#include "../debug_new_pass.hpp"
#include "backend/gna_limitations.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"
#include "transformations/utils/transformation_helper.hpp"

using namespace ov;
using namespace ov::opset10;
using namespace ov::pass::pattern;
using namespace ov::intel_gna::pass;
using namespace ov::intel_gna::pass::helper;
using namespace ov::intel_gna::limitations;

namespace {
#if 0
using NodePtr = std::shared_ptr<Node>;

template <typename NodeT>
std::shared_ptr<ov::Node> FindInputNode(ov::Node* node) {
    for (size_t input_idx = 0; input_idx < node->get_input_size(); ++input_idx) {
        std::shared_ptr<ov::Node> input_node = node->get_input_node_shared_ptr(input_idx);
        auto target_node = ov::as_type_ptr<NodeT>(input_node);
        if (target_node)
            return target_node;
    }
    return {};
}

std::shared_ptr<Constant> GetTransposeConstant(Node* node) {
    auto transpose_node = dynamic_cast<Transpose*>(node);
    if (!transpose_node)
        return {};

    auto constant_node = as_type_ptr<Constant>(transpose_node->input_value(1).get_node_shared_ptr());
    if (!constant_node)
        return {};

    return constant_node;
}

Node* FindFirstConsumer(const NodePtr& node) {
    for (size_t output_idx = 0; output_idx < node->get_output_size(); ++output_idx) {
        auto inputs = node->get_output_target_inputs(output_idx);
        if (inputs.empty())
            continue;
        return inputs.begin()->get_node();
    }
    return nullptr;
}

bool HasSameOutputTransposeNodes(const NodePtr& main_node) {
    AxisVector first_transpose_axis_order;
    {
        Node* first_consumer = FindFirstConsumer(main_node);
        if (!first_consumer)
            return false;
        auto constant_node = GetTransposeConstant(first_consumer);
        if (!constant_node)
            return false;
        first_transpose_axis_order = constant_node->get_axis_vector_val();
    }

    for (size_t output_idx = 0; output_idx < main_node->get_output_size(); ++output_idx) {
        for (auto& input : main_node->get_output_target_inputs(output_idx)) {
            auto constant_node = GetTransposeConstant(input.get_node());
            if (!constant_node)
                return false;

            AxisVector transpose_axis_order = constant_node->get_axis_vector_val();
            if (transpose_axis_order.size() != first_transpose_axis_order.size())
                return false;
            if (!std::equal(transpose_axis_order.begin(),
                            transpose_axis_order.end(),
                            first_transpose_axis_order.begin()))
                return false;
        }
    }
    return true;
}

bool HasInputSplitAndTransposeSiblings(const Output<Node>& output) {
    NodePtr main_node = FindInputNode<Split>(output.get_node());
    if (!main_node) {
        main_node = FindInputNode<VariadicSplit>(output.get_node());
    }
    if (!main_node) {
        return false;
    }

    return HasSameOutputTransposeNodes(main_node);
}

bool IsSplitSinked(const Output<Node>& output) {
    return HasInputSplitAndTransposeSiblings(output) && is_sinking_node(output);
}
#endif

bool is_sinked(const Output<Node>& output) {
    auto split_node = output.get_node_shared_ptr();
    for (size_t output_idx = 0; output_idx < split_node->get_output_size(); ++output_idx) {
        for (auto& input : split_node->get_output_target_inputs(output_idx)) {
            auto transpose = ov::as_type_ptr<Transpose>(input.get_node()->shared_from_this());
            if (transpose && !is_transpose_supported(transpose))
                return true;
        }
    }
    return false;
}

std::vector<size_t> CreateGatherIndices(const ov::Shape& input_shape, const ov::Shape& order) {
    if (input_shape.size() < 2 || input_shape.size() > 4) {
        THROW_GNA_EXCEPTION << "Usupported shape size: " << input_shape.size();
    }

    ov::Shape input_shape_4d = input_shape;
    ov::Shape order_4d = order;
    // Just to simplify the code we transform all shapes to 4d by adding 1 dimentions at the end
    while (input_shape_4d.size() < 4) {
        input_shape_4d.push_back(1);
        order_4d.push_back(order_4d.size());
    }
    ov::Shape output_shape_4d = TransposeShape(input_shape_4d, order_4d);

    // common case when shape is 4d
    std::vector<size_t> xyz_4d = {input_shape_4d[3] * input_shape_4d[2] * input_shape_4d[1],
                                  input_shape_4d[3] * input_shape_4d[2],
                                  input_shape_4d[3],
                                  1};

    std::vector<size_t> xyz = TransposeShape(xyz_4d, order_4d);
    std::vector<size_t> gather_order;

    for (size_t n = 0; n < output_shape_4d[0]; ++n) {
        for (size_t i = 0; i < output_shape_4d[1]; ++i) {
            for (size_t j = 0; j < output_shape_4d[2]; ++j) {
                for (size_t k = 0; k < output_shape_4d[3]; ++k) {
                    gather_order.push_back(n * xyz[0] + i * xyz[1] + j * xyz[2] + k * xyz[3]);
                }
            }
        }
    }

    return gather_order;
}

}  // namespace

#if 0
/*
 * We follow Transpose operations rather than Split. We cannot create matcher pattern
 * for Split with Transpose outputs since Split can have different number of outputs.
 * We just can:
 * - specify Split as searched node and check if it has transpose outputs
 * - specify Transpose as searched node and check if it has Split input
 * Transformations are called on each found node in sorted order from the start to end
 * of the network. When we proceed Split backward sinking we move input transpose
 * to the input of the Split operation.
 * Consider case Split (1) -> Split (2) -> Transpose
 * If specify Split as main searched node after first transformation work we will have
 * Split (1) -> Transpose -> Split(2)
 * Matcher pass will not call TSSplitBackward since
 * - matcher pattern has no Transpose label
 * - Split (1) has already been proceeded
 * Adding Split(2) into the working queue as register_new_node(split)
 * cannot help us. We just can try to find all input Split operations and add them with
 * register_new_node(). Implemented way is simpler.
 *
 * We sink Transpose through Split operation in a backward way only if all the output
 * nodes are the same Transpose. We can:
 * - clone Split with all outputs except Transpose
 *   causes performance problems
 * - add reversed Transpose operations on all outputs except sinking Transpose
 *   nothing to do with new added output Transposes
 */
TSSplitBackward::TSSplitBackward() {
    MATCHER_SCOPE(TSSplitBackward);

    auto transpose_const_label = wrap_type<Constant>();
    auto transpose_label = wrap_type<Transpose>({any_input(), transpose_const_label}, IsSplitSinked);

    matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose_const_label_node = as_type_ptr<Constant>(pattern_to_output.at(transpose_const_label).get_node_shared_ptr());
        auto transpose_label_node = pattern_to_output.at(transpose_label).get_node();

        NodePtr split = FindInputNode<Split>(transpose_label_node);
        if (!split) {
            split = FindInputNode<VariadicSplit>(transpose_label_node);
        }

        if (!split) {
            return false;
        }

        const Shape& split_input_shape = split->get_input_shape(0);
        const size_t split_input_dims = std::accumulate(split_input_shape.begin(), split_input_shape.end(), 1, std::multiplies<Shape::value_type>());

        auto reshape_input_const = std::make_shared<Constant>(ov::element::i64,
                                                        ov::Shape{1},
                                                        split_input_dims);
        auto reshape_input = std::make_shared<Reshape>(split->input_value(0), reshape_input_const, false);

        std::vector<size_t> gather_indices_value = CreateGatherIndices(reshape_input->get_input_shape(0), transpose_const_label_node->get_axis_vector_val());
        auto gather_axis = std::make_shared<Constant>(ov::element::i64, ov::Shape{}, 0);
        auto gather_indices =
            std::make_shared<Constant>(ov::element::i64, ov::Shape{gather_indices_value.size()}, gather_indices_value);
        auto gather = std::make_shared<Gather>(reshape_input, gather_indices, gather_axis);

        auto split_axis_new = std::make_shared<Constant>(ov::element::i64,
                                                        ov::Shape{},
                                                        0);
        auto split_new = split->clone_with_new_inputs({gather, split_axis_new});
    
        for (size_t i = 0; i < split->get_output_size(); ++i) {
            auto output_target_inputs = split->get_output_target_inputs(i);
            if (output_target_inputs.empty())
                continue;
            auto split_output_transpose_prev = ov::as_type_ptr<Transpose>(output_target_inputs.begin()->get_node()->shared_from_this());
            auto reshape_output_const_new = std::make_shared<Constant>(ov::element::i64,
                                                        ov::Shape{split_output_transpose_prev->get_output_shape(0).size()},
                                                        split_output_transpose_prev->get_output_shape(0));
            auto reshape_output_new = std::make_shared<Reshape>(split_new->output(i), reshape_output_const_new, false);
            ov::replace_node_update_name(split_output_transpose_prev, reshape_output_new);
        }
        return true;
    };

    auto m = std::make_shared<Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
#endif

TSSplitBackward::TSSplitBackward() {
    MATCHER_SCOPE(TSSplitBackward);

    auto split_node_label = wrap_type<Split>(is_sinked);

    matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        auto& split_node_label_output = pattern_to_output.at(split_node_label);
        auto split_node = as_type_ptr<Split>(split_node_label_output.get_node_shared_ptr());

        std::vector<AxisVector> gather_indices_vecs;
        for (size_t output_idx = 0; output_idx < split_node->get_output_size(); ++output_idx) {
            for (auto& input : split_node->get_output_target_inputs(output_idx)) {
                auto transpose = ov::as_type_ptr<Transpose>(input.get_node()->shared_from_this());

                if (transpose && !is_transpose_supported(transpose)) {
                    EMUTEX_DEBUG_CHECKPOINT;
                    auto transpose_const = ov::as_type_ptr<Constant>(transpose->get_input_node_shared_ptr(1));
                    if (!transpose_const)
                        return false;
                    auto gather_indices_value =
                        CreateGatherIndices(transpose->get_input_shape(0), transpose_const->get_axis_vector_val());
                    gather_indices_vecs.push_back(gather_indices_value);
                } else {
                    EMUTEX_DEBUG_CHECKPOINT;
                    const Shape& input_shape = input.get_shape();
                    const size_t input_dims = std::accumulate(input_shape.begin(),
                                                              input_shape.end(),
                                                              1,
                                                              std::multiplies<Shape::value_type>());
                    std::vector<size_t> indices(input_dims);
                    std::iota(indices.begin(), indices.end(), 1);
                    gather_indices_vecs.push_back(indices);
                    continue;
                }
            }
        }

        const Shape& split_input_shape = split_node->get_input_shape(0);
        const size_t split_input_dims = std::accumulate(split_input_shape.begin(),
                                                        split_input_shape.end(),
                                                        1,
                                                        std::multiplies<Shape::value_type>());

        const Shape reshape_input_shape = {1, split_input_dims};
        auto reshape_input_const = std::make_shared<Constant>(ov::element::i64, ov::Shape{2}, reshape_input_shape);
        auto reshape_input = std::make_shared<Reshape>(split_node->input_value(0), reshape_input_const, false);

        std::vector<size_t> gather_indices_value;
        {
            size_t shift = 0;
            for (const auto& indices : gather_indices_vecs) {
                for (size_t i = 0; i < indices.size(); ++i) {
                    gather_indices_value.push_back(indices[i] + shift);
                }
                shift += indices.size();
            }
        }

        auto gather_axis = std::make_shared<Constant>(ov::element::i64, ov::Shape{}, 1);
        auto gather_indices =
            std::make_shared<Constant>(ov::element::i64, ov::Shape{gather_indices_value.size()}, gather_indices_value);
        auto gather = std::make_shared<Gather>(reshape_input, gather_indices, gather_axis);

        auto split_axis_new = std::make_shared<Constant>(ov::element::i64, ov::Shape{}, 1);
        auto split_new = std::make_shared<Split>(gather, split_axis_new, split_node->get_num_splits());

        for (size_t output_idx = 0; output_idx < split_node->get_output_size(); ++output_idx) {
            for (auto& input : split_node->get_output_target_inputs(output_idx)) {
                auto transpose = ov::as_type_ptr<Transpose>(input.get_node()->shared_from_this());
                if (transpose && !is_transpose_supported(transpose)) {
                    EMUTEX_DEBUG_CHECKPOINT;
                    auto reshape_output_const_new =
                        std::make_shared<Constant>(ov::element::i64,
                                                   ov::Shape{transpose->get_output_shape(0).size()},
                                                   transpose->get_output_shape(0));
                    auto reshape_output_new =
                        std::make_shared<Reshape>(split_new->output(output_idx), reshape_output_const_new, false);
                    ov::replace_node_update_name(transpose, reshape_output_new);
                } else {
                    EMUTEX_DEBUG_CHECKPOINT;
                    for (auto consumer : split_node->output(output_idx).get_target_inputs()) {
                        consumer.replace_source_output(split_new->output(output_idx));
                    }
                }
            }
        }
        return true;
    };

    auto m = std::make_shared<Matcher>(split_node_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
