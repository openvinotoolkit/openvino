// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/matmul_horizontal_fusing.hpp"
#include "transformations/utils/utils.hpp"

#include "itt.hpp"
#include "transformations/init_node_info.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset8.hpp>

bool is_matmul_with_weights(const std::shared_ptr<ngraph::Node>& node) {
    auto weights = node->get_input_node_shared_ptr(1);
    if (ngraph::is_type<ngraph::opset8::Constant>(weights)) {
        return true;
    }

    if (ngraph::is_type<ngraph::opset8::Multiply>(weights) &&
        ngraph::is_type<ngraph::opset8::Constant>(weights->get_input_node_shared_ptr(1))) {
        weights = weights->get_input_node_shared_ptr(0);
    }

    if (ngraph::is_type<ngraph::opset8::Subtract>(weights)) {
        auto constant_path = weights->get_input_node_shared_ptr(1);
        if ((ngraph::is_type<ngraph::opset8::Constant>(constant_path)) ||
            (ngraph::is_type<ngraph::opset8::Convert>(constant_path) &&
                ngraph::is_type<ngraph::opset8::Constant>(constant_path->get_input_node_shared_ptr(0)))) {
            weights = weights->get_input_node_shared_ptr(0);
        }
    }

    if (ngraph::is_type<ngraph::opset8::Convert>(weights)) {
        weights = weights->get_input_node_shared_ptr(0);
        return ngraph::is_type<ngraph::opset8::Constant>(weights);
    }

    return false;
}

std::shared_ptr<ngraph::Node> broadcast_if_necessary(
    const std::shared_ptr<ngraph::Node>& constant,
    const ngraph::PartialShape& data_shape,
    const size_t bcast_axis) {
    auto const_shape = constant->get_output_shape(0);
    if (shape_size(const_shape) > 1) {
        return constant;
    }

    const_shape = ngraph::Shape(data_shape.rank().get_length(), 1);
    const_shape[bcast_axis] = data_shape[bcast_axis].get_length();
    auto reshape_constant = ngraph::opset8::Constant::create(ngraph::element::i64, { const_shape.size() }, const_shape);
    return ngraph::op::util::make_try_fold<ngraph::opset8::Broadcast>(constant, reshape_constant);
}

std::shared_ptr<ngraph::Node> fuse_weights_path(const ngraph::NodeVector& matmuls, const bool transpose_weights) {
    const auto matmul = matmuls[0];
    const auto weights_shape = matmul->get_input_shape(1);
    const size_t concat_axis = transpose_weights ? weights_shape.size() - 2 : weights_shape.size() - 1;

    ngraph::NodeVector mul_constants;
    ngraph::NodeVector sub_constants;
    std::shared_ptr<ngraph::Node> convert;
    ngraph::NodeVector weights;

    for (const auto& elem : matmuls) {
        auto weights_path = elem->get_input_node_shared_ptr(1);
        if (ngraph::is_type<ngraph::opset8::Multiply>(weights_path)) {
            mul_constants.emplace_back(broadcast_if_necessary(weights_path->get_input_node_shared_ptr(1), weights_shape, concat_axis));
            weights_path = weights_path->get_input_node_shared_ptr(0);
        }
        if (ngraph::is_type<ngraph::opset8::Subtract>(weights_path)) {
            sub_constants.emplace_back(broadcast_if_necessary(weights_path->get_input_node_shared_ptr(1), weights_shape, concat_axis));
            weights_path = weights_path->get_input_node_shared_ptr(0);
        }
        if (ngraph::is_type<ngraph::opset8::Convert>(weights_path)) {
            convert = weights_path;
            weights_path = weights_path->get_input_node_shared_ptr(0);
        }
        if (ngraph::is_type<ngraph::opset8::Constant>(weights_path)) {
            weights.emplace_back(weights_path);
        }
    }

    std::shared_ptr<ngraph::Node> new_weights = ngraph::op::util::make_try_fold<ngraph::opset8::Concat>(weights, concat_axis);
    ngraph::copy_runtime_info(weights, new_weights);

    if (convert) {
        new_weights = convert->clone_with_new_inputs({ new_weights });
        ngraph::copy_runtime_info(convert, new_weights);
    }

    if (!sub_constants.empty()) {
        const auto new_sub_const = ngraph::op::util::make_try_fold<ngraph::opset8::Concat>(sub_constants, concat_axis);
        new_weights = std::make_shared<ngraph::opset8::Subtract>(new_weights, new_sub_const);
    }

    if (!mul_constants.empty()) {
        const auto new_mul_const = ngraph::op::util::make_try_fold<ngraph::opset8::Concat>(mul_constants, concat_axis);
        new_weights = std::make_shared<ngraph::opset8::Multiply>(new_weights, new_mul_const);
    }

    return new_weights;
}

template <typename T>
std::shared_ptr<T> fuse_elwise(const std::shared_ptr<ngraph::Node>& last_fused_node, ngraph::NodeVector& last_original_nodes) {
    ngraph::Shape constant_shape;
    ngraph::NodeVector constants_to_fuse;
    ngraph::NodeVector eltwises_to_fuse;
    const auto input_rank = last_fused_node->get_input_partial_shape(0).rank();

    auto validate_constant = [&input_rank](const std::shared_ptr<ngraph::Node>& constant) {
        auto bias_shape = constant->get_output_shape(0);
        auto bias_rank = bias_shape.size();
        size_t rank = input_rank.get_length();

        if ((bias_rank != 1 && bias_rank != rank) ||
            (std::count_if(bias_shape.begin(), bias_shape.end(), [](const size_t elem) { return elem > 1; }) != 1)) {
            return false;
        }

        // We can fuse only per-channel biases after matmul with constant (channel is the last dimension)
        if (bias_rank > 1 && bias_shape[bias_rank - 1] == 1) {
            return false;
        }

        return true;
    };

    for (const auto& last_node : last_original_nodes) {
        const auto target_inputs = last_node->output(0).target_inputs();
        if (target_inputs.size() > 1) {
            return nullptr;
        }

        const auto eltwise = target_inputs[0];
        if (!ngraph::is_type<T>(eltwise)) {
            return nullptr;
        }

        const auto constant = eltwise->get_input_node_shared_ptr(1);
        if (!ngraph::is_type<ngraph::opset8::Constant>(constant)) {
            return nullptr;
        }

        const auto eltwise_consumers = eltwise->output(0).target_inputs();
        if (std::any_of(eltwise_consumers.begin(), eltwise_consumers.end(),
            [](const std::shared_ptr<ngraph::Node>& elem) { return ngraph::is_type<ngraph::opset8::Result>(elem); })) {
            return nullptr;
        }

        if (constant_shape.empty() && validate_constant(constant)) {
            constant_shape = eltwise->get_input_shape(1);
        } else if (eltwise->get_input_shape(1) != constant_shape) {
            return nullptr;
        }

        eltwises_to_fuse.emplace_back(eltwise);
        constants_to_fuse.emplace_back(constant);
    }

    auto concat_idx = 0;
    if (constants_to_fuse[0]->get_output_shape(0).size() > 1) {
        concat_idx = last_fused_node->get_input_partial_shape(0).rank().get_length() - 1;
    }

    const auto new_constant = ngraph::op::util::make_try_fold<ngraph::opset8::Concat>(constants_to_fuse, concat_idx);
    const auto new_eltwise = std::make_shared<T>(last_fused_node->output(0), new_constant);
    new_eltwise->set_friendly_name(eltwises_to_fuse[0]->get_friendly_name() + "/Fused");
    ngraph::copy_runtime_info(eltwises_to_fuse, new_eltwise);

    for (size_t i = 0; i < last_original_nodes.size(); ++i) {
        last_original_nodes[i] = eltwises_to_fuse[i];
    }

    return new_eltwise;
}

std::shared_ptr<ngraph::Node> fuse_fake_quantizes(const std::shared_ptr<ngraph::Node>& last_fused_node, ngraph::NodeVector& last_original_nodes) {
    std::shared_ptr<ngraph::Node> fake_quantize;
    ngraph::NodeVector fq_to_fuse;
    ngraph::NodeVector il_to_fuse;
    ngraph::NodeVector ih_to_fuse;
    ngraph::NodeVector ol_to_fuse;
    ngraph::NodeVector oh_to_fuse;

    for (auto& last_node : last_original_nodes) {
        const auto target_inputs = last_node->output(0).target_inputs();
        if (target_inputs.size() > 1) {
            return nullptr;
        }

        const auto neighbour_fq = target_inputs[0];
        if (!ngraph::is_type<ngraph::opset8::FakeQuantize>(neighbour_fq) ||
            !ngraph::is_type<ngraph::opset8::Constant>(neighbour_fq->get_input_node_shared_ptr(1)) ||
            !ngraph::is_type<ngraph::opset8::Constant>(neighbour_fq->get_input_node_shared_ptr(2)) ||
            !ngraph::is_type<ngraph::opset8::Constant>(neighbour_fq->get_input_node_shared_ptr(3)) ||
            !ngraph::is_type<ngraph::opset8::Constant>(neighbour_fq->get_input_node_shared_ptr(4))) {
            return nullptr;
        }

        const auto fq_consumers = neighbour_fq->output(0).target_inputs();
        if (std::any_of(fq_consumers.begin(), fq_consumers.end(),
            [](const std::shared_ptr<ngraph::Node>& elem) { return ngraph::is_type<ngraph::opset8::Result>(elem); })) {
            return nullptr;
        }

        if (fake_quantize == nullptr) {
            fake_quantize = neighbour_fq;
        } else if (fake_quantize->get_input_shape(1) != neighbour_fq->get_input_shape(1) ||
            fake_quantize->get_input_shape(2) != neighbour_fq->get_input_shape(2) ||
            fake_quantize->get_input_shape(3) != neighbour_fq->get_input_shape(3) ||
            fake_quantize->get_input_shape(4) != neighbour_fq->get_input_shape(4)) {
            return nullptr;
        }

        fq_to_fuse.emplace_back(neighbour_fq);
        il_to_fuse.emplace_back(neighbour_fq->get_input_node_shared_ptr(1));
        ih_to_fuse.emplace_back(neighbour_fq->get_input_node_shared_ptr(2));
        ol_to_fuse.emplace_back(neighbour_fq->get_input_node_shared_ptr(3));
        oh_to_fuse.emplace_back(neighbour_fq->get_input_node_shared_ptr(4));
    }

    const auto input_rank = last_fused_node->get_input_partial_shape(0).rank().get_length();
    auto channels_idx = input_rank - 1;

    auto get_fused_constant = [&](ngraph::NodeVector& constants) {
        const auto constant = ngraph::as_type_ptr<ngraph::opset8::Constant>(constants[0]);
        if (ngraph::shape_size(constant->get_shape()) == 1) {
            bool constants_are_identical = true;
            const float value = ngraph::as_type_ptr<ngraph::opset8::Constant>(constants[0])->cast_vector<float>()[0];
            for (size_t i = 1; i < constants.size(); ++i) {
                if (ngraph::shape_size(constant->get_output_shape(0)) != 1 ||
                    ngraph::as_type_ptr<ngraph::opset8::Constant>(constants[i])->cast_vector<float>()[0] != value) {
                    constants_are_identical = false;
                    break;
                }
            }

            if (constants_are_identical) {
                std::shared_ptr<ngraph::Node> result = ngraph::opset8::Constant::create(
                    constant->get_element_type(), constant->get_shape(), { value });
                return result;
            }
        }

        for (size_t i = 0; i < constants.size(); ++i) {
            constants[i] = broadcast_if_necessary(constants[i], fq_to_fuse[i]->get_input_partial_shape(0), channels_idx);
        }

        return ngraph::op::util::make_try_fold<ngraph::opset8::Concat>(constants, channels_idx);
    };

    const auto new_il = get_fused_constant(il_to_fuse);
    const auto new_ih = get_fused_constant(ih_to_fuse);
    const auto new_ol = get_fused_constant(ol_to_fuse);
    const auto new_oh = get_fused_constant(oh_to_fuse);
    const auto new_fake_quantize = fake_quantize->clone_with_new_inputs({ last_fused_node, new_il, new_ih, new_ol, new_oh });
    new_fake_quantize->set_friendly_name(fake_quantize->get_friendly_name() + "/Fused");
    ngraph::copy_runtime_info(fq_to_fuse, new_fake_quantize);

    for (size_t i = 0; i < last_original_nodes.size(); ++i) {
        last_original_nodes[i] = fq_to_fuse[i];
    }

    return new_fake_quantize;
}

std::shared_ptr<ngraph::Node> fuse_reshapes(const std::shared_ptr<ngraph::Node>& last_fused_node, ngraph::NodeVector& last_original_nodes) {
    std::vector<std::int64_t> reshape_pattern;
    ngraph::NodeVector reshapes_to_fuse;

    auto validate_reshape = [](const std::shared_ptr<ngraph::Node>& reshape) {
        auto input_pshape = reshape->get_input_partial_shape(0);
        auto input_rank = input_pshape.rank();
        auto output_pshape = reshape->get_output_partial_shape(0);
        auto output_rank = output_pshape.rank();
        if (output_rank.is_dynamic() || input_rank.is_dynamic() || output_pshape.rank().get_length() != 4) {
            return false;
        }

        auto last_input_dimension = input_pshape[input_rank.get_length() - 1];
        auto out_h = output_pshape[2];
        auto out_w = output_pshape[3];

        if (last_input_dimension.is_static() && out_h.is_static() && out_w.is_static()) {
            return last_input_dimension == out_h * out_w;
        }

        return true;
    };

    for (const auto& last_node : last_original_nodes) {
        const auto target_inputs = last_node->output(0).target_inputs();
        if (target_inputs.size() > 1) {
            return nullptr;
        }

        const auto neighbour_reshape = target_inputs[0];
        if (!ngraph::is_type<ngraph::opset8::Reshape>(neighbour_reshape)) {
            return nullptr;
        }

        const auto reshape_pattern_const = ngraph::as_type_ptr<ngraph::opset8::Constant>(neighbour_reshape->get_input_node_shared_ptr(1));
        if (!reshape_pattern_const) {
            return nullptr;
        }

        const auto reshape_consumers = neighbour_reshape->output(0).target_inputs();
        if (std::any_of(reshape_consumers.begin(), reshape_consumers.end(),
            [](const std::shared_ptr<ngraph::Node>& elem) { return ngraph::is_type<ngraph::opset8::Result>(elem); })) {
            return nullptr;
        }

        const auto neighbour_reshape_pattern = reshape_pattern_const->cast_vector<std::int64_t>();
        if (reshape_pattern.empty() && validate_reshape(neighbour_reshape)) {
            reshape_pattern = neighbour_reshape_pattern;
        } else if (reshape_pattern != neighbour_reshape_pattern) {
            return nullptr;
        }
        reshapes_to_fuse.emplace_back(neighbour_reshape);
    }

    const size_t num_fused_nodes = last_original_nodes.size();
    auto new_reshape_pattern = reshape_pattern;
    new_reshape_pattern[new_reshape_pattern.size() - 2] *= num_fused_nodes;

    const auto new_reshape_const = ngraph::opset8::Constant::create(
        ngraph::element::i64, ngraph::Shape{ new_reshape_pattern.size() }, new_reshape_pattern);
    const auto new_reshape = reshapes_to_fuse[0]->clone_with_new_inputs({ last_fused_node, new_reshape_const });
    new_reshape->set_friendly_name(reshapes_to_fuse[0]->get_friendly_name() + "/Fused");
    ngraph::copy_runtime_info(reshapes_to_fuse, new_reshape);

    for (size_t i = 0; i < last_original_nodes.size(); ++i) {
        last_original_nodes[i] = reshapes_to_fuse[i];
    }

    return new_reshape;
}

std::shared_ptr<ngraph::Node> fuse_transposes(
    const std::shared_ptr<ngraph::Node>& last_fused_node,
    ngraph::NodeVector& last_original_nodes,
    size_t& split_axis) {
    std::vector<std::int64_t> transpose_values;
    ngraph::NodeVector transposes_to_fuse;

    for (const auto& last_node : last_original_nodes) {
        const auto target_inputs = last_node->output(0).target_inputs();
        if (target_inputs.size() > 1) {
            return nullptr;
        }

        const auto neighbour_transpose = target_inputs[0];
        if (!ngraph::is_type<ngraph::opset8::Transpose>(neighbour_transpose)) {
            return nullptr;
        }

        const auto transpose_const = ngraph::as_type_ptr<ngraph::opset8::Constant>(neighbour_transpose->get_input_node_shared_ptr(1));
        if (!transpose_const) {
            return nullptr;
        }

        const auto reshape_consumers = neighbour_transpose->output(0).target_inputs();
        if (std::any_of(reshape_consumers.begin(), reshape_consumers.end(),
            [](const std::shared_ptr<ngraph::Node>& elem) { return ngraph::is_type<ngraph::opset8::Result>(elem); })) {
            return nullptr;
        }


        const auto neighbour_transpose_values = transpose_const->cast_vector<std::int64_t>();
        if (transpose_values.empty() && neighbour_transpose_values.size() > 2) {
            transpose_values = neighbour_transpose_values;
        } else if (transpose_values != neighbour_transpose_values) {
            return nullptr;
        }
        transposes_to_fuse.emplace_back(neighbour_transpose);
    }

    const auto new_transpose_const = ngraph::opset8::Constant::create(
        ngraph::element::i64, ngraph::Shape{ transpose_values.size() }, transpose_values);
    const auto new_transpose = transposes_to_fuse[0]->clone_with_new_inputs({ last_fused_node, new_transpose_const });
    new_transpose->set_friendly_name(transposes_to_fuse[0]->get_friendly_name() + "/Fused");
    ngraph::copy_runtime_info(transposes_to_fuse, new_transpose);

    for (size_t i = 0; i < last_original_nodes.size(); ++i) {
        last_original_nodes[i] = transposes_to_fuse[i];
    }

    for (size_t i = 0; i < transpose_values.size(); ++i) {
        if (transpose_values[i] == 2) {
            split_axis = i;
            break;
        }
    }

    return new_transpose;
}

bool fuse_matmuls(const ngraph::NodeVector& matmuls) {
    const size_t matmuls_num = matmuls.size();
    if (matmuls_num < 2ul) {
        return false;
    }

    const auto matmul = ngraph::as_type_ptr<ngraph::opset8::MatMul>(matmuls[0]);
    const auto new_weights = fuse_weights_path(matmuls, matmul->get_transpose_b());
    const auto new_matmul = matmul->clone_with_new_inputs({ matmul->input_value(0), new_weights });
    new_matmul->set_friendly_name(matmul->get_friendly_name() + "/Fused");
    ngraph::copy_runtime_info(matmuls, new_matmul);

    std::shared_ptr<ngraph::Node> last_fused_node = new_matmul;
    ngraph::NodeVector last_original_nodes = matmuls;

    // optional part: fuse biases and FQ if possible
    const auto fused_subs = fuse_elwise<ngraph::opset8::Subtract>(last_fused_node, last_original_nodes);
    if (fused_subs != nullptr) {
        last_fused_node = fused_subs;
    }

    const auto fused_muls = fuse_elwise<ngraph::opset8::Multiply>(last_fused_node, last_original_nodes);
    if (fused_muls != nullptr) {
        last_fused_node = fused_muls;
    }

    const auto fused_biases = fuse_elwise<ngraph::opset8::Add>(last_fused_node, last_original_nodes);
    if (fused_biases != nullptr) {
        last_fused_node = fused_biases;
    }

    const auto fused_fake_quantize = fuse_fake_quantizes(last_fused_node, last_original_nodes);
    if (fused_fake_quantize != nullptr) {
        last_fused_node = fused_fake_quantize;
    }

    const auto split_input_rank = last_fused_node->get_output_partial_shape(0).rank().get_length();
    size_t split_axis_value = split_input_rank - 1;

    // Fuse reshapes which separate last dimension in two dimensions (output is 4D tensor)
    const auto fused_reshape = fuse_reshapes(last_fused_node, last_original_nodes);
    if (fused_reshape != nullptr) {
        last_fused_node = fused_reshape;
        split_axis_value = 2;
    }

    // Fuse [0, 2, 1, 3] transposes
    const auto fused_transpose = fuse_transposes(last_fused_node, last_original_nodes, split_axis_value);
    if (fused_transpose != nullptr) {
        last_fused_node = fused_transpose;
    }

    const auto split_axis = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { split_axis_value });
    const auto split = std::make_shared<ngraph::opset8::Split>(last_fused_node, split_axis, matmuls_num);
    split->set_friendly_name(new_matmul->get_friendly_name() + "/Split");
    ngraph::copy_runtime_info(last_fused_node, split);

    for (size_t i = 0; i < matmuls_num; ++i) {
        auto output = last_original_nodes[i]->output(0);
        output.replace(split->output(i));
    }

    return true;
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::MatMulHorizontalFusion, "MatMulHorizontalFusion", 0);

bool ngraph::pass::MatMulHorizontalFusion::run_on_function(std::shared_ptr<ngraph::Function> f) {
    RUN_ON_FUNCTION_SCOPE(MatMulHorizontalFusion);
    bool rewritten = false;
    for (const auto& node : f->get_ordered_ops()) {
        const auto outputs = node->outputs();
        if (std::all_of(outputs.begin(), outputs.end(), [](const ngraph::Output<Node>& elem) { return elem.target_inputs().size() == 1ul; })) {
            continue;
        }

        for (const auto& output : outputs) {
            const auto consumers = output.target_inputs();
            std::vector<std::shared_ptr<ngraph::opset8::MatMul>> matmuls;
            for (const auto& consumer : consumers) {
                const auto matmul = ngraph::as_type_ptr<ngraph::opset8::MatMul>(consumer);
                if (matmul && matmul->get_output_partial_shape(0).rank().is_static() && is_matmul_with_weights(matmul)) {
                    matmuls.emplace_back(matmul);
                }
            }

            if (matmuls.size() < 2) {
                continue;
            }

            auto get_weights = [](const std::shared_ptr<ngraph::Node>& matmul) {
                auto weighs_path = matmul->get_input_node_shared_ptr(1);
                if (ngraph::is_type<ngraph::opset8::Constant>(weighs_path)) {
                    return weighs_path;
                }

                if (ngraph::is_type<ngraph::opset8::Multiply>(weighs_path)) {
                    weighs_path = weighs_path->get_input_node_shared_ptr(0);
                }

                if (ngraph::is_type<ngraph::opset8::Subtract>(weighs_path)) {
                    weighs_path = weighs_path->get_input_node_shared_ptr(0);
                }

                if (ngraph::is_type<ngraph::opset8::Convert>(weighs_path)) {
                    weighs_path = weighs_path->get_input_node_shared_ptr(0);
                }

                return weighs_path;
            };

            const auto matmul = matmuls[0];
            const bool transpose_a = matmul->get_transpose_a();
            const bool transpose_b = matmul->get_transpose_b();
            const auto weights = get_weights(matmul);

            auto gold_and_cur_are_similar = [&](const std::shared_ptr<ngraph::opset8::MatMul>& target) {
                if (target->get_transpose_a() != transpose_a ||
                    target->get_transpose_b() != transpose_b) {
                    return false;
                }

                auto target_weights = get_weights(target);
                if (weights->get_output_element_type(0) != target_weights->get_output_element_type(0) ||
                    weights->get_output_shape(0) != target_weights->get_output_shape(0)) {
                    return false;
                }

                return true;
            };

            NodeVector matmuls_to_fuse;
            for (size_t i = 0; i < matmuls.size(); ++i) {
                const auto neighbor_matmul_consumers = matmuls[i]->output(0).target_inputs();
                if (std::any_of(neighbor_matmul_consumers.begin(), neighbor_matmul_consumers.end(),
                    [](const std::shared_ptr<ngraph::Node>& elem) { return ngraph::is_type<ngraph::opset8::Result>(elem); })) {
                    continue;
                }

                if (!gold_and_cur_are_similar(matmuls[i])) {
                    continue;
                }

                std::shared_ptr<ngraph::Node> last_node = matmuls[i];
                matmuls_to_fuse.emplace_back(matmuls[i]);
            }

            rewritten |= fuse_matmuls(matmuls_to_fuse);
        }
    }

    return rewritten;
}
