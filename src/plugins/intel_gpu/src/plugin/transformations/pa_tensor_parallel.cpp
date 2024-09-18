// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pa_tensor_parallel.hpp"
#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include <memory>
#include <vector>

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "intel_gpu/op/rank_constant.hpp"
#include "intel_gpu/op/sync_tensor.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/gather.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
namespace ov {
namespace intel_gpu {

std::shared_ptr<ov::Node> PATensorParallelFusion::find_first_fc_before_pa(std::shared_ptr<ov::Node> root_node) {
    auto get_output_node = [](const ov::Output<ov::Node>& output) -> std::shared_ptr<ov::Node> {
        return output.get_node_shared_ptr();
    };
    auto get_input_node = [&get_output_node](const ov::Input<ov::Node>& input) -> std::shared_ptr<ov::Node> {
        return get_output_node(input.get_source_output());
    };
    auto cur_node = get_input_node(root_node->inputs()[0]);
    if (ov::is_type<ov::intel_gpu::op::FullyConnectedCompressed>(cur_node) ||
        ov::is_type<ov::intel_gpu::op::FullyConnected>(cur_node)) {
        return cur_node;
    }
    return find_first_fc_before_pa(cur_node);
}

void PATensorParallelFusion::find_first_fcs_before_pa(std::shared_ptr<ov::Node> root_node) {
    has_visited_fc.clear();
    vector_visited_fc.clear();
    for (size_t i = 0; i < 3; i++) {
        auto first_fc_before_pa = find_first_fc_before_pa(root_node->get_input_node_shared_ptr(i));
        if (has_visited_fc.insert(first_fc_before_pa).second) {
            vector_visited_fc.push_back(first_fc_before_pa);
        }
    }
}

std::shared_ptr<ov::Node> PATensorParallelFusion::find_first_fc_after_pa(std::shared_ptr<ov::Node> root_node) {
    const auto& users = root_node->get_users();
    if (users.size() != 1)
        return nullptr;
    auto cur_node = users[0];

    if (ov::is_type<ov::op::v0::Result>(cur_node)) {
        return nullptr;
    }

    if (ov::is_type<ov::op::PagedAttentionExtension>(cur_node)) {
        return nullptr;
    }

    if (ov::is_type<ov::intel_gpu::op::FullyConnected>(cur_node)) {
        return cur_node;
    }
    if (ov::is_type<ov::intel_gpu::op::FullyConnectedCompressed>(cur_node)) {
        return cur_node;
    }
    return find_first_fc_after_pa(cur_node);
}

void PATensorParallelFusion::find_ops_in_fc_to_pa(std::shared_ptr<ov::Node> input) {
    if (ov::is_type<ov::intel_gpu::op::FullyConnectedCompressed>(input) ||
        ov::is_type<ov::intel_gpu::op::FullyConnected>(input) || ov::is_type<ov::op::internal::RoPE>(input)) {
        if (has_visited.insert(input).second) {
            vector_visited.push_back(input);
        }
    }
    auto users = input->get_users();
    size_t users_num = users.size();
    if (ov::is_type<ov::op::internal::RoPE>(input)) {
        users_num = 1;
    }
    for (size_t i = 0; i < users_num; ++i) {
        auto cur_node = users[i];
        if (ov::is_type<ov::op::v0::Result>(cur_node)) {
            continue;
        }
        if (ov::is_type<ov::op::v0::Constant>(cur_node)) {
            continue;
        }
        if (ov::is_type<ov::op::PagedAttentionExtension>(cur_node)) {
            return;
        }
        if (has_visited.insert(cur_node).second) {
            vector_visited.push_back(cur_node);
        }
        find_ops_in_fc_to_pa(cur_node);
    }
}

void PATensorParallelFusion::find_ops_in_pa_to_fc(std::shared_ptr<ov::Node> input) {
    auto users = input->get_users();
    size_t users_num = users.size();
    for (size_t i = 0; i < users_num; ++i) {
        auto cur_node = users[i];
        if (ov::is_type<ov::op::v0::Result>(cur_node)) {
            continue;
        }
        if (ov::is_type<ov::intel_gpu::op::FullyConnectedCompressed>(cur_node) ||
            ov::is_type<ov::intel_gpu::op::FullyConnected>(cur_node)) {
            if (has_visited.insert(cur_node).second) {
                vector_visited.push_back(cur_node);
            }
            return;
        }
        if (has_visited.insert(cur_node).second) {
            vector_visited.push_back(cur_node);
        }
        find_ops_in_pa_to_fc(cur_node);
    }
}

PATensorParallelFusion::PATensorParallelFusion(size_t world_size, size_t world_rank) {
    using namespace ov::pass::pattern;
    auto paged_attention = wrap_type<ov::op::PagedAttentionExtension>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        OPENVINO_ASSERT(pattern_map.count(paged_attention));
        std::shared_ptr<Node> m_pa = nullptr;
        m_pa = pattern_map.at(paged_attention).get_node_shared_ptr();

        auto split_fc = [&](std::shared_ptr<ov::Node>& org_fc,
                            op::TP_MODE tp_mode, const std::vector<int64_t> qkv_parts) -> std::pair<std::shared_ptr<ov::Node>, size_t> {
            const auto& m_data = org_fc->get_input_node_shared_ptr(0);
            const auto& weight_node = org_fc->get_input_node_shared_ptr(1);
            const auto& m_bias = org_fc->get_input_node_shared_ptr(2);
            std::shared_ptr<Node> splitted_fc = nullptr;
            auto weights_pshape = weight_node->get_output_partial_shape(0);
            auto reshaped_pshape = weights_pshape;
            int split_dim_range = 0;
            int split_axis = tp_mode == op::TP_MODE::ALL_GATHERH ? 0 : -1;
            if (weights_pshape.size() != 2) {
                auto reshape_to_2d = [](const ov::PartialShape& shape, const ov::Dimension& feature, size_t rank) {
                    auto static_shape = shape.to_shape();
                    size_t total = std::accumulate(static_shape.begin(), static_shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
                    auto dim = feature.is_static() ? feature.get_length() : static_cast<int64_t>(static_shape[rank - 1]);
                    return ov::PartialShape{ static_cast<int64_t>(total) / dim, dim }.to_shape();
                };
                auto input0_pshape = org_fc->get_input_node_shared_ptr(0)->get_output_partial_shape(0);
                auto feature = input0_pshape[input0_pshape.size() - 1ul];
                reshaped_pshape = reshape_to_2d(weights_pshape, feature, weights_pshape.size());
            }
            split_dim_range = reshaped_pshape.to_shape()[split_axis];
            {
                // transform to rank constant
                auto ranked_weight = std::make_shared<ov::intel_gpu::op::RankConstant>(weight_node, world_size, world_rank, tp_mode, qkv_parts);
                std::shared_ptr<ov::Node> ranked_bias, ranked_scale, ranked_zp;
                if (!std::dynamic_pointer_cast<op::Placeholder>(m_bias)) {
                    ranked_bias = std::make_shared<ov::intel_gpu::op::RankConstant>(m_bias, world_size, world_rank, tp_mode, qkv_parts);
                }
                auto compressed_fc = std::dynamic_pointer_cast<op::FullyConnectedCompressed>(org_fc);
                if (compressed_fc) {
                    auto scale_node = compressed_fc->get_input_node_shared_ptr(3);
                    if (tp_mode == op::TP_MODE::ALL_REDUCE) {
                        ranked_scale = scale_node;
                        if (scale_node->get_shape()[1] > 1)
                            ranked_scale =
                                std::make_shared<ov::intel_gpu::op::RankConstant>(scale_node, world_size, world_rank, tp_mode, qkv_parts);
                    } else {
                        ranked_scale = std::make_shared<ov::intel_gpu::op::RankConstant>(scale_node, world_size, world_rank, tp_mode, qkv_parts);
                    }
                    if (compressed_fc->inputs().size() > 4) {
                        auto zp_node = compressed_fc->get_input_node_shared_ptr(4);
                        // scalar zp
                        auto zp_shape = zp_node->get_output_shape(0);
                        bool is_scalar = (ov::shape_size(zp_node->get_output_shape(0)) == 1);
                        if (!is_scalar) {
                            if (tp_mode == op::TP_MODE::ALL_REDUCE) {
                                if (zp_node->get_shape()[1] > 1)
                                    ranked_zp =
                                        std::make_shared<ov::intel_gpu::op::RankConstant>(zp_node, world_size, world_rank, tp_mode, qkv_parts);
                            } else {
                                ranked_zp = std::make_shared<ov::intel_gpu::op::RankConstant>(zp_node, world_size, world_rank, tp_mode, qkv_parts);
                            }
                        }
                    }
                }
                auto output_type = m_pa->get_output_element_type(0);
                if (compressed_fc) {
                    if (compressed_fc->inputs().size() > 4)
                        splitted_fc = std::make_shared<op::FullyConnectedCompressed>(
                            m_data,
                            ranked_weight,
                            ranked_bias ? ranked_bias : m_bias,
                            ranked_scale,
                            ranked_zp ? ranked_zp : compressed_fc->get_input_node_shared_ptr(4),
                            output_type);
                    else
                        splitted_fc = std::make_shared<op::FullyConnectedCompressed>(
                            m_data,
                            ranked_weight,
                            ranked_bias ? ranked_bias : m_bias,
                            ranked_scale,
                            output_type);

                } else {
                    splitted_fc = std::make_shared<op::FullyConnected>(m_data,
                                                                ranked_weight,
                                                                ranked_bias ? ranked_bias : m_bias,
                                                                output_type);
                }
                org_fc->get_rt_info().insert({"splitted", true});
                return {splitted_fc, split_dim_range};
            }
        };
        auto new_variadicsplit_node =
            [](std::shared_ptr<ov::Node>& split_node,
                std::vector<int64_t>& split_parts) {
            auto split_name = split_node->get_friendly_name() + "_tp";
            auto axis = ov::op::v0::Constant::create(ov::element::i64,
                                                     ov::Shape{1},
                                                     {split_node->get_input_partial_shape(0).size() - 1});

            auto split_axis_value =
                split_node->get_input_partial_shape(0)[split_node->get_input_partial_shape(0).size() - 1].get_length();
            auto split_values = split_parts;
            int32_t split_unit_size = split_axis_value / std::accumulate(split_parts.begin(), split_parts.end(), 0);
            std::for_each(split_values.begin(), split_values.end(), [&](int64_t& d) {
                d *= split_unit_size;
            });

            auto split_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, split_values);
            auto new_split = std::make_shared<ov::op::v1::VariadicSplit>(split_node->get_input_node_shared_ptr(0),
                                                                         axis,
                                                                         split_const);
            new_split->set_friendly_name(split_name);
            copy_runtime_info(split_node, new_split);
            replace_node(split_node, new_split);
        };
        auto new_reshape_node =
            [](std::shared_ptr<ov::Node>& reshape_node, const int& half_head_num, const int& head_size) {
            std::vector<int32_t> new_shape = {};
            auto input_shape = reshape_node->get_input_partial_shape(0);
            int64_t sum_size = 1;
            for (size_t i = 0; i < input_shape.size(); i++) {
                if (input_shape[i].is_dynamic())
                    continue;
                sum_size = sum_size * input_shape[i].get_length();
            }
            // int64_t sum_size = half_head_num*head_size;
            auto out_put_shape = reshape_node->get_output_partial_shape(0);
            for (size_t i = 0; i < out_put_shape.size(); i++) {
                if (out_put_shape[i].is_dynamic()) {
                    new_shape.push_back(-1);
                    continue;
                }
                new_shape.push_back(out_put_shape[i].get_length());
            }
            for (size_t i = 0; i < out_put_shape.size(); i++) {
                if (!(out_put_shape[i].is_dynamic()) && !(out_put_shape[i].compatible(1)) &&
                    !(out_put_shape[i].compatible(3)) && !(out_put_shape[i].compatible(128))) {
                    if (out_put_shape.size() == 2)
                        new_shape[i] = sum_size;
                    else
                        new_shape[i] = sum_size / head_size;
                    break;
                }
            }
            if ((out_put_shape.size() == 4) && (out_put_shape[2].compatible(3)))
                new_shape[3] = sum_size/3;
            if (out_put_shape.size() == 3)
                new_shape[2] = sum_size;
            auto shape0 =
                std::make_shared<ov::op::v0::Constant>(ov::element::i32,
                                                       ov::Shape{reshape_node->get_output_partial_shape(0).size()},
                                                       new_shape);
            auto new_reshape =
                std::make_shared<ov::op::v1::Reshape>(reshape_node->get_input_source_output(0), shape0, true);
            new_reshape->set_friendly_name(reshape_node->get_friendly_name() + "_tp");
            copy_runtime_info(reshape_node, new_reshape);
            replace_node(reshape_node, new_reshape);
        };
        auto new_add_node = [&](std::shared_ptr<ov::Node>& add_node, const std::vector<int64_t> qkv_parts) {
            auto rank_constant =
                std::make_shared<ov::intel_gpu::op::RankConstant>(add_node->get_input_node_shared_ptr(1),
                                                                  world_size,
                                                                  world_rank,
                                                                  op::TP_MODE::ALL_REDUCE, qkv_parts);
            auto new_add = std::make_shared<ov::op::v1::Add>(add_node->get_input_source_output(0), rank_constant);
            new_add->set_friendly_name(add_node->get_friendly_name() + "_tp");
            copy_runtime_info(add_node, new_add);
            replace_node(add_node, new_add);
        };
        auto pa_sync_concat = [&]() {
            std::shared_ptr<ov::intel_gpu::op::SyncTensor> sync_node =
                std::make_shared<ov::intel_gpu::op::SyncTensor>(m_pa->output(0),
                                                                world_size,
                                                                world_rank,
                                                                m_pa->get_output_partial_shape(0)[-1].get_length(),
                                                                m_pa->get_output_element_type(0),
                                                                ov::intel_gpu::op::TP_MODE::ALL_GATHERH);
            sync_node->set_friendly_name(m_pa->get_friendly_name() + "_TP_pa");
            if (sync_node->get_gpu_p2p_enabled()) {
                copy_runtime_info(m_pa, sync_node);
                m_pa->get_users()[0]->input(0).replace_source_output(sync_node->output(0));
            } else {
                auto concat_node = std::make_shared<ov::op::v0::Concat>(sync_node->outputs(), -1);
                concat_node->set_friendly_name(m_pa->get_friendly_name() + "_ALLGATHER");
                copy_runtime_info(m_pa, concat_node);
                m_pa->get_users()[0]->input(0).replace_source_output(concat_node->output(0));
            }
        };
        auto fc_after_pa_sync = [&](std::shared_ptr<ov::Node>& fc_node, const std::vector<int64_t> qkv_parts) {
            std::map<int, std::shared_ptr<ov::Node>> org_users;
            for (auto u : fc_node->get_users()) {
                for (size_t idx = 0; idx < u->inputs().size(); ++idx) {
                    if (u->get_input_node_shared_ptr(idx) == fc_node) {
                        org_users.insert({idx, u});
                    }
                }
            }
            auto new_fc = split_fc(fc_node, op::TP_MODE::ALL_REDUCE, qkv_parts).first;
            new_fc->get_rt_info().insert({"splitted", true});
            std::shared_ptr<ov::intel_gpu::op::SyncTensor> sync_node;
            sync_node =
                std::make_shared<ov::intel_gpu::op::SyncTensor>(new_fc,
                                                                world_size,
                                                                world_rank,
                                                                fc_node->get_input_node_shared_ptr(1)->get_shape()[-1],
                                                                fc_node->get_element_type(),
                                                                ov::intel_gpu::op::TP_MODE::ALL_REDUCE);
            sync_node->set_friendly_name(fc_node->get_friendly_name() + "_TP");
            copy_runtime_info(fc_node, new_fc);
            for (auto& iter : org_users) {
                iter.second->input(iter.first).replace_source_output(sync_node->output(0));
            }
            fc_node->clear_control_dependencies();
        };

        if (m_pa) {
            int half_head_num = m_pa->get_input_node_shared_ptr(3)->get_output_partial_shape(0)[1].get_length();
            int head_size = m_pa->get_input_node_shared_ptr(3)->get_output_partial_shape(0)[2].get_length();
            int q_size = m_pa->get_input_node_shared_ptr(0)->get_output_partial_shape(0)[1].get_length();
            int k_size = m_pa->get_input_node_shared_ptr(1)->get_output_partial_shape(0)[1].get_length();
            int v_size = m_pa->get_input_node_shared_ptr(2)->get_output_partial_shape(0)[1].get_length();
            std::vector<int64_t> qkv_parts = {q_size, k_size, v_size};
            int32_t qkv_min_part = *min_element(qkv_parts.begin(), qkv_parts.end());
            std::for_each(qkv_parts.begin(), qkv_parts.end(), [&](int64_t& d) { d/=qkv_min_part;});
            find_first_fcs_before_pa(m_pa);
            for (size_t j = 0; j < vector_visited_fc.size(); j++) {
                has_visited.clear();
                vector_visited.clear();
                find_ops_in_fc_to_pa(vector_visited_fc[j]);
                for (size_t i = 0; i < vector_visited.size(); i++) {
                    auto cur_node = vector_visited[i];
                    if (ov::is_type<ov::intel_gpu::op::FullyConnected>(cur_node) ||
                        ov::is_type<ov::intel_gpu::op::FullyConnectedCompressed>(cur_node)) {
                        std::shared_ptr<ov::Node> new_fc = nullptr;
                        if (vector_visited_fc.size() > 1)
                            new_fc = split_fc(cur_node, op::TP_MODE::ALL_GATHERH, qkv_parts).first;
                        else
                            new_fc = split_fc(cur_node, op::TP_MODE::ALL_GATHERQKV, qkv_parts).first;
                        new_fc->set_friendly_name(cur_node->get_friendly_name());
                        new_fc->get_rt_info().insert({"splitted", true});
                        copy_runtime_info(cur_node, new_fc);
                        replace_node(cur_node, new_fc);
                        continue;
                    }
                    if (ov::is_type<ov::op::v1::VariadicSplit>(cur_node)) {
                        new_variadicsplit_node(cur_node, qkv_parts);
                        continue;
                    }
                    if (ov::is_type<ov::op::v1::Reshape>(cur_node)) {
                        new_reshape_node(cur_node, half_head_num, head_size);
                        continue;
                    }
                    if (ov::is_type<ov::op::v1::Add>(cur_node)) {
                        new_add_node(cur_node, qkv_parts);
                        continue;
                    }
                    if (ov::is_type<ov::op::internal::RoPE>(cur_node)) {
                        auto old_shape = cur_node->get_output_partial_shape(0);
                        cur_node->validate_and_infer_types();
                    }
                    if (ov::is_type<ov::op::v1::Transpose>(cur_node)) {
                        auto old_shape = cur_node->get_output_partial_shape(0);
                        cur_node->validate_and_infer_types();
                    }
                    if (ov::is_type<ov::op::v8::Gather>(cur_node)) {
                        auto old_shape = cur_node->get_output_partial_shape(0);
                        cur_node->validate_and_infer_types();
                    }
                }
            }
            m_pa->validate_and_infer_types();
            std::shared_ptr<ov::Node> first_fc_after_pa = find_first_fc_after_pa(m_pa);
            if (first_fc_after_pa) {
                auto compressed_fc = std::dynamic_pointer_cast<op::FullyConnectedCompressed>(first_fc_after_pa);

                if (compressed_fc && (compressed_fc->get_input_node_shared_ptr(3)->get_shape()[1] % 2 != 0)) {
                    auto scale_node_dims = compressed_fc->get_input_node_shared_ptr(3)->get_shape()[1];
                    if ((scale_node_dims != 1 && scale_node_dims % 2 != 0)) {
                        std::cout << "skip shape: " << compressed_fc->get_input_node_shared_ptr(3)->get_shape() << std::endl;
                        pa_sync_concat();
                        return true;
                    }
                }
                has_visited.clear();
                vector_visited.clear();
                find_ops_in_pa_to_fc(m_pa);
                for (size_t i = 0; i < vector_visited.size(); i++) {
                    auto cur_node = vector_visited[i];
                    if (ov::is_type<ov::intel_gpu::op::FullyConnected>(cur_node) ||
                        ov::is_type<ov::intel_gpu::op::FullyConnectedCompressed>(cur_node)) {
                        fc_after_pa_sync(cur_node, qkv_parts);
                        continue;
                    }

                    if (ov::is_type<ov::op::v1::Reshape>(cur_node)) {
                        new_reshape_node(cur_node, half_head_num, head_size);
                        continue;
                    }
                }
            } else {
                pa_sync_concat();
            }
        }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(paged_attention, "PATensorParallelFusion");
    this->register_matcher(m, callback);
}
}  // namespace intel_gpu
}  // namespace ov