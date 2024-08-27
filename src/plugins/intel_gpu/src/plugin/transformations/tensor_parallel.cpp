// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tensor_parallel.hpp"
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
namespace ov {
namespace intel_gpu {

std::shared_ptr<ov::Node> TensorParallelFusion::fused_fc_before_pa(std::shared_ptr<ov::Node> root_node) {
    auto get_output_node = [](const ov::Output<ov::Node>& output) -> std::shared_ptr<ov::Node> {
        return output.get_node_shared_ptr();
    };
    auto get_input_node = [&get_output_node](const ov::Input<ov::Node>& input) -> std::shared_ptr<ov::Node> {
        return get_output_node(input.get_source_output());
    };
    auto cur_node = get_input_node(root_node->inputs()[0]);
    if (ov::is_type<ov::intel_gpu::op::FullyConnectedCompressed>(cur_node)) {
        // std::cout << "FullyConnectedCompressed: " << cur_node->get_name() << ", "
        //           << cur_node->get_input_partial_shape(0) << std::endl;
        return cur_node;
    }
    return fused_fc_before_pa(cur_node);
}

std::shared_ptr<ov::Node> TensorParallelFusion::find_first_fc_after_pa(std::shared_ptr<ov::Node> root_node) {
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

TensorParallelFusion::TensorParallelFusion(size_t world_size, size_t world_rank) {
    using namespace ov::pass::pattern;
    auto data = any_input();
    auto weights = any_input();
    auto bias = any_input();
    auto fully_connected = wrap_type<op::FullyConnected>({data, weights, bias});
    auto fully_connected_compressed = wrap_type<op::FullyConnectedCompressed>({data, weights, bias, any_input()});
    auto fully_connected_compressed_with_zp = wrap_type<op::FullyConnectedCompressed>({data, weights, bias, any_input(), any_input()});
    auto in0 = any_input();
    auto in1 = any_input();
    auto in2 = any_input();
    auto in3 = any_input();
    auto in4 = any_input();
    auto in5 = any_input();
    auto in6 = any_input();
    auto in7 = any_input();
    auto in8 = any_input();
    auto in9 = any_input();
    auto in10 = any_input();
    auto in11 = any_input();
    auto in12 = any_input();
    auto paged_attention = wrap_type<ov::op::PagedAttentionExtension>(
        {in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12});
    // auto paged_attention_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{fully_connected});
    auto fully_connected_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{fully_connected,
                                                                                    fully_connected_compressed,
                                                                                    fully_connected_compressed_with_zp, paged_attention});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        OPENVINO_ASSERT(pattern_map.count(fully_connected_compressed_with_zp) ||
                        pattern_map.count(fully_connected_compressed) || pattern_map.count(fully_connected) || pattern_map.count(paged_attention));
        std::shared_ptr<Node> m_fc = nullptr;
        std::shared_ptr<Node> m_pa = nullptr;
        if (pattern_map.find(fully_connected) != pattern_map.end()) {
            m_fc = pattern_map.at(fully_connected).get_node_shared_ptr();
        } else if (pattern_map.find(fully_connected_compressed) != pattern_map.end()) {
            m_fc = pattern_map.at(fully_connected_compressed).get_node_shared_ptr();
        } else if (pattern_map.find(fully_connected_compressed_with_zp) != pattern_map.end()) {
            m_fc = pattern_map.at(fully_connected_compressed_with_zp).get_node_shared_ptr();
        } else {
            m_fc = pattern_map.at(paged_attention).get_node_shared_ptr();
        }

        auto split_fc = [&] (std::shared_ptr<ov::Node>& org_fc, op::TP_MODE tp_mode) -> std::pair<std::shared_ptr<ov::Node>, size_t> {
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
                auto ranked_weight = std::make_shared<ov::intel_gpu::op::RankConstant>(weight_node, world_size, world_rank, tp_mode);
                std::shared_ptr<ov::Node> ranked_bias, ranked_scale, ranked_zp;
                if (!std::dynamic_pointer_cast<op::Placeholder>(m_bias)) {
                    ranked_bias = std::make_shared<ov::intel_gpu::op::RankConstant>(m_bias, world_size, world_rank, tp_mode);
                }
                auto compressed_fc = std::dynamic_pointer_cast<op::FullyConnectedCompressed>(org_fc);
                if (compressed_fc) {
                    auto scale_node = compressed_fc->get_input_node_shared_ptr(3);
                    if (tp_mode == op::TP_MODE::ALL_REDUCE) {
                        if (scale_node->get_shape()[1] > 1)
                            ranked_scale =
                                std::make_shared<ov::intel_gpu::op::RankConstant>(scale_node, world_size, world_rank, tp_mode);
                    } else {
                        ranked_scale = std::make_shared<ov::intel_gpu::op::RankConstant>(scale_node, world_size, world_rank, tp_mode);
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
                                        std::make_shared<ov::intel_gpu::op::RankConstant>(zp_node, world_size, world_rank, tp_mode);
                            } else {
                                ranked_zp = std::make_shared<ov::intel_gpu::op::RankConstant>(zp_node, world_size, world_rank, tp_mode);
                            }
                        }
                    }
                }
                auto output_type = m_fc->get_output_element_type(0);
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
                return {splitted_fc, split_dim_range};
            }
        };
        if (m_fc->get_friendly_name().find("PagedAttentionExtension") != std::string::npos) {
            const auto& m_data_in0 = pattern_map.at(in0).get_node_shared_ptr();
            auto first_fc_before_pa = fused_fc_before_pa(m_data_in0);


            // need to find a common solution instead of name
            auto new_fc = split_fc(first_fc_before_pa, op::TP_MODE::ALL_GATHERQKV).first;
            new_fc->set_friendly_name(first_fc_before_pa->get_friendly_name());
            copy_runtime_info(first_fc_before_pa, new_fc);
            replace_node(first_fc_before_pa, new_fc);

            auto print_shape = [&](const std::shared_ptr<ov::Node>& m_data) {
                 std::cout << m_data->get_friendly_name() << ": '";
                 for (size_t shape_id = 0; shape_id < m_data->get_output_partial_shape(0).size(); shape_id++) {
                     if (!m_data->get_output_partial_shape(0)[shape_id].is_dynamic()) {
                         int64_t len = m_data->get_output_partial_shape(0)[shape_id].get_length();
                         std::cout << len << ", ";
                     } else {
                         std::cout << "?" << ", ";
                     }
                 }
                 std::cout << "'\n";
            };

            std::vector<int64_t> orig_n_sizes;
            // std::cout << "new_fc outsize: " << new_fc->get_output_partial_shape(0)[2].get_length() << std::endl;
            // std::cout << "new_fc outsize: " << new_fc->get_output_partial_shape(0)[-1].get_length() << std::endl;
            // print_shape(new_fc);
            // merge weights, scale, zp
            for (int i = 0; i < 3; i ++) {
                orig_n_sizes.push_back(new_fc->get_output_partial_shape(0)[-1].get_length()/3);
            }
            // for (auto fc : fc_nodes) {
            // }
            int half_head_num = 0;
            int head_size = 0;
            for (auto user : new_fc->get_users()) {
                auto fc_user = std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(user);
                // print_shape(fc_user);
                // std::cout << "fc_user name: " << fc_user->get_friendly_name() << std::endl;

                auto split_name = fc_user->get_friendly_name() + "_tp";
                // std::cout << "new_fc->get_output_partial_shape(0).size() - 1: "
                //           << new_fc->get_output_partial_shape(0).size() - 1 << std::endl;
                auto axis_const = ov::op::v0::Constant::create(ov::element::i64,
                                                               ov::Shape{1},
                                                               {new_fc->get_output_partial_shape(0).size() - 1});
                auto split_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, orig_n_sizes);
                auto new_split = std::make_shared<ov::op::v1::VariadicSplit>(new_fc, axis_const, split_const);
                new_split->set_friendly_name(split_name);
                copy_runtime_info(fc_user, new_split);
                replace_node(fc_user, new_split);
                // print_shape(new_split);

                // auto shape0 = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2},
                // std::vector<int32_t>{680, 240}); auto reshape0 = std::make_shared<ov::op::v1::Reshape>(data, shape0,
                // false);
                int index = 0;
                for (auto user_1 : new_split->get_users()) {
                    auto split_user = std::dynamic_pointer_cast<ov::op::v1::Add>(user_1);
                    // print_shape(split_user);
                    // std::cout << "split_user name: " << split_user->get_friendly_name() << std::endl;
                    auto rank_constant = std::make_shared<ov::intel_gpu::op::RankConstant>(
                        split_user->get_input_node_shared_ptr(1), world_size, world_rank, op::TP_MODE::ALL_REDUCE);
                    // print_shape(rank_constant);
                    auto new_add = std::make_shared<ov::op::v1::Add>(new_split->output(index), rank_constant);
                    new_add->set_friendly_name(split_user->get_friendly_name() + "_tp");
                    copy_runtime_info(split_user, new_add);
                    replace_node(split_user, new_add);
                    // print_shape(new_add);
                    for (auto user_1 : new_add->get_users()) {
                        auto reshape_user = std::dynamic_pointer_cast<ov::op::v1::Reshape>(user_1);
                        print_shape(reshape_user);
                        std::cout << "*************************************\n";
                        // std::cout << "split_user name: " << split_user->get_friendly_name() << std::endl;
                        half_head_num = reshape_user->get_output_partial_shape(0)[2].get_length() / 2;
                        head_size = reshape_user->get_output_partial_shape(0)[3].get_length();
                        auto shape0 = std::make_shared<ov::op::v0::Constant>(
                            ov::element::i32,
                            ov::Shape{4},
                            std::vector<int32_t>{-1,
                                                 1,
                                                 half_head_num,
                                                 head_size});
                        // const auto input_node = new_split->get_input_source_output(index);
                        auto new_reshape = std::make_shared<ov::op::v1::Reshape>(new_add->output(0), shape0, true);
                        new_reshape->set_friendly_name(reshape_user->get_friendly_name() + "_tp");
                        copy_runtime_info(reshape_user, new_reshape);
                        replace_node(reshape_user, new_reshape);
                        // print_shape(new_reshape);
                        index++;
                        if (index == 3) {
                            // for (auto user_2 : new_reshape->get_users()) {
                            //     auto reahpe_user = std::dynamic_pointer_cast<ov::op::v1::Reshape>(user_2);
                            //     print_shape(reahpe_user);
                            //     auto shape1 = std::make_shared<ov::op::v0::Constant>(ov::element::i32,
                            //                                                     ov::Shape{2},
                            //                                                     std::vector<int32_t>{-1, 2048});
                            //     auto new_reshape_2 = std::make_shared<ov::op::v1::Reshape>(new_reshape, shape1, true);
                            //     copy_runtime_info(reahpe_user, new_reshape_2);
                            //     replace_node(reahpe_user, new_reshape_2);
                            //     print_shape(new_reshape_2);
                            // // std::cout << "reahpe_user name: " << reahpe_user->get_friendly_name() << std::endl;
                            // // index++;
                            // // std::cout << "index: " << index << std::endl;
                            // }
                            continue;
                        }
                        for (auto user_2 : new_reshape->get_users()) {
                            auto reahpe_user = std::dynamic_pointer_cast<ov::op::v1::Reshape>(user_2);
                            // print_shape(reahpe_user);
                            // std::cout << "reahpe_user name: " << reahpe_user->get_friendly_name() << std::endl;
                            auto shape0 = std::make_shared<ov::op::v0::Constant>(ov::element::i32,
                                                                                ov::Shape{4},
                                                                                std::vector<int32_t>{-1, half_head_num, 1, head_size});
                            // const auto input_node = new_split->get_input_source_output(index);
                            auto new_transpose = std::make_shared<ov::op::v1::Reshape>(new_reshape, shape0, true);
                            new_transpose->set_friendly_name(reahpe_user->get_friendly_name() + "_tp");
                            copy_runtime_info(reahpe_user, new_transpose);
                            replace_node(reahpe_user, new_transpose);
                            // print_shape(new_transpose);


                            auto reshpe_after_add = std::dynamic_pointer_cast<ov::op::v1::Reshape>(new_transpose->get_users()[0]->get_users()[0]);
                            auto shape1 = std::make_shared<ov::op::v0::Constant>(ov::element::i32,
                                                                                ov::Shape{4},
                                                                                std::vector<int32_t>{-1, 1, half_head_num, head_size});
                            auto new_reshape_1 =  std::make_shared<ov::op::v1::Reshape>(new_transpose->get_users()[0], shape1, true);
                            new_reshape_1->set_friendly_name(reshpe_after_add->get_friendly_name() + "_tp");
                            copy_runtime_info(reshpe_after_add, new_reshape_1);
                            replace_node(reshpe_after_add, new_reshape_1);
                            // print_shape(new_reshape_1);
                            for (auto user : new_reshape_1->get_users()) {
                                auto reahpe_user_2 = std::dynamic_pointer_cast<ov::op::v1::Reshape>(user);
                                // print_shape(reahpe_user_2);
                                auto shape1 = std::make_shared<ov::op::v0::Constant>(ov::element::i32,
                                                                                ov::Shape{2},
                                                                                std::vector<int32_t>{-1, half_head_num * head_size});
                                auto new_reshape_2 = std::make_shared<ov::op::v1::Reshape>(new_reshape_1, shape1, true);
                                new_reshape_2->set_friendly_name(reahpe_user_2->get_friendly_name() + "_tp");
                                copy_runtime_info(reahpe_user_2, new_reshape_2);
                                replace_node(reahpe_user_2, new_reshape_2);
                                // print_shape(new_reshape_2);
                            }
                        }
                    }
                }
            }
            std::cout << "fc fuesd split end\n";
            std::shared_ptr<ov::Node> first_fc_after_pa = nullptr;
            {
                auto root = m.get_match_root();
                if (root) {
                    first_fc_after_pa = find_first_fc_after_pa(root);
                }
            }
            if (first_fc_after_pa) {
                auto compressed_fc = std::dynamic_pointer_cast<op::FullyConnectedCompressed>(first_fc_after_pa);

                if (compressed_fc && (compressed_fc->get_input_node_shared_ptr(3)->get_shape()[1] % 2 != 0)) {
                    auto scale_node_dims = compressed_fc->get_input_node_shared_ptr(3)->get_shape()[1];
                    if (scale_node_dims != 1 && scale_node_dims % 2 != 0) {
                        int pa_split_index_length = m_fc->get_output_partial_shape(0)[-1].get_length();
                        std::shared_ptr<ov::intel_gpu::op::SyncTensor> sync_node;
                        sync_node = std::make_shared<ov::intel_gpu::op::SyncTensor>(m_fc->output(0),
                                                                                    world_size,
                                                                                    pa_split_index_length,
                                                                                    ov::element::f16);
                        sync_node->set_friendly_name(m_fc->get_friendly_name() + "_TP");

                        auto concat_node = std::make_shared<ov::op::v0::Concat>(sync_node->outputs(), -1);
                        concat_node->set_friendly_name(m_fc->get_friendly_name() + "_ALLGATHER");
                        std::cout << "concat_node->outputs(): " << concat_node->outputs().size() << std::endl;
                        copy_runtime_info(m_fc, concat_node);
                        m_fc->get_users()[0]->input(0).replace_source_output(concat_node->output(0));
                        return true;
                    }
                }
                {
                    for (auto user_1 : m_fc->get_users()) {
                        auto reshape_user = std::dynamic_pointer_cast<ov::op::v1::Reshape>(user_1);
                        if (reshape_user) {
                            std::map<int, std::shared_ptr<ov::Node>> org_users;
                            for (auto u : user_1->get_users()) {
                                for (size_t idx = 0; idx < u->inputs().size(); ++idx) {
                                    if (u->get_input_node_shared_ptr(idx) == user_1) {
                                        org_users.insert({idx, u});
                                    }
                                }
                            }
                            auto shape0 = std::make_shared<ov::op::v0::Constant>(
                                ov::element::i32,
                                ov::Shape{4},
                                std::vector<int32_t>{-1, 1, half_head_num, head_size});
                            auto new_reshape = std::make_shared<ov::op::v1::Reshape>(m_fc->output(0), shape0, true);
                            new_reshape->set_friendly_name(reshape_user->get_friendly_name() + "_tp");
                            for (auto& iter : org_users) {
                                iter.second->input(iter.first).replace_source_output(new_reshape->output(0));
                            }
                            reshape_user->clear_control_dependencies();
                            for (auto user : new_reshape->get_users()) {
                                auto reahpe_user_2 = std::dynamic_pointer_cast<ov::op::v1::Reshape>(user);
                                auto shape1 = std::make_shared<ov::op::v0::Constant>(
                                    ov::element::i32,
                                    ov::Shape{3},
                                    std::vector<int32_t>{-1, 1, half_head_num * head_size});
                                auto new_reshape_2 = std::make_shared<ov::op::v1::Reshape>(new_reshape, shape1, true);
                                new_reshape_2->set_friendly_name(reahpe_user_2->get_friendly_name() + "_tp");
                                copy_runtime_info(reahpe_user_2, new_reshape_2);
                                replace_node(reahpe_user_2, new_reshape_2);
                            }
                        }
                    }
                }

                std::map<int, std::shared_ptr<ov::Node>> org_users;
                for (auto u : first_fc_after_pa->get_users()) {
                    for (size_t idx = 0; idx < u->inputs().size(); ++idx) {
                        if (u->get_input_node_shared_ptr(idx) == first_fc_after_pa) {
                            org_users.insert({idx, u});
                        }
                    }
                }
                auto new_fc = split_fc(first_fc_after_pa, op::TP_MODE::ALL_REDUCE).first;
                std::shared_ptr<ov::intel_gpu::op::SyncTensor> sync_node;
                sync_node = std::make_shared<ov::intel_gpu::op::SyncTensor>(
                    new_fc,
                    world_size,
                    first_fc_after_pa->get_input_node_shared_ptr(1)->get_shape()[-1],
                    first_fc_after_pa->get_element_type(),
                    ov::intel_gpu::op::TP_MODE::ALL_REDUCE);
                sync_node->set_friendly_name(first_fc_after_pa->get_friendly_name() + "_TP");
                copy_runtime_info(first_fc_after_pa, new_fc);
                for (auto& iter : org_users) {
                    iter.second->input(iter.first).replace_source_output(sync_node->output(0));
                }
                first_fc_after_pa->clear_control_dependencies();
            } else {
                int pa_split_index_length = m_fc->get_output_partial_shape(0)[-1].get_length();
                std::shared_ptr<ov::intel_gpu::op::SyncTensor> sync_node;
                sync_node = std::make_shared<ov::intel_gpu::op::SyncTensor>(m_fc->output(0),
                                                                            world_size,
                                                                            pa_split_index_length,
                                                                            ov::element::f16);
                sync_node->set_friendly_name(m_fc->get_friendly_name() + "_TP");

                auto concat_node = std::make_shared<ov::op::v0::Concat>(sync_node->outputs(), -1);
                concat_node->set_friendly_name(m_fc->get_friendly_name() + "_ALLGATHER");
                std::cout << "concat_node->outputs(): " << concat_node->outputs().size() << std::endl;
                copy_runtime_info(m_fc, concat_node);
                m_fc->get_users()[0]->input(0).replace_source_output(concat_node->output(0));
            }
        }
        // else {
        //     std::cout << "*********************\n";
        //     // some accuracy lost, disable for now
        //     auto splitted_context = split_fc(m_fc, op::TP_MODE::ALL_GATHERH);
        //     auto new_fc = splitted_context.first;
        //     new_fc->set_friendly_name(m_fc->get_friendly_name());
        //     copy_runtime_info(m_fc, new_fc);
        //     replace_node(m_fc, new_fc);

        //     if (new_fc->get_users().size() == 1) {
        //         for (auto& iter : new_fc->get_users()) {
        //             if (ov::is_type<ov::op::v1::Multiply>(iter))
        //                 return true;
        //         }
        //     }
        //     std::shared_ptr<ov::op::v4::Swish> activation;
        //     std::shared_ptr<ov::op::v1::Multiply> eltwise_node;
        //     //bool elwise_flag = false;
        //     for (auto& iter : new_fc->get_users()) {
        //         if (ov::is_type<ov::op::v4::Swish>(iter)) {
        //             activation = std::dynamic_pointer_cast<ov::op::v4::Swish>(iter);
        //             if (activation->get_users().size() == 1) {
        //                 for (auto& iter2 : activation->get_users())
        //                     if (ov::is_type<ov::op::v1::Multiply>(iter2))
        //                         eltwise_node = std::dynamic_pointer_cast<ov::op::v1::Multiply>(iter2);
        //             }
        //         }
        //     }
        //     {
        //         std::map<int, std::shared_ptr<ov::Node>> org_users;
        //         auto node_to_operate = eltwise_node ? eltwise_node : activation ? activation : new_fc;
        //         for (auto u : node_to_operate->get_users()) {
        //             for (size_t idx = 0; idx < u->inputs().size(); ++idx) {
        //                 if (u->get_input_node_shared_ptr(idx) == node_to_operate) {
        //                     org_users.insert({idx, u});
        //                 }
        //             }
        //         }
        //         std::shared_ptr<ov::intel_gpu::op::SyncTensor> sync_node;
        //         sync_node = std::make_shared<ov::intel_gpu::op::SyncTensor>(node_to_operate, world_size, splitted_context.second,
        //                                                                    new_fc->get_element_type());
        //         sync_node->set_friendly_name(new_fc->get_friendly_name()+ "_TP");

        //         auto concat_node = std::make_shared<ov::op::v0::Concat>(sync_node->outputs(), -1);
        //         concat_node->set_friendly_name(new_fc->get_friendly_name()+ "_ALLGATHER");
        //         copy_runtime_info(new_fc, concat_node);
        //         for (auto& iter : org_users) {
        //             iter.second->input(iter.first).replace_source_output(concat_node->output(0));
        //         }
        //         new_fc->clear_control_dependencies();
        //     }
        // }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fully_connected_m, "FullyConnectedTensorParallelFusion");
    this->register_matcher(m, callback);
}
}  // namespace intel_gpu
}  // namespace ov