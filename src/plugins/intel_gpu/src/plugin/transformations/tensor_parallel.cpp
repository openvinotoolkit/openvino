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

        if (m_fc->get_friendly_name().find("self_attn.q_proj/prim::PythonOp/MatMul_fused") != std::string::npos) {
            // need to find a common solution instead of name
            auto new_fc = split_fc(m_fc, op::TP_MODE::ALL_GATHERQKV).first;
            new_fc->set_friendly_name(m_fc->get_friendly_name());
            copy_runtime_info(m_fc, new_fc);
            replace_node(m_fc, new_fc);

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
            for (auto user : new_fc->get_users()) {
                auto fc_user = std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(user);
                print_shape(fc_user);
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
                print_shape(new_split);

                // auto shape0 = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2},
                // std::vector<int32_t>{680, 240}); auto reshape0 = std::make_shared<ov::op::v1::Reshape>(data, shape0,
                // false);
                int index = 0;
                for (auto user_1 : new_split->get_users()) {
                    auto split_user = std::dynamic_pointer_cast<ov::op::v1::Add>(user_1);
                    print_shape(split_user);
                    // std::cout << "split_user name: " << split_user->get_friendly_name() << std::endl;
                    auto rank_constant = std::make_shared<ov::intel_gpu::op::RankConstant>(
                        split_user->get_input_node_shared_ptr(1), world_size, world_rank, op::TP_MODE::ALL_REDUCE);
                    print_shape(rank_constant);
                    auto new_add = std::make_shared<ov::op::v1::Add>(new_split->output(index), rank_constant);
                    new_add->set_friendly_name(split_user->get_friendly_name() + "_tp");
                    copy_runtime_info(split_user, new_add);
                    replace_node(split_user, new_add);
                    print_shape(new_add);
                    for (auto user_1 : new_add->get_users()) {
                        auto reshape_user = std::dynamic_pointer_cast<ov::op::v1::Reshape>(user_1);
                        print_shape(reshape_user);
                        // std::cout << "split_user name: " << split_user->get_friendly_name() << std::endl;
                        auto shape0 = std::make_shared<ov::op::v0::Constant>(ov::element::i32,
                                                                            ov::Shape{4},
                                                                            std::vector<int32_t>{-1, 1, 20, 128});
                    // const auto input_node = new_split->get_input_source_output(index);
                        auto new_reshape = std::make_shared<ov::op::v1::Reshape>(new_add->output(0), shape0, true);
                        new_reshape->set_friendly_name(reshape_user->get_friendly_name() + "_tp");
                        copy_runtime_info(reshape_user, new_reshape);
                        replace_node(reshape_user, new_reshape);
                        print_shape(new_reshape);
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
                            print_shape(reahpe_user);
                            // std::cout << "reahpe_user name: " << reahpe_user->get_friendly_name() << std::endl;
                            auto shape0 = std::make_shared<ov::op::v0::Constant>(ov::element::i32,
                                                                                ov::Shape{4},
                                                                                std::vector<int32_t>{-1, 20, 1, 128});
                            // const auto input_node = new_split->get_input_source_output(index);
                            auto new_transpose = std::make_shared<ov::op::v1::Reshape>(new_reshape, shape0, true);
                            new_transpose->set_friendly_name(reahpe_user->get_friendly_name() + "_tp");
                            copy_runtime_info(reahpe_user, new_transpose);
                            replace_node(reahpe_user, new_transpose);
                            print_shape(new_transpose);


                            auto reshpe_after_add = std::dynamic_pointer_cast<ov::op::v1::Reshape>(new_transpose->get_users()[0]->get_users()[0]);
                            auto shape1 = std::make_shared<ov::op::v0::Constant>(ov::element::i32,
                                                                                ov::Shape{4},
                                                                                std::vector<int32_t>{-1, 1, 20, 128});
                            auto new_reshape_1 =  std::make_shared<ov::op::v1::Reshape>(new_transpose->get_users()[0], shape1, true);
                            new_reshape_1->set_friendly_name(reshpe_after_add->get_friendly_name() + "_tp");
                            copy_runtime_info(reshpe_after_add, new_reshape_1);
                            replace_node(reshpe_after_add, new_reshape_1);
                            print_shape(new_reshape_1);
                            for (auto user : new_reshape_1->get_users()) {
                                auto reahpe_user_2 = std::dynamic_pointer_cast<ov::op::v1::Reshape>(user);
                                print_shape(reahpe_user_2);
                                auto shape1 = std::make_shared<ov::op::v0::Constant>(ov::element::i32,
                                                                                ov::Shape{2},
                                                                                std::vector<int32_t>{-1, 2560});
                                auto new_reshape_2 = std::make_shared<ov::op::v1::Reshape>(new_reshape_1, shape1, true);
                                new_reshape_2->set_friendly_name(reahpe_user_2->get_friendly_name() + "_tp");
                                copy_runtime_info(reahpe_user_2, new_reshape_2);
                                replace_node(reahpe_user_2, new_reshape_2);
                                print_shape(new_reshape_2);
                            }
                        }
                    }
                }
            }
        } else if (m_fc->get_friendly_name().find("PagedAttentionExtension") != std::string::npos) {
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

            for (auto user_1 : m_fc->get_users()) {
                auto reshape_user = std::dynamic_pointer_cast<ov::op::v1::Reshape>(user_1);
                if (reshape_user) {
                    print_shape(reshape_user);
                    // std::cout << "split_user name: " << split_user->get_friendly_name() << std::endl;
                    auto shape0 = std::make_shared<ov::op::v0::Constant>(ov::element::i32,
                                                                        ov::Shape{4},
                                                                        std::vector<int32_t>{-1, 1, 20, 128});
                // const auto input_node = new_split->get_input_source_output(index);
                    auto new_reshape = std::make_shared<ov::op::v1::Reshape>(m_fc->output(0), shape0, true);
                    new_reshape->set_friendly_name(reshape_user->get_friendly_name() + "_tp");
                    // copy_runtime_info(reshape_user, new_reshape);
                    // replace_node(reshape_user, new_reshape);
                    print_shape(new_reshape);
                    std::shared_ptr<ov::intel_gpu::op::SyncTensor> sync_node;
                    sync_node =
                        std::make_shared<ov::intel_gpu::op::SyncTensor>(new_reshape, world_size, 5120, ov::element::f16);
                    sync_node->set_friendly_name(new_reshape->get_friendly_name() + "_TP");
                    print_shape(sync_node);

                    auto concat_node = std::make_shared<ov::op::v0::Concat>(sync_node->outputs(), -1);
                    concat_node->set_friendly_name(new_reshape->get_friendly_name() + "_ALLGATHER");
                    std::cout << "concat_node->outputs(): " << concat_node->outputs().size() << std::endl;
                    copy_runtime_info(reshape_user, concat_node);
                    replace_node(reshape_user, concat_node);
                }
            }

            // std::cout << "find pa\n";
            // const auto& m_data_in0 = pattern_map.at(in0).get_node_shared_ptr();
            // const auto& m_data_in1 = pattern_map.at(in1).get_node_shared_ptr();
            // const auto& m_data_in2 = pattern_map.at(in2).get_node_shared_ptr();
            // const auto& m_data_in3 = pattern_map.at(in3).get_node_shared_ptr();
            // const auto& m_data_in4 = pattern_map.at(in4).get_node_shared_ptr();
            // const auto& m_data_in5 = pattern_map.at(in5).get_node_shared_ptr();
            // const auto& m_data_in6 = pattern_map.at(in6).get_node_shared_ptr();
            // const auto& m_data_in7 = pattern_map.at(in7).get_node_shared_ptr();
            // const auto& m_data_in8 = pattern_map.at(in8).get_node_shared_ptr();
            // const auto& m_data_in9 = pattern_map.at(in9).get_node_shared_ptr();
            // const auto& m_data_in10 = pattern_map.at(in10).get_node_shared_ptr();
            // const auto& m_data_in11 = pattern_map.at(in11).get_node_shared_ptr();
            // const auto& m_data_in12 = pattern_map.at(in12).get_node_shared_ptr();

            // auto print_shape = [&](const std::shared_ptr<ov::Node>& m_data) {
            //     std::cout << m_data->get_friendly_name() << ": '";
            //     for (size_t shape_id = 0; shape_id < m_data->get_output_partial_shape(0).size(); shape_id++) {
            //         if (!m_data->get_output_partial_shape(0)[shape_id].is_dynamic()) {
            //             int64_t len = m_data->get_output_partial_shape(0)[shape_id].get_length();
            //             std::cout << len << ", ";
            //         } else {
            //             std::cout << "?" << ", ";
            //         }
            //     }
            //     std::cout << "'\n";
            // };

            // // std::shared_ptr<Node> m_pa = nullptr;
            // // if (pattern_map.find(fully_connected) != pattern_map.end())
            // //     m_pa = pattern_map.at(fully_connected).get_node_shared_ptr();
            // print_shape(m_data_in0);
            // print_shape(m_data_in1);
            // print_shape(m_data_in2);
            // print_shape(m_data_in3);
            // print_shape(m_data_in4);
            // print_shape(m_data_in5);
            // print_shape(m_data_in6);
            // print_shape(m_data_in7);
            // print_shape(m_data_in8);
            // print_shape(m_data_in9);
            // print_shape(m_data_in10);
            // print_shape(m_data_in11);
            // print_shape(m_data_in12);
            // int w_rank = world_rank;
            // int w_size = world_size;
            // std::cout << "w-size: " << w_size << std::endl;
            // std::cout << m_data_in0->get_friendly_name() << std::endl;
            // if (w_size != 1) {
            //     int slice_axis_length = m_data_in0->get_output_partial_shape(0)[-1].get_length();
            //     std::cout << "slice_axis_length: " << slice_axis_length << std::endl;
            //     auto scop = std::div(slice_axis_length, w_size).quot;
            //     auto start = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {w_rank * scop});
            //     auto stop = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {(w_rank + 1) * scop});
            //     auto step = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
            //     int64_t input_axis_value = m_data_in0->get_output_partial_shape(0).size() - 1;
            //     std::cout << "input_axis_value: " << input_axis_value << std::endl;
            //     auto input_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {input_axis_value});
            //     auto new_in0 = std::make_shared<ov::op::v8::Slice>(m_data_in0, start, stop, step, input_axis);
            //     // print_shape(new_in0);

            //     int slice_axis_length1 = m_data_in1->get_output_partial_shape(0)[-1].get_length();
            //     auto scop1 = std::div(slice_axis_length1, w_size).quot;
            //     auto start1 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {w_rank * scop1});
            //     auto stop1 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {(w_rank + 1) * scop1});
            //     auto step1 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
            //     int64_t input_axis_value1 = m_data_in1->get_output_partial_shape(0).size() - 1;
            //     auto input_axis1 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {input_axis_value1});
            //     auto new_in1 = std::make_shared<ov::op::v8::Slice>(m_data_in1, start1, stop1, step1, input_axis1);

            //     int slice_axis_length2 = m_data_in2->get_output_partial_shape(0)[-1].get_length();
            //     auto scop2 = std::div(slice_axis_length2, w_size).quot;
            //     auto start2 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {w_rank * scop2});
            //     auto stop2 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {(w_rank + 1) * scop2});
            //     auto step2 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
            //     int64_t input_axis_value2 = m_data_in2->get_output_partial_shape(0).size() - 1;
            //     auto input_axis2 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {input_axis_value2});
            //     auto new_in2 = std::make_shared<ov::op::v8::Slice>(m_data_in2, start2, stop2, step2, input_axis2);
            //     std::cout << "m_fc name: " << m_fc->get_friendly_name() << std::endl;
            //     std::cout << "m_fc->outputs(): " << m_fc->outputs().size() << std::endl;


            //     OutputVector params;
            //         params = {m_data_in0,
            //                             m_data_in1,
            //                             m_data_in2,
            //                             m_data_in3,
            //                             m_data_in4,
            //                             m_data_in5,
            //                             m_data_in6,
            //                             m_data_in7,
            //                             m_data_in8,
            //                             m_data_in9,
            //                             m_data_in10,
            //                             m_data_in11,
            //                             m_data_in12};
            //     // }
            //     std::shared_ptr<Node> new_pa = nullptr;
            //     new_pa = std::make_shared<ov::op::PagedAttentionExtension>(params);

            //     std::shared_ptr<ov::intel_gpu::op::SyncTensor> sync_node;
            //     sync_node =
            //         std::make_shared<ov::intel_gpu::op::SyncTensor>(new_pa, w_size, 5120, ov::element::f16);
            //     sync_node->set_friendly_name(new_pa->get_friendly_name() + "_TP");
            //     print_shape(sync_node);

            //     auto concat_node = std::make_shared<ov::op::v0::Concat>(sync_node->outputs(), -1);
            //     concat_node->set_friendly_name(new_pa->get_friendly_name() + "_ALLGATHER");
            //     std::cout << "concat_node->outputs(): " << concat_node->outputs().size() << std::endl;
            //     copy_runtime_info(m_fc, concat_node);
            //     replace_node(m_fc, concat_node);
            //     print_shape(concat_node);
            //     m_fc->clear_control_dependencies();
            // }
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