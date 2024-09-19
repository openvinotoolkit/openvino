// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remaining_fc_parallel.hpp"
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
std::shared_ptr<ov::Node> RemainFCParallelFusion::find_first_fc_after_multiply(std::shared_ptr<ov::Node> root_node) {
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
    return find_first_fc_after_multiply(cur_node);
}

RemainFCParallelFusion::RemainFCParallelFusion(size_t world_size, size_t world_rank) {
    using namespace ov::pass::pattern;
    auto data = any_input();
    auto weights = any_input();
    auto bias = any_input();
    auto fully_connected = wrap_type<op::FullyConnected>({data, weights, bias});
    auto fully_connected_compressed = wrap_type<op::FullyConnectedCompressed>({data, weights, bias, any_input()});
    auto fully_connected_compressed_with_zp = wrap_type<op::FullyConnectedCompressed>({data, weights, bias, any_input(), any_input()});
    auto fully_connected_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{fully_connected,
                                                                                    fully_connected_compressed,
                                                                                    fully_connected_compressed_with_zp});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        OPENVINO_ASSERT(pattern_map.count(fully_connected_compressed_with_zp) ||
                        pattern_map.count(fully_connected_compressed) || pattern_map.count(fully_connected));
        std::shared_ptr<Node> m_fc = nullptr;
        std::shared_ptr<Node> m_pa = nullptr;
        if (pattern_map.find(fully_connected) != pattern_map.end()) {
            m_fc = pattern_map.at(fully_connected).get_node_shared_ptr();
        } else if (pattern_map.find(fully_connected_compressed) != pattern_map.end()) {
            m_fc = pattern_map.at(fully_connected_compressed).get_node_shared_ptr();
        } else if (pattern_map.find(fully_connected_compressed_with_zp) != pattern_map.end()) {
            m_fc = pattern_map.at(fully_connected_compressed_with_zp).get_node_shared_ptr();
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
                        ranked_scale = scale_node;
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
                org_fc->get_rt_info().insert({"splitted", true});
                return {splitted_fc, split_dim_range};
            }
        };
        {
            auto get_output_node = [](const ov::Output<ov::Node>& output) -> std::shared_ptr<ov::Node> {
                return output.get_node_shared_ptr();
            };
            auto get_input_node = [&get_output_node](const ov::Input<ov::Node>& input) -> std::shared_ptr<ov::Node> {
                return get_output_node(input.get_source_output());
            };
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
            auto fc_after_pa_sync = [&](std::shared_ptr<ov::Node>& fc_node) {
                std::map<int, std::shared_ptr<ov::Node>> org_users;
                for (auto u : fc_node->get_users()) {
                    for (size_t idx = 0; idx < u->inputs().size(); ++idx) {
                        if (u->get_input_node_shared_ptr(idx) == fc_node) {
                            org_users.insert({idx, u});
                        }
                    }
                }
                // print_shape(fc_node->get_input_node_shared_ptr(0));
                // print_shape(fc_node->get_input_node_shared_ptr(1));
                // print_shape(fc_node->get_input_node_shared_ptr(2));
                // print_shape(fc_node->get_input_node_shared_ptr(3));
                auto new_fc = split_fc(fc_node, op::TP_MODE::ALL_REDUCE).first;
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
            if (m_fc->get_rt_info().find("splitted") != m_fc->get_rt_info().end()) {
                if (m_fc->get_rt_info()["splitted"].as<bool>()) {
                    return false;
                }
            }
            // std::cout << "m_fc->get_friendly_name(): " << m_fc->get_friendly_name() << std::endl;
            // some accuracy lost, disable for now
            if (m_fc->get_friendly_name().find("mlp.gate_proj") != std::string::npos) {
                auto splitted_context = split_fc(m_fc, op::TP_MODE::ALL_GATHERH);
                auto new_fc = splitted_context.first;
                new_fc->set_friendly_name(m_fc->get_friendly_name());
                copy_runtime_info(m_fc, new_fc);
                replace_node(m_fc, new_fc);

                // if (new_fc->get_users().size() == 1) {
                //     for (auto& iter : new_fc->get_users()) {
                //         if (ov::is_type<ov::op::v1::Multiply>(iter)) {
                //             // return true;
                //             std::shared_ptr<ov::Node> first_fc_after_pa = find_first_fc_after_multiply(new_fc);
                //             if (first_fc_after_pa != nullptr) {
                //                 std::cout << "first_fc_after_pa: " << first_fc_after_pa->get_friendly_name() << std::endl;
                //                 fc_after_pa_sync(first_fc_after_pa);
                //             }
                //         }
                //     }
                // }
                std::shared_ptr<ov::op::v4::Swish> activation;
                std::shared_ptr<ov::op::v1::Multiply> eltwise_node;
                //bool elwise_flag = false;
                for (auto& iter : new_fc->get_users()) {
                    if (ov::is_type<ov::op::v4::Swish>(iter)) {
                        activation = std::dynamic_pointer_cast<ov::op::v4::Swish>(iter);
                        // print_shape(activation);
                        // print_shape(activation->get_input_node_shared_ptr(0));
                        // print_shape(new_fc);
                        auto new_swish = std::make_shared<ov::op::v4::Swish>(activation->get_input_source_output(0));
                        new_swish->set_friendly_name(activation->get_friendly_name());
                        copy_runtime_info(activation, new_swish);
                        replace_node(activation, new_swish);
                        // print_shape(new_swish);


                        if (new_swish->get_users().size() == 1) {
                            for (auto& iter2 : new_swish->get_users())
                                if (ov::is_type<ov::op::v1::Multiply>(iter2)) {
                                    eltwise_node = std::dynamic_pointer_cast<ov::op::v1::Multiply>(iter2);
                                    // std::cout << eltwise_node->get_friendly_name() << std::endl;
                                    // print_shape(eltwise_node);
                                    auto up_node = get_input_node(eltwise_node->inputs()[1]);

                                    std::map<int, std::shared_ptr<ov::Node>> org_users;
                                    for (auto u : up_node->get_users()) {
                                        for (size_t idx = 0; idx < u->inputs().size(); ++idx) {
                                            if (u->get_input_node_shared_ptr(idx) == up_node) {
                                                org_users.insert({idx, u});
                                            }
                                        }
                                    }

                                    // std::cout << up_node->get_friendly_name() << std::endl;
                                    // print_shape(up_node);
                                    auto splitted_context = split_fc(up_node, op::TP_MODE::ALL_GATHERH);
                                    auto new_up = splitted_context.first;
                                    new_up->set_friendly_name(up_node->get_friendly_name());
                                    copy_runtime_info(up_node, new_up);
                                    // replace_node(up_node, new_up);
                                    for (auto& iter : org_users) {
                                        iter.second->input(iter.first).replace_source_output(new_up->output(0));
                                    }
                                    // print_shape(new_up);
                                    // std::cout << "**********************************\n";

                                    auto new_multiply = std::make_shared<ov::op::v1::Multiply>(
                                        eltwise_node->get_input_source_output(0),
                                        eltwise_node->get_input_source_output(1));
                                    new_multiply->set_friendly_name(eltwise_node->get_friendly_name());
                                    copy_runtime_info(eltwise_node, new_multiply);
                                    replace_node(eltwise_node, new_multiply);
                                    // print_shape(new_multiply);
                                    // print_shape(new_multiply->get_input_node_shared_ptr(0));
                                    // print_shape(new_multiply->get_input_node_shared_ptr(1));


                                    std::shared_ptr<ov::Node> first_fc_after_pa = find_first_fc_after_multiply(new_multiply);
                                    if (first_fc_after_pa != nullptr) {
                                        // std::cout << "first_fc_after_pa: " << first_fc_after_pa->get_friendly_name() << std::endl;
                                        fc_after_pa_sync(first_fc_after_pa);
                                    }
                                }
                        }
                    }
                }
            }
            // auto splitted_context = split_fc(m_fc, op::TP_MODE::ALL_GATHERH);
            // auto new_fc = splitted_context.first;
            // new_fc->set_friendly_name(m_fc->get_friendly_name());
            // copy_runtime_info(m_fc, new_fc);
            // replace_node(m_fc, new_fc);

            // // if (new_fc->get_users().size() == 1) {
            // //     for (auto& iter : new_fc->get_users()) {
            // //         if (ov::is_type<ov::op::v1::Multiply>(iter)) {
            // //             // return true;
            // //             std::shared_ptr<ov::Node> first_fc_after_pa = find_first_fc_after_multiply(new_fc);
            // //             if (first_fc_after_pa != nullptr) {
            // //                 std::cout << "first_fc_after_pa: " << first_fc_after_pa->get_friendly_name() << std::endl;
            // //                 fc_after_pa_sync(first_fc_after_pa);
            // //             }
            // //         }
            // //     }
            // // }
            // std::shared_ptr<ov::op::v4::Swish> activation;
            // std::shared_ptr<ov::op::v1::Multiply> eltwise_node;
            // //bool elwise_flag = false;
            // for (auto& iter : new_fc->get_users()) {
            //     if (ov::is_type<ov::op::v4::Swish>(iter)) {
            //         activation = std::dynamic_pointer_cast<ov::op::v4::Swish>(iter);
            //         if (activation->get_users().size() == 1) {
            //             for (auto& iter2 : activation->get_users())
            //                 if (ov::is_type<ov::op::v1::Multiply>(iter2)) {
            //                     eltwise_node = std::dynamic_pointer_cast<ov::op::v1::Multiply>(iter2);
            //                     std::cout << eltwise_node->get_friendly_name() << std::endl;
            //                 }
            //         }
            //     }
            // }
            // {
                // std::map<int, std::shared_ptr<ov::Node>> org_users;
                // auto node_to_operate = eltwise_node ? eltwise_node : activation ? activation : new_fc;
                // for (auto u : node_to_operate->get_users()) {
                //     for (size_t idx = 0; idx < u->inputs().size(); ++idx) {
                //         if (u->get_input_node_shared_ptr(idx) == node_to_operate) {
                //             org_users.insert({idx, u});
                //         }
                //     }
                // }
                // std::shared_ptr<ov::Node> first_fc_after_pa = find_first_fc_after_multiply(node_to_operate);
                // if (first_fc_after_pa != nullptr) {
                //     std::cout << "first_fc_after_pa: " << first_fc_after_pa->get_friendly_name() << std::endl;
                //     fc_after_pa_sync(first_fc_after_pa);
                // }

                // std::shared_ptr<ov::intel_gpu::op::SyncTensor> sync_node;
                // sync_node = std::make_shared<ov::intel_gpu::op::SyncTensor>(node_to_operate, world_size, splitted_context.second,
                //                                                            new_fc->get_element_type());
                // sync_node->set_friendly_name(new_fc->get_friendly_name()+ "_TP");

                // auto concat_node = std::make_shared<ov::op::v0::Concat>(sync_node->outputs(), -1);
                // concat_node->set_friendly_name(new_fc->get_friendly_name()+ "_ALLGATHER");
                // copy_runtime_info(new_fc, concat_node);
                // for (auto& iter : org_users) {
                //     iter.second->input(iter.first).replace_source_output(concat_node->output(0));
                // }
                // new_fc->clear_control_dependencies();
            // }
        }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fully_connected_m, "FullyConnectedRemainFCParallelFusion");
    this->register_matcher(m, callback);
}
}  // namespace intel_gpu
}  // namespace ov