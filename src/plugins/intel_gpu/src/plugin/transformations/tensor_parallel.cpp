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
#include "intel_gpu/op/sync_tensor.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/utils/utils.hpp"
namespace ov {
namespace intel_gpu {

TensorParallelFusion::TensorParallelFusion(size_t world_size) {
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
    //auto swish_m = wrap_type<ov::op::v4::Swish>({fully_connected_m});
    //auto fc_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{swish_m, fully_connected_m});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        OPENVINO_ASSERT(pattern_map.count(fully_connected_compressed_with_zp) ||
                        pattern_map.count(fully_connected_compressed) || pattern_map.count(fully_connected));
        std::shared_ptr<Node> m_fc = nullptr;
        if (pattern_map.find(fully_connected) != pattern_map.end()) {
            m_fc = pattern_map.at(fully_connected).get_node_shared_ptr();
        } else if (pattern_map.find(fully_connected_compressed) != pattern_map.end()) {
            m_fc = pattern_map.at(fully_connected_compressed).get_node_shared_ptr();
        } else {
            m_fc = pattern_map.at(fully_connected_compressed_with_zp).get_node_shared_ptr();
        }
        if (m_fc->get_users().size() == 1) {
            for (auto& iter : m_fc->get_users()) {
                if (ov::is_type<ov::op::v1::Multiply>(iter))
                    return false;
            }
        }
        std::shared_ptr<ov::op::v4::Swish> activation;
        std::shared_ptr<ov::op::v1::Multiply> eltwise_node;
        //bool elwise_flag = false;
        for (auto& iter : m_fc->get_users()) {
            if (ov::is_type<ov::op::v4::Swish>(iter)) {
                activation = std::dynamic_pointer_cast<ov::op::v4::Swish>(iter);
                if (activation->get_users().size() == 1) {
                    for (auto& iter2 : activation->get_users())
                        if (ov::is_type<ov::op::v1::Multiply>(iter2))
                            eltwise_node = std::dynamic_pointer_cast<ov::op::v1::Multiply>(iter2);
                }
                // have accuracy issue... to be debugged further
                //activation = nullptr;
            }
        }
        const auto& m_data = pattern_map.at(data).get_node_shared_ptr();
        const auto& m_weights = pattern_map.at(weights).get_node_shared_ptr();
        const auto& m_bias = pattern_map.at(bias).get_node_shared_ptr();
        {
            std::map<int, std::shared_ptr<ov::Node>> org_users;
            auto node_to_operate = eltwise_node ? eltwise_node : activation ? activation : m_fc;
            for (auto u : node_to_operate->get_users()) {
                for (size_t idx = 0; idx < u->inputs().size(); ++idx) {
                    if (u->get_input_node_shared_ptr(idx) == node_to_operate) {
                        org_users.insert({idx, u});
                    }
                }
            }
            auto wgt_item = m_fc->get_input_node_shared_ptr(1);
            auto weights_pshape = wgt_item->get_output_partial_shape(0);
            auto reshaped_pshape = weights_pshape.to_shape();
            if (weights_pshape.size() != 2) {
                auto reshape_to_2d = [](const ov::PartialShape& shape, const ov::Dimension& feature, size_t rank) {
                    auto static_shape = shape.to_shape();
                    size_t total = std::accumulate(static_shape.begin(), static_shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
                    auto dim = feature.is_static() ? feature.get_length() : static_cast<int64_t>(static_shape[rank - 1]);
                    return ov::PartialShape{ static_cast<int64_t>(total) / dim, dim }.to_shape();
                };
                auto input0_pshape = m_fc->get_input_node_shared_ptr(0)->get_output_partial_shape(0);
                auto feature = input0_pshape[input0_pshape.size() - 1ul];
                auto reshaped_pshape = weights_pshape;
                reshaped_pshape = reshape_to_2d(weights_pshape, feature, weights_pshape.size());
            }
            // split weight
            auto split_dim_range = reshaped_pshape[0];
            //auto has_activation = pattern_map.count(swish_m);
            std::shared_ptr<ov::intel_gpu::op::SyncTensor> sync_node;
            sync_node = std::make_shared<ov::intel_gpu::op::SyncTensor>(node_to_operate, world_size, split_dim_range, m_fc->get_element_type());
            sync_node->set_friendly_name(m_fc->get_friendly_name()+ "_TP");

            auto concat_node = std::make_shared<ov::op::v0::Concat>(sync_node->outputs(), -1);
            concat_node->set_friendly_name(m_fc->get_friendly_name()+ "_ALLGATHER");
            copy_runtime_info(m_fc, concat_node);
            for (auto& iter : org_users) {
                iter.second->input(iter.first).replace_source_output(concat_node->output(0));
            }
            m_fc->clear_control_dependencies();
        }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fully_connected_m, "FullyConnectedTensorParallelFusion");
    this->register_matcher(m, callback);
}
}  // namespace intel_gpu
}  // namespace ov