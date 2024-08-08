// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fc_all_reduce.hpp"

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/op/add.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "intel_gpu/op/sync_tensor.hpp"
#include "intel_gpu/op/util.hpp"
#include "openvino/op/slice.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "intel_gpu/op/rank_constant.hpp"
#include <cstdlib>

namespace ov {
namespace intel_gpu {

FullyConnectedSplitInput::FullyConnectedSplitInput(size_t world_size, size_t rank_size) {
    using namespace ov::pass::pattern;

    auto data = any_input();
    auto weights = any_input();
    auto bias = any_input();
    auto fully_connected = wrap_type<op::FullyConnected>({data, weights, bias}, consumers_count(1));
    auto fully_connected_compressed = wrap_type<op::FullyConnectedCompressed>({data, weights, bias, any_input()});
    auto fully_connected_compressed_with_zp = wrap_type<op::FullyConnectedCompressed>({data, weights, bias, any_input(), any_input()});
    auto fully_connected_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{fully_connected,
                                                                                    fully_connected_compressed,
                                                                                    fully_connected_compressed_with_zp});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        OPENVINO_ASSERT(pattern_map.count(fully_connected_compressed_with_zp) ||
                        pattern_map.count(fully_connected_compressed) || pattern_map.count(fully_connected));

        const auto& m_data = pattern_map.at(data).get_node_shared_ptr();
        const auto& m_weights = pattern_map.at(weights).get_node_shared_ptr();
        const auto& m_bias = pattern_map.at(bias).get_node_shared_ptr();
        // const auto& m_fc = pattern_map.at(fully_connected).get_node_shared_ptr();

        std::shared_ptr<Node> m_fc = nullptr;
        if (pattern_map.find(fully_connected) != pattern_map.end()) {
            m_fc = pattern_map.at(fully_connected).get_node_shared_ptr();
        } else if (pattern_map.find(fully_connected_compressed) != pattern_map.end()) {
            m_fc = pattern_map.at(fully_connected_compressed).get_node_shared_ptr();
        } else {
            m_fc = pattern_map.at(fully_connected_compressed_with_zp).get_node_shared_ptr();
        }

        std::map<int, std::shared_ptr<ov::Node>> org_users;
        for (auto u : m_fc->get_users()) {
            for (size_t idx = 0; idx < u->inputs().size(); ++idx) {
                if (u->get_input_node_shared_ptr(idx) == m_fc) {
                    org_users.insert({idx, u});
                }
            }
        }

        int w_rank = rank_size;
        int w_size = world_size;
        if (w_size != 1) {
            int slice_axis_length = m_data->get_output_partial_shape(0)[-1].get_length();
            auto scop = std::div(slice_axis_length, w_size).quot;
            auto start = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {w_rank * scop});
            auto stop = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {(w_rank + 1) * scop});
            auto step = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
            int64_t input_axis_value = m_data->get_output_partial_shape(0).size() - 1;
            auto input_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {input_axis_value});
            auto data_slice = std::make_shared<ov::op::v8::Slice>(m_data, start, stop, step, input_axis);

            auto ranked_weight = std::make_shared<ov::intel_gpu::op::RankConstant>(m_weights, world_size, rank_size);

            std::shared_ptr<ov::Node> ranked_bias, ranked_scale, ranked_zp;

            if (!std::dynamic_pointer_cast<op::Placeholder>(m_bias)) {
                ranked_bias = std::make_shared<ov::intel_gpu::op::RankConstant>(m_bias, world_size, rank_size);
            }

            std::shared_ptr<Node> new_fc = nullptr;
            auto compressed_fc = std::dynamic_pointer_cast<op::FullyConnectedCompressed>(m_fc);
            if (compressed_fc) {
                auto scale_node = compressed_fc->get_input_node_shared_ptr(3);
                if (scale_node->get_shape()[1] > 1)
                    ranked_scale = std::make_shared<ov::intel_gpu::op::RankConstant>(scale_node, world_size, rank_size);
                else
                    ranked_scale = compressed_fc->get_input_node_shared_ptr(3);
                if (compressed_fc->inputs().size() > 4) {
                    auto zp_node = compressed_fc->get_input_node_shared_ptr(4);
                    if (zp_node->get_shape()[1] > 1)
                        ranked_zp =
                            std::make_shared<ov::intel_gpu::op::RankConstant>(zp_node, world_size, rank_size);
                    else
                        ranked_zp = compressed_fc->get_input_node_shared_ptr(4);
                    new_fc = std::make_shared<op::FullyConnectedCompressed>(data_slice,
                                                                            ranked_weight,
                                                                            ranked_bias ? ranked_bias : m_bias,
                                                                            ranked_scale,
                                                                            ranked_zp,
                                                                            m_fc->get_element_type());
                } else {
                    new_fc = std::make_shared<op::FullyConnectedCompressed>(data_slice,
                                                                            ranked_weight,
                                                                            ranked_bias ? ranked_bias : m_bias,
                                                                            ranked_scale,
                                                                            m_fc->get_element_type());
                }
            } else {
                new_fc =
                    std::make_shared<op::FullyConnected>(data_slice, ranked_weight, m_bias, m_fc->get_element_type());
            }
            std::shared_ptr<ov::intel_gpu::op::SyncTensor> sync_node;
            sync_node = std::make_shared<ov::intel_gpu::op::SyncTensor>(new_fc,
                                                                        w_size,
                                                                        m_weights->get_shape()[-1],
                                                                        m_fc->get_element_type(),
                                                                        ov::intel_gpu::op::TP_MODE::ALL_REDUCE);
            sync_node->set_friendly_name(m_fc->get_friendly_name() + "_TP");
            copy_runtime_info(m_fc, new_fc);
            for (auto& iter : org_users) {
                iter.second->input(iter.first).replace_source_output(sync_node->output(0));
            }
            m_fc->clear_control_dependencies();
        }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fully_connected_m, "FullyConnectedSplitInput");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
