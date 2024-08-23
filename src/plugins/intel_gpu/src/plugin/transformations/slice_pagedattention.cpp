// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "slice_pagedattention.hpp"

#include <cstdlib>

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "intel_gpu/op/rank_constant.hpp"
#include "intel_gpu/op/sync_tensor.hpp"
#include "intel_gpu/op/util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

PagedAttentionSplitInput::PagedAttentionSplitInput(size_t world_size, size_t rank_size) {
    using namespace ov::pass::pattern;

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
    auto fully_connected = wrap_type<ov::op::PagedAttentionExtension>(
        {in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12});
    auto paged_attention_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{fully_connected});
    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        OPENVINO_ASSERT(pattern_map.count(fully_connected));

        const auto& m_data_in0 = pattern_map.at(in0).get_node_shared_ptr();
        const auto& m_data_in1 = pattern_map.at(in1).get_node_shared_ptr();
        const auto& m_data_in2 = pattern_map.at(in2).get_node_shared_ptr();
        const auto& m_data_in3 = pattern_map.at(in3).get_node_shared_ptr();
        const auto& m_data_in4 = pattern_map.at(in4).get_node_shared_ptr();
        const auto& m_data_in5 = pattern_map.at(in5).get_node_shared_ptr();
        const auto& m_data_in6 = pattern_map.at(in6).get_node_shared_ptr();
        const auto& m_data_in7 = pattern_map.at(in7).get_node_shared_ptr();
        const auto& m_data_in8 = pattern_map.at(in8).get_node_shared_ptr();
        const auto& m_data_in9 = pattern_map.at(in9).get_node_shared_ptr();
        const auto& m_data_in10 = pattern_map.at(in10).get_node_shared_ptr();
        const auto& m_data_in11 = pattern_map.at(in11).get_node_shared_ptr();
        const auto& m_data_in12 = pattern_map.at(in12).get_node_shared_ptr();

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

        std::shared_ptr<Node> m_pa = nullptr;
        if (pattern_map.find(fully_connected) != pattern_map.end())
            m_pa = pattern_map.at(fully_connected).get_node_shared_ptr();
        print_shape(m_data_in0);
        print_shape(m_data_in1);
        print_shape(m_data_in2);
        print_shape(m_data_in3);
        print_shape(m_data_in4);
        print_shape(m_data_in5);
        print_shape(m_data_in6);
        print_shape(m_data_in7);
        print_shape(m_data_in8);
        print_shape(m_data_in9);
        print_shape(m_data_in10);
        print_shape(m_data_in11);
        print_shape(m_data_in12);
        int w_rank = rank_size;
        int w_size = world_size;
        std::cout << "w-size: " << w_size << std::endl;
        if (w_size != 1) {
            int slice_axis_length = m_data_in0->get_output_partial_shape(0)[-1].get_length();
            std::cout << "slice_axis_length: " << slice_axis_length << std::endl;
            auto scop = std::div(slice_axis_length, w_size).quot;
            auto start = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {w_rank * scop});
            auto stop = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {(w_rank + 1) * scop});
            auto step = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
            int64_t input_axis_value = m_data_in0->get_output_partial_shape(0).size() - 1;
            std::cout << "input_axis_value: " << input_axis_value << std::endl;
            auto input_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {input_axis_value});
            auto new_in0 = std::make_shared<ov::op::v8::Slice>(m_data_in0, start, stop, step, input_axis);
            // print_shape(new_in0);

            int slice_axis_length1 = m_data_in1->get_output_partial_shape(0)[-1].get_length();
            auto scop1 = std::div(slice_axis_length1, w_size).quot;
            auto start1 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {w_rank * scop1});
            auto stop1 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {(w_rank + 1) * scop1});
            auto step1 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
            int64_t input_axis_value1 = m_data_in1->get_output_partial_shape(0).size() - 1;
            auto input_axis1 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {input_axis_value1});
            auto new_in1 = std::make_shared<ov::op::v8::Slice>(m_data_in1, start1, stop1, step1, input_axis1);

            int slice_axis_length2 = m_data_in2->get_output_partial_shape(0)[-1].get_length();
            auto scop2 = std::div(slice_axis_length2, w_size).quot;
            auto start2 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {w_rank * scop2});
            auto stop2 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {(w_rank + 1) * scop2});
            auto step2 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
            int64_t input_axis_value2 = m_data_in2->get_output_partial_shape(0).size() - 1;
            auto input_axis2 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {input_axis_value2});
            auto new_in2 = std::make_shared<ov::op::v8::Slice>(m_data_in2, start2, stop2, step2, input_axis2);
            std::cout << "m_pa name: " << m_pa->get_friendly_name() << std::endl;

            // auto get_output_node = [](const ov::Output<ov::Node>& output) -> std::shared_ptr<ov::Node> {
            //     return output.get_node_shared_ptr();
            // };
            // auto get_input_node = [&get_output_node](const ov::Input<ov::Node>& input) -> std::shared_ptr<ov::Node> {
            //     return get_output_node(input.get_source_output());
            // };
            // for (auto& input : m_data_in0->inputs()) {
            //     const auto& node = get_input_node(input);
            //     std::cout << "pa input: " << node->get_friendly_name() << std::endl;
            //     auto shape0 = std::make_shared<ov::op::v0::Constant>(ov::element::i32,
            //                                                             ov::Shape{4},
            //                                                             std::vector<int32_t>{-1, 1, 16, 128});
            //     // const auto input_node = new_split->get_input_source_output(index);
            //     auto new_reshape = std::make_shared<ov::op::v1::Reshape>(new_split->output(index), shape0, true);
            //     new_reshape->set_friendly_name(split_user->get_friendly_name() + "_tp");
            //     copy_runtime_info(split_user, new_reshape);
            //     replace_node(split_user, new_reshape);


            // }
            // auto shape0 = std::make_shared<ov::op::v0::Constant>(ov::element::i32,
            //                                                             ov::Shape{4},
            //                                                             std::vector<int32_t>{-1, 1, 16, 128});
            // auto new_m_data_in0 = std::make_shared<ov::op::v1::Reshape>(new_split->output(index), shape0, true);




            OutputVector params;
            // if (m_pa->get_friendly_name() == "PagedAttentionExtension_50423") {
                // params = {m_data_in0,
                //                     m_data_in1,
                //                     m_data_in2,
                //                     m_data_in3,
                //                     m_data_in4,
                //                     m_data_in5,
                //                     m_data_in6,
                //                     m_data_in7,
                //                     m_data_in8,
                //                     m_data_in9,
                //                     m_data_in10,
                //                     m_data_in11,
                //                     m_data_in12};
            // } else {
                params = {new_in0,
                                    new_in1,
                                    new_in2,
                                    m_data_in3,
                                    m_data_in4,
                                    m_data_in5,
                                    m_data_in6,
                                    m_data_in7,
                                    m_data_in8,
                                    m_data_in9,
                                    m_data_in10,
                                    m_data_in11,
                                    m_data_in12};
            // }
            std::shared_ptr<Node> new_pa = nullptr;
            new_pa = std::make_shared<ov::op::PagedAttentionExtension>(params);

            std::shared_ptr<ov::intel_gpu::op::SyncTensor> sync_node;
            sync_node = std::make_shared<ov::intel_gpu::op::SyncTensor>(new_pa,
                                                                        w_size,
                                                                        4096,
                                                                        new_pa->get_element_type(),
                                                                        ov::intel_gpu::op::TP_MODE::ALL_REDUCE);
            sync_node->set_friendly_name(new_pa->get_friendly_name() + "_TP");

            auto concat_node = std::make_shared<ov::op::v0::Concat>(sync_node->outputs(), -1);
            concat_node->set_friendly_name(new_pa->get_friendly_name() + "_ALLGATHER");
            copy_runtime_info(m_pa, concat_node);
            replace_node(m_pa, concat_node);
            m_pa->clear_control_dependencies();
        }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(paged_attention_m, "PagedAttentionSplitInput");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
