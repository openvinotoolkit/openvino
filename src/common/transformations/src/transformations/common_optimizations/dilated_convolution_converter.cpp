// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/dilated_convolution_converter.hpp"

#include <memory>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <vector>

#include "itt.hpp"
#include "transformations/utils/utils.hpp"

ngraph::pass::DilatedConvolutionConverter::DilatedConvolutionConverter() {
    MATCHER_SCOPE(DilatedConvolutionConverter);
    auto data_pattern = pattern::any_input();
    auto block_shape_pattern = pattern::wrap_type<opset6::Constant>();
    auto pads_begin_pattern = pattern::wrap_type<opset6::Constant>();
    auto pads_end_pattern = pattern::wrap_type<opset6::Constant>();
    auto space_to_batch_pattern = pattern::wrap_type<opset6::SpaceToBatch>(
        {data_pattern, block_shape_pattern, pads_begin_pattern, pads_end_pattern});
    auto conv_pattern = pattern::wrap_type<opset6::Convolution>({space_to_batch_pattern, pattern::any_input()});
    auto crops_begin_pattern = pattern::wrap_type<opset6::Constant>();
    auto crops_end_pattern = pattern::wrap_type<opset6::Constant>();
    auto batch_to_space_pattern = pattern::wrap_type<opset6::BatchToSpace>(
        {conv_pattern, pattern::any_input(), crops_begin_pattern, crops_end_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto block_shape =
            std::dynamic_pointer_cast<opset6::Constant>(pattern_map.at(block_shape_pattern).get_node_shared_ptr());
        if (!block_shape)
            return false;
        auto pads_begin =
            std::dynamic_pointer_cast<opset6::Constant>(pattern_map.at(pads_begin_pattern).get_node_shared_ptr());
        if (!pads_begin)
            return false;
        auto pads_end =
            std::dynamic_pointer_cast<opset6::Constant>(pattern_map.at(pads_end_pattern).get_node_shared_ptr());
        if (!pads_end)
            return false;
        auto crops_begin =
            std::dynamic_pointer_cast<opset6::Constant>(pattern_map.at(crops_begin_pattern).get_node_shared_ptr());
        if (!crops_begin)
            return false;
        auto crops_end =
            std::dynamic_pointer_cast<opset6::Constant>(pattern_map.at(crops_end_pattern).get_node_shared_ptr());
        if (!crops_end)
            return false;
        auto conv = std::dynamic_pointer_cast<opset6::Convolution>(pattern_map.at(conv_pattern).get_node_shared_ptr());
        if (!conv)
            return false;

        auto block_shape_val = block_shape->cast_vector<size_t>();

        auto dilations = conv->get_dilations();
        for (size_t i = 0; i < dilations.size(); i++)
            dilations[i] = block_shape_val[i + 2];
        auto pads_begin_val = pads_begin->cast_vector<std::ptrdiff_t>();
        auto pads_end_val = pads_end->cast_vector<std::ptrdiff_t>();
        if (!(pads_begin_val[0] == 0 && pads_begin_val[1] == 0 && pads_end_val[0] == 0 && pads_end_val[1] == 0))
            return false;
        auto crops_begin_val = crops_begin->cast_vector<std::ptrdiff_t>();
        auto crops_end_val = crops_end->cast_vector<std::ptrdiff_t>();
        std::vector<std::ptrdiff_t> new_pads_begin;
        for (size_t i = 2; i < pads_begin_val.size(); i++)
            new_pads_begin.push_back(pads_begin_val[i] - crops_begin_val[i]);
        std::vector<std::ptrdiff_t> new_pads_end;
        for (size_t i = 2; i < pads_end_val.size(); i++)
            new_pads_end.push_back(pads_end_val[i] - crops_end_val[i]);
        auto new_conv = register_new_node<opset6::Convolution>(pattern_map.at(data_pattern),
                                                               conv->input_value(1),
                                                               conv->get_strides(),
                                                               new_pads_begin,
                                                               new_pads_end,
                                                               dilations,
                                                               op::PadType::EXPLICIT);

        auto batch_to_space = pattern_map.at(batch_to_space_pattern).get_node_shared_ptr();
        new_conv->set_friendly_name(batch_to_space->get_friendly_name());

        copy_runtime_info(
            {
                pattern_map.at(space_to_batch_pattern).get_node_shared_ptr(),
                pattern_map.at(conv_pattern).get_node_shared_ptr(),
                batch_to_space,
            },
            new_conv);
        replace_node(batch_to_space, new_conv);
        MATCHER_SCOPE_ENABLE(DilatedConvolutionConverter);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(batch_to_space_pattern, matcher_name);
    this->register_matcher(m, callback);
}
