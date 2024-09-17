// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/dilated_convolution_converter.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/batch_to_space.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/space_to_batch.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

// Replace the following graph SpaceToBatch -> Convolution(GroupConvolution) -> BatchToSpace with single
// Convolution(GroupConvolution) node
ov::pass::DilatedConvolutionConverter::DilatedConvolutionConverter() {
    MATCHER_SCOPE(DilatedConvolutionConverter);
    auto data_pattern = pattern::any_input();
    auto block_shape_pattern = pattern::wrap_type<ov::op::v0::Constant>();
    auto pads_begin_pattern = pattern::wrap_type<ov::op::v0::Constant>();
    auto pads_end_pattern = pattern::wrap_type<ov::op::v0::Constant>();
    auto space_to_batch_pattern = pattern::wrap_type<ov::op::v1::SpaceToBatch>(
        {data_pattern, block_shape_pattern, pads_begin_pattern, pads_end_pattern});
    auto conv_p = pattern::wrap_type<ov::op::v1::Convolution>({space_to_batch_pattern, pattern::any_input()});
    auto gconv_p = pattern::wrap_type<ov::op::v1::GroupConvolution>({space_to_batch_pattern, pattern::any_input()});
    auto conv_pattern = std::make_shared<pattern::op::Or>(OutputVector{conv_p, gconv_p});
    auto crops_begin_pattern = pattern::wrap_type<ov::op::v0::Constant>();
    auto crops_end_pattern = pattern::wrap_type<ov::op::v0::Constant>();
    auto batch_to_space_pattern = pattern::wrap_type<ov::op::v1::BatchToSpace>(
        {conv_pattern, pattern::any_input(), crops_begin_pattern, crops_end_pattern});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto block_shape =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(block_shape_pattern).get_node_shared_ptr());
        if (!block_shape)
            return false;
        auto pads_begin =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(pads_begin_pattern).get_node_shared_ptr());
        if (!pads_begin)
            return false;
        auto pads_end = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(pads_end_pattern).get_node_shared_ptr());
        if (!pads_end)
            return false;
        auto crops_begin =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(crops_begin_pattern).get_node_shared_ptr());
        if (!crops_begin)
            return false;
        auto crops_end = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(crops_end_pattern).get_node_shared_ptr());
        if (!crops_end)
            return false;

        auto block_shape_val = block_shape->cast_vector<size_t>();

        ov::Strides dilations;
        ov::Strides strides;
        std::shared_ptr<ov::Node> conv_node;
        if (pattern_map.count(conv_p)) {
            conv_node = pattern_map.at(conv_p).get_node_shared_ptr();
            auto conv = ov::as_type_ptr<ov::op::v1::Convolution>(conv_node);
            if (!conv)
                return false;
            dilations = conv->get_dilations();
            strides = conv->get_strides();
        } else if (pattern_map.count(gconv_p)) {
            conv_node = pattern_map.at(gconv_p).get_node_shared_ptr();
            auto conv = ov::as_type_ptr<ov::op::v1::GroupConvolution>(conv_node);
            if (!conv)
                return false;
            dilations = conv->get_dilations();
            strides = conv->get_strides();
        } else {
            return false;
        }

        for (size_t i = 0; i < dilations.size(); i++)
            dilations[i] = block_shape_val[i + 2];
        auto pads_begin_val = pads_begin->cast_vector<std::ptrdiff_t>();
        auto pads_end_val = pads_end->cast_vector<std::ptrdiff_t>();
        if (!(pads_begin_val[0] == 0 && pads_begin_val[1] == 0 && pads_end_val[0] == 0 && pads_end_val[1] == 0))
            return false;
        auto crops_begin_val = crops_begin->cast_vector<std::ptrdiff_t>();
        auto crops_end_val = crops_end->cast_vector<std::ptrdiff_t>();
        std::vector<std::ptrdiff_t> new_pads_begin;
        for (size_t i = 2; i < pads_begin_val.size(); i++) {
            if (pads_begin_val[i] < crops_begin_val[i])
                return false;
            new_pads_begin.push_back(pads_begin_val[i] - crops_begin_val[i]);
        }
        std::vector<std::ptrdiff_t> new_pads_end;
        for (size_t i = 2; i < pads_end_val.size(); i++) {
            if (pads_end_val[i] < crops_end_val[i])
                return false;
            new_pads_end.push_back(pads_end_val[i] - crops_end_val[i]);
        }
        std::shared_ptr<ov::Node> new_conv;
        if (pattern_map.count(gconv_p)) {
            new_conv = register_new_node<ov::op::v1::GroupConvolution>(pattern_map.at(data_pattern),
                                                                       conv_node->input_value(1),
                                                                       strides,
                                                                       new_pads_begin,
                                                                       new_pads_end,
                                                                       dilations,
                                                                       op::PadType::EXPLICIT);
        } else {
            new_conv = register_new_node<ov::op::v1::Convolution>(pattern_map.at(data_pattern),
                                                                  conv_node->input_value(1),
                                                                  strides,
                                                                  new_pads_begin,
                                                                  new_pads_end,
                                                                  dilations,
                                                                  op::PadType::EXPLICIT);
        }

        auto batch_to_space = pattern_map.at(batch_to_space_pattern).get_node_shared_ptr();
        new_conv->set_friendly_name(batch_to_space->get_friendly_name());

        copy_runtime_info(
            {
                pattern_map.at(space_to_batch_pattern).get_node_shared_ptr(),
                conv_node,
                batch_to_space,
            },
            new_conv);
        replace_node(batch_to_space, new_conv);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(batch_to_space_pattern, matcher_name);
    this->register_matcher(m, callback);
}
