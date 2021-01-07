// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/pad_fusion.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset5.hpp>

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(pass::PadFusion, "PadFusion", 0);

template <typename T>
static bool can_be_fused(const std::shared_ptr<opset5::Pad>& pad, const std::shared_ptr<T>& node,
                         const std::shared_ptr<opset5::Constant>& pad_value_const,
                         const std::shared_ptr<opset5::Constant>& pads_begin, const std::shared_ptr<opset5::Constant>& pads_end) {
    if (!pad)
        return false;
    if (!node)
        return false;
    if (pad->get_pad_mode() != op::PadMode::CONSTANT)
        return false;

    if (!pad_value_const)
        return false;
    auto pad_value = pad_value_const->cast_vector<float>()[0];
    if (pad_value != 0.0f)
        return false;

    if (!pads_begin || !is_vector(pads_begin->get_shape()))
        return false;
    if (!pads_end || !is_vector(pads_end->get_shape()))
        return false;
    if (node->get_pads_begin().size() != shape_size(pads_begin->get_shape()) - 2)
        return false;
    if (node->get_pads_end().size() != shape_size(pads_end->get_shape()) - 2)
        return false;

    return true;
}

template <typename T>
static std::tuple<Shape, Shape> new_pad_values(const std::shared_ptr<opset5::Constant>& pads_begin,
                                               const std::shared_ptr<opset5::Constant>& pads_end,
                                               const std::shared_ptr<T>& node) {
    auto node_pads_begin = node->get_pads_begin();
    auto node_pads_end = node->get_pads_end();
    auto pads_begin_val = pads_begin->cast_vector<size_t>();
    auto pads_end_val = pads_end->cast_vector<size_t>();

    std::transform(node_pads_begin.begin(), node_pads_begin.end(),
                   pads_begin_val.begin() + 2, node_pads_begin.begin(),
                   [] (size_t a, size_t b) -> size_t { return a + b; });
    std::transform(node_pads_end.begin(), node_pads_end.end(),
                   pads_end_val.begin() + 2, node_pads_end.begin(),
                   [] (size_t a, size_t b) -> size_t { return a + b; });

    return std::make_tuple(node_pads_begin, node_pads_end);
}

NGRAPH_RTTI_DEFINITION(pass::PadFusionAvgPool, "PadFusionAvgPool", 0);

pass::PadFusionAvgPool::PadFusionAvgPool() {
    auto data_pattern = pattern::any_input();
    auto pads_begin_pattern = pattern::wrap_type<opset5::Constant>();
    auto pads_end_pattern = pattern::wrap_type<opset5::Constant>();
    auto pad_value_pattern = pattern::wrap_type<opset5::Constant>();
    auto pad_node_pattern = pattern::wrap_type<opset5::Pad>({data_pattern, pads_begin_pattern, pads_end_pattern, pad_value_pattern});
    auto avg_pool_pattern = pattern::wrap_type<opset5::AvgPool>({pad_node_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_value_map();
        auto data = pattern_map[data_pattern];
        auto pad = std::dynamic_pointer_cast<opset5::Pad>(pattern_map[pad_node_pattern].get_node_shared_ptr());
        auto pad_value_const = std::dynamic_pointer_cast<opset5::Constant>(pattern_map[pad_value_pattern].get_node_shared_ptr());
        auto pads_begin = std::dynamic_pointer_cast<opset5::Constant>(pattern_map[pads_begin_pattern].get_node_shared_ptr());
        auto pads_end = std::dynamic_pointer_cast<opset5::Constant>(pattern_map[pads_end_pattern].get_node_shared_ptr());
        auto avg_pool = std::dynamic_pointer_cast<opset5::AvgPool>(pattern_map[avg_pool_pattern].get_node_shared_ptr());
        if (!can_be_fused(pad, avg_pool, pad_value_const, pads_begin, pads_end))
            return false;

        Shape new_pads_begin, new_pads_end;
        std::tie(new_pads_begin, new_pads_end) = new_pad_values(pads_begin, pads_end, avg_pool);
        auto new_avg_pool = std::make_shared<opset5::AvgPool>(data, avg_pool->get_strides(),
                                                              new_pads_begin, new_pads_end,
                                                              avg_pool->get_kernel(), avg_pool->get_exclude_pad(),
                                                              avg_pool->get_rounding_type());
        new_avg_pool->set_friendly_name(avg_pool->get_friendly_name());

        copy_runtime_info(avg_pool, new_avg_pool);
        replace_node(avg_pool, new_avg_pool);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(avg_pool_pattern, "PadFusionAvgPool");
    this->register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(pass::PadFusionMaxPool, "PadFusionMaxPool", 0);

pass::PadFusionMaxPool::PadFusionMaxPool() {
    auto data_pattern = pattern::any_input();
    auto pads_begin_pattern = pattern::wrap_type<opset5::Constant>();
    auto pads_end_pattern = pattern::wrap_type<opset5::Constant>();
    auto pad_value_pattern = pattern::wrap_type<opset5::Constant>();
    auto pad_node_pattern = pattern::wrap_type<opset5::Pad>({data_pattern, pads_begin_pattern, pads_end_pattern, pad_value_pattern});
    auto max_pool_pattern = pattern::wrap_type<opset5::MaxPool>({pad_node_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_value_map();
        auto data = pattern_map[data_pattern];
        auto pad = std::dynamic_pointer_cast<opset5::Pad>(pattern_map[pad_node_pattern].get_node_shared_ptr());
        auto pad_value_const = std::dynamic_pointer_cast<opset5::Constant>(pattern_map[pad_value_pattern].get_node_shared_ptr());
        auto pads_begin = std::dynamic_pointer_cast<opset5::Constant>(pattern_map[pads_begin_pattern].get_node_shared_ptr());
        auto pads_end = std::dynamic_pointer_cast<opset5::Constant>(pattern_map[pads_end_pattern].get_node_shared_ptr());
        auto max_pool = std::dynamic_pointer_cast<opset5::MaxPool>(pattern_map[max_pool_pattern].get_node_shared_ptr());
        if (!can_be_fused(pad, max_pool, pad_value_const, pads_begin, pads_end))
            return false;

        Shape new_pads_begin, new_pads_end;
        std::tie(new_pads_begin, new_pads_end) = new_pad_values(pads_begin, pads_end, max_pool);
        auto new_max_pool = std::make_shared<opset5::MaxPool>(data, max_pool->get_strides(),
                                                              new_pads_begin, new_pads_end,
                                                              max_pool->get_kernel(), max_pool->get_rounding_type(),
                                                              max_pool->get_auto_pad());
        new_max_pool->set_friendly_name(max_pool->get_friendly_name());

        copy_runtime_info(max_pool, new_max_pool);
        replace_node(max_pool, new_max_pool);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(max_pool_pattern, "PadFusionMaxPool");
    this->register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(pass::PadFusionConvolution, "PadFusionConvolution", 0);

static std::tuple<CoordinateDiff, CoordinateDiff> new_pad_values(const std::shared_ptr<opset5::Constant>& pads_begin,
                                                                 const std::shared_ptr<opset5::Constant>& pads_end,
                                                                 const std::shared_ptr<opset5::Convolution>& node) {
    auto node_pads_begin = node->get_pads_begin();
    auto node_pads_end = node->get_pads_end();
    auto pads_begin_val = pads_begin->cast_vector<size_t>();
    auto pads_end_val = pads_end->cast_vector<size_t>();

    std::transform(node_pads_begin.begin(), node_pads_begin.end(),
                   pads_begin_val.begin() + 2, node_pads_begin.begin(),
                   [] (std::ptrdiff_t a, size_t b) -> std::ptrdiff_t { return a + b; });
    std::transform(node_pads_end.begin(), node_pads_end.end(),
                   pads_end_val.begin() + 2, node_pads_end.begin(),
                   [] (std::ptrdiff_t a, size_t b) -> std::ptrdiff_t { return a + b; });

    return std::make_tuple(node_pads_begin, node_pads_end);
}



pass::PadFusionConvolution::PadFusionConvolution() {
    auto data_pattern = pattern::any_input();
    auto filter_pattern = pattern::any_input();
    auto pads_begin_pattern = pattern::wrap_type<opset5::Constant>();
    auto pads_end_pattern = pattern::wrap_type<opset5::Constant>();
    auto pad_value_pattern = pattern::wrap_type<opset5::Constant>();
    auto pad_node_pattern = pattern::wrap_type<opset5::Pad>({data_pattern, pads_begin_pattern, pads_end_pattern, pad_value_pattern});
    auto conv_pattern = pattern::wrap_type<opset5::Convolution>({pad_node_pattern, filter_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_value_map();
        auto data = pattern_map[data_pattern];
        auto filter = pattern_map[filter_pattern];
        auto pad = std::dynamic_pointer_cast<opset5::Pad>(pattern_map[pad_node_pattern].get_node_shared_ptr());
        auto pad_value_const = std::dynamic_pointer_cast<opset5::Constant>(pattern_map[pad_value_pattern].get_node_shared_ptr());
        auto pads_begin = std::dynamic_pointer_cast<opset5::Constant>(pattern_map[pads_begin_pattern].get_node_shared_ptr());
        auto pads_end = std::dynamic_pointer_cast<opset5::Constant>(pattern_map[pads_end_pattern].get_node_shared_ptr());
        auto conv = std::dynamic_pointer_cast<opset5::Convolution>(pattern_map[conv_pattern].get_node_shared_ptr());
        if (!can_be_fused(pad, conv, pad_value_const, pads_begin, pads_end))
            return false;

        CoordinateDiff new_pads_begin, new_pads_end;
        std::tie(new_pads_begin, new_pads_end) = new_pad_values(pads_begin, pads_end, conv);
        auto new_conv = std::make_shared<opset5::Convolution>(data, filter, conv->get_strides(),
                                                              new_pads_begin, new_pads_end,
                                                              conv->get_dilations(), conv->get_auto_pad());
        new_conv->set_friendly_name(conv->get_friendly_name());

        copy_runtime_info(conv, new_conv);
        replace_node(conv, new_conv);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(conv_pattern, "PadFusionConvolution");
    this->register_matcher(m, callback);
}
