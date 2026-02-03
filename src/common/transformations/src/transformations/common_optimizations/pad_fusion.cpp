// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/pad_fusion.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/util/pad_base.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace op_util = ov::op::util;

namespace ov::pass {

namespace {
template <typename T>
bool can_be_fused(const std::shared_ptr<op_util::PadBase>& pad,
                  const std::shared_ptr<T>& node,
                  const std::shared_ptr<Node>& pad_value_node,
                  const std::shared_ptr<v0::Constant>& pads_begin,
                  const std::shared_ptr<v0::Constant>& pads_end) {
    if (!pad || pad->get_pad_mode() != ov::op::PadMode::CONSTANT)
        return false;
    if (!node)
        return false;

    auto pad_value_const = ov::util::get_constant_from_source(pad_value_node);
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

    auto pads_begin_val = pads_begin->cast_vector<int64_t>();
    auto pads_end_val = pads_end->cast_vector<int64_t>();
    for (size_t i = 0; i < 2; i++) {
        if (pads_begin_val[i] != 0 || pads_end_val[i] != 0)
            return false;
    }
    auto pred = [](int64_t a) {
        return a < 0;
    };
    if (std::any_of(pads_begin_val.begin(), pads_begin_val.end(), pred) ||
        std::any_of(pads_end_val.begin(), pads_end_val.end(), pred)) {
        return false;
    }

    return true;
}

template <typename T>
std::tuple<Shape, Shape> new_pooling_pad_values(const std::shared_ptr<v0::Constant>& pads_begin,
                                                const std::shared_ptr<v0::Constant>& pads_end,
                                                const std::shared_ptr<T>& node) {
    auto node_pads_begin = node->get_pads_begin();
    auto node_pads_end = node->get_pads_end();
    auto pads_begin_val = pads_begin->cast_vector<size_t>();
    auto pads_end_val = pads_end->cast_vector<size_t>();

    std::transform(node_pads_begin.begin(),
                   node_pads_begin.end(),
                   pads_begin_val.begin() + 2,
                   node_pads_begin.begin(),
                   [](size_t a, size_t b) -> size_t {
                       return a + b;
                   });
    std::transform(node_pads_end.begin(),
                   node_pads_end.end(),
                   pads_end_val.begin() + 2,
                   node_pads_end.begin(),
                   [](size_t a, size_t b) -> size_t {
                       return a + b;
                   });

    return std::make_tuple(node_pads_begin, node_pads_end);
}

template <typename T>
std::tuple<CoordinateDiff, CoordinateDiff> new_conv_pad_values(const std::shared_ptr<v0::Constant>& pads_begin,
                                                               const std::shared_ptr<v0::Constant>& pads_end,
                                                               const std::shared_ptr<T>& node) {
    auto node_pads_begin = node->get_pads_begin();
    auto node_pads_end = node->get_pads_end();
    auto pads_begin_val = pads_begin->cast_vector<size_t>();
    auto pads_end_val = pads_end->cast_vector<size_t>();

    std::transform(node_pads_begin.begin(),
                   node_pads_begin.end(),
                   pads_begin_val.begin() + 2,
                   node_pads_begin.begin(),
                   [](std::ptrdiff_t a, size_t b) -> std::ptrdiff_t {
                       return a + b;
                   });
    std::transform(node_pads_end.begin(),
                   node_pads_end.end(),
                   pads_end_val.begin() + 2,
                   node_pads_end.begin(),
                   [](std::ptrdiff_t a, size_t b) -> std::ptrdiff_t {
                       return a + b;
                   });

    return std::make_tuple(node_pads_begin, node_pads_end);
}
}  // namespace

PadFusionAvgPool::PadFusionAvgPool() {
    MATCHER_SCOPE(PadFusionAvgPool);
    auto data_pattern = pattern::any_input();
    auto pads_begin_pattern = pattern::wrap_type<v0::Constant>();
    auto pads_end_pattern = pattern::wrap_type<v0::Constant>();
    auto pad_value_pattern = pattern::any_input();
    auto pad_node_pattern =
        pattern::wrap_type<op_util::PadBase>({data_pattern, pads_begin_pattern, pads_end_pattern, pad_value_pattern},
                                             pattern::consumers_count(1));
    auto avg_pool_pattern = pattern::wrap_type<v1::AvgPool>({pad_node_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_value_map();
        auto data = pattern_map[data_pattern];
        auto pad = ov::as_type_ptr<op_util::PadBase>(pattern_map[pad_node_pattern].get_node_shared_ptr());
        auto pad_value = pattern_map[pad_value_pattern].get_node_shared_ptr();
        auto pads_begin = ov::as_type_ptr<v0::Constant>(pattern_map[pads_begin_pattern].get_node_shared_ptr());
        auto pads_end = ov::as_type_ptr<v0::Constant>(pattern_map[pads_end_pattern].get_node_shared_ptr());
        auto avg_pool = ov::as_type_ptr<v1::AvgPool>(pattern_map[avg_pool_pattern].get_node_shared_ptr());
        if (!can_be_fused(pad, avg_pool, pad_value, pads_begin, pads_end))
            return false;

        std::shared_ptr<v1::AvgPool> new_avg_pool;
        if (avg_pool->get_exclude_pad()) {
            const auto& avg_pads_begin = avg_pool->get_pads_begin();
            const auto& avg_pads_end = avg_pool->get_pads_end();
            bool avg_pads_begin_are_zeros =
                std::all_of(avg_pads_begin.begin(), avg_pads_begin.end(), [](size_t p) -> bool {
                    return p == 0;
                });
            bool avg_pads_end_are_zeros = std::all_of(avg_pads_end.begin(), avg_pads_end.end(), [](size_t p) -> bool {
                return p == 0;
            });
            if (!avg_pads_begin_are_zeros || !avg_pads_end_are_zeros)
                return false;
            auto pads_begin_val = pads_begin->cast_vector<size_t>();
            auto pads_end_val = pads_end->cast_vector<size_t>();
            new_avg_pool = std::make_shared<v1::AvgPool>(data,
                                                         avg_pool->get_strides(),
                                                         Shape{pads_begin_val.begin() + 2, pads_begin_val.end()},
                                                         Shape{pads_end_val.begin() + 2, pads_end_val.end()},
                                                         avg_pool->get_kernel(),
                                                         false,
                                                         avg_pool->get_rounding_type(),
                                                         ov::op::PadType::EXPLICIT);
        } else {
            Shape new_pads_begin, new_pads_end;
            std::tie(new_pads_begin, new_pads_end) = new_pooling_pad_values(pads_begin, pads_end, avg_pool);
            new_avg_pool = std::make_shared<v1::AvgPool>(data,
                                                         avg_pool->get_strides(),
                                                         new_pads_begin,
                                                         new_pads_end,
                                                         avg_pool->get_kernel(),
                                                         false,
                                                         avg_pool->get_rounding_type(),
                                                         ov::op::PadType::EXPLICIT);
        }
        new_avg_pool->set_friendly_name(avg_pool->get_friendly_name());

        copy_runtime_info({pad, avg_pool}, new_avg_pool);
        replace_node(avg_pool, new_avg_pool);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(avg_pool_pattern, matcher_name);
    this->register_matcher(m, callback);
}

PadFusionConvolution::PadFusionConvolution() {
    MATCHER_SCOPE(PadFusionConvolution);
    auto data_pattern = pattern::any_input();
    auto filter_pattern = pattern::any_input();
    auto pads_begin_pattern = pattern::wrap_type<v0::Constant>();
    auto pads_end_pattern = pattern::wrap_type<v0::Constant>();
    auto pad_value_pattern = pattern::any_input();
    auto pad_node_pattern =
        pattern::wrap_type<op_util::PadBase>({data_pattern, pads_begin_pattern, pads_end_pattern, pad_value_pattern},
                                             pattern::consumers_count(1));
    auto conv_pattern = pattern::wrap_type<v1::Convolution>({pad_node_pattern, filter_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_value_map();
        auto data = pattern_map[data_pattern];
        auto filter = pattern_map[filter_pattern];
        auto pad = ov::as_type_ptr<op_util::PadBase>(pattern_map[pad_node_pattern].get_node_shared_ptr());
        auto pad_value = pattern_map[pad_value_pattern].get_node_shared_ptr();
        auto pads_begin = ov::as_type_ptr<v0::Constant>(pattern_map[pads_begin_pattern].get_node_shared_ptr());
        auto pads_end = ov::as_type_ptr<v0::Constant>(pattern_map[pads_end_pattern].get_node_shared_ptr());
        auto conv = ov::as_type_ptr<v1::Convolution>(pattern_map[conv_pattern].get_node_shared_ptr());
        if (!can_be_fused(pad, conv, pad_value, pads_begin, pads_end))
            return false;

        CoordinateDiff new_pads_begin, new_pads_end;
        std::tie(new_pads_begin, new_pads_end) = new_conv_pad_values(pads_begin, pads_end, conv);
        auto new_conv = std::make_shared<v1::Convolution>(data,
                                                          filter,
                                                          conv->get_strides(),
                                                          new_pads_begin,
                                                          new_pads_end,
                                                          conv->get_dilations(),
                                                          ov::op::PadType::EXPLICIT);
        new_conv->set_friendly_name(conv->get_friendly_name());

        copy_runtime_info({pad, conv}, new_conv);
        replace_node(conv, new_conv);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(conv_pattern, matcher_name);
    this->register_matcher(m, callback);
}

PadFusionConvolutionBackpropData::PadFusionConvolutionBackpropData() {
    MATCHER_SCOPE(PadFusionConvolutionBackpropData);
    auto data_pattern = pattern::any_input();
    auto filter_pattern = pattern::any_input();
    auto pads_begin_pattern = pattern::wrap_type<v0::Constant>();
    auto pads_end_pattern = pattern::wrap_type<v0::Constant>();
    auto pad_value_pattern = pattern::any_input();
    auto pad_node_pattern =
        pattern::wrap_type<op_util::PadBase>({data_pattern, pads_begin_pattern, pads_end_pattern, pad_value_pattern},
                                             pattern::consumers_count(1));
    auto conv_pattern = pattern::wrap_type<v1::ConvolutionBackpropData>({pad_node_pattern, filter_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_value_map();
        auto data = pattern_map[data_pattern];
        auto filter = pattern_map[filter_pattern];
        auto pad = ov::as_type_ptr<op_util::PadBase>(pattern_map[pad_node_pattern].get_node_shared_ptr());
        auto pad_value = pattern_map[pad_value_pattern].get_node_shared_ptr();
        auto pads_begin = ov::as_type_ptr<v0::Constant>(pattern_map[pads_begin_pattern].get_node_shared_ptr());
        auto pads_end = ov::as_type_ptr<v0::Constant>(pattern_map[pads_end_pattern].get_node_shared_ptr());
        auto conv = ov::as_type_ptr<v1::ConvolutionBackpropData>(pattern_map[conv_pattern].get_node_shared_ptr());
        if (!can_be_fused(pad, conv, pad_value, pads_begin, pads_end))
            return false;

        auto conv_pads_begin = conv->get_pads_begin();
        auto conv_pads_end = conv->get_pads_end();
        auto pads_begin_val = pads_begin->cast_vector<std::ptrdiff_t>();
        auto pads_end_val = pads_end->cast_vector<std::ptrdiff_t>();
        for (size_t i = 0; i < conv_pads_begin.size(); i++) {
            if (conv_pads_begin[i] < pads_begin_val[i + 2] || conv_pads_end[i] < pads_end_val[i + 2])
                return false;
            conv_pads_begin[i] -= pads_begin_val[i + 2];
            conv_pads_end[i] -= pads_end_val[i + 2];
        }

        auto new_conv = std::make_shared<v1::ConvolutionBackpropData>(data,
                                                                      filter,
                                                                      conv->get_strides(),
                                                                      conv_pads_begin,
                                                                      conv_pads_end,
                                                                      conv->get_dilations(),
                                                                      ov::op::PadType::EXPLICIT,
                                                                      conv->get_output_padding());
        new_conv->set_friendly_name(conv->get_friendly_name());

        copy_runtime_info({pad, conv}, new_conv);
        replace_node(conv, new_conv);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(conv_pattern, matcher_name);
    this->register_matcher(m, callback);
}

PadFusionGroupConvolution::PadFusionGroupConvolution() {
    MATCHER_SCOPE(PadFusionGroupConvolution);
    auto data_pattern = pattern::any_input();
    auto filter_pattern = pattern::any_input();
    auto pads_begin_pattern = pattern::wrap_type<v0::Constant>();
    auto pads_end_pattern = pattern::wrap_type<v0::Constant>();
    auto pad_value_pattern = pattern::any_input();
    auto pad_node_pattern =
        pattern::wrap_type<op_util::PadBase>({data_pattern, pads_begin_pattern, pads_end_pattern, pad_value_pattern},
                                             pattern::consumers_count(1));
    auto conv_pattern = pattern::wrap_type<v1::GroupConvolution>({pad_node_pattern, filter_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_value_map();
        auto data = pattern_map[data_pattern];
        auto filter = pattern_map[filter_pattern];
        auto pad = ov::as_type_ptr<op_util::PadBase>(pattern_map[pad_node_pattern].get_node_shared_ptr());
        auto pad_value = pattern_map[pad_value_pattern].get_node_shared_ptr();
        auto pads_begin = ov::as_type_ptr<v0::Constant>(pattern_map[pads_begin_pattern].get_node_shared_ptr());
        auto pads_end = ov::as_type_ptr<v0::Constant>(pattern_map[pads_end_pattern].get_node_shared_ptr());
        auto conv = ov::as_type_ptr<v1::GroupConvolution>(pattern_map[conv_pattern].get_node_shared_ptr());
        if (!can_be_fused(pad, conv, pad_value, pads_begin, pads_end))
            return false;

        CoordinateDiff new_pads_begin, new_pads_end;
        std::tie(new_pads_begin, new_pads_end) = new_conv_pad_values(pads_begin, pads_end, conv);
        auto new_conv = std::make_shared<v1::GroupConvolution>(data,
                                                               filter,
                                                               conv->get_strides(),
                                                               new_pads_begin,
                                                               new_pads_end,
                                                               conv->get_dilations(),
                                                               ov::op::PadType::EXPLICIT);
        new_conv->set_friendly_name(conv->get_friendly_name());

        copy_runtime_info({pad, conv}, new_conv);
        replace_node(conv, new_conv);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(conv_pattern, matcher_name);
    this->register_matcher(m, callback);
}

PadFusionGroupConvolutionBackpropData::PadFusionGroupConvolutionBackpropData() {
    MATCHER_SCOPE(PadFusionGroupConvolutionBackpropData);
    auto data_pattern = pattern::any_input();
    auto filter_pattern = pattern::any_input();
    auto pads_begin_pattern = pattern::wrap_type<v0::Constant>();
    auto pads_end_pattern = pattern::wrap_type<v0::Constant>();
    auto pad_value_pattern = pattern::any_input();
    auto pad_node_pattern =
        pattern::wrap_type<op_util::PadBase>({data_pattern, pads_begin_pattern, pads_end_pattern, pad_value_pattern},
                                             pattern::consumers_count(1));
    auto conv_pattern = pattern::wrap_type<v1::GroupConvolutionBackpropData>({pad_node_pattern, filter_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_value_map();
        auto data = pattern_map[data_pattern];
        auto filter = pattern_map[filter_pattern];
        auto pad = ov::as_type_ptr<op_util::PadBase>(pattern_map[pad_node_pattern].get_node_shared_ptr());
        auto pad_value = pattern_map[pad_value_pattern].get_node_shared_ptr();
        auto pads_begin = ov::as_type_ptr<v0::Constant>(pattern_map[pads_begin_pattern].get_node_shared_ptr());
        auto pads_end = ov::as_type_ptr<v0::Constant>(pattern_map[pads_end_pattern].get_node_shared_ptr());
        auto conv = ov::as_type_ptr<v1::GroupConvolutionBackpropData>(pattern_map[conv_pattern].get_node_shared_ptr());
        if (!can_be_fused(pad, conv, pad_value, pads_begin, pads_end))
            return false;

        auto conv_pads_begin = conv->get_pads_begin();
        auto conv_pads_end = conv->get_pads_end();
        auto pads_begin_val = pads_begin->cast_vector<std::ptrdiff_t>();
        auto pads_end_val = pads_end->cast_vector<std::ptrdiff_t>();
        for (size_t i = 0; i < conv_pads_begin.size(); i++) {
            if (conv_pads_begin[i] < pads_begin_val[i + 2] || conv_pads_end[i] < pads_end_val[i + 2])
                return false;
            conv_pads_begin[i] -= pads_begin_val[i + 2];
            conv_pads_end[i] -= pads_end_val[i + 2];
        }

        auto new_conv = std::make_shared<v1::GroupConvolutionBackpropData>(data,
                                                                           filter,
                                                                           conv->get_strides(),
                                                                           conv_pads_begin,
                                                                           conv_pads_end,
                                                                           conv->get_dilations(),
                                                                           ov::op::PadType::EXPLICIT,
                                                                           conv->get_output_padding());
        new_conv->set_friendly_name(conv->get_friendly_name());

        copy_runtime_info({pad, conv}, new_conv);
        replace_node(conv, new_conv);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(conv_pattern, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::pass
