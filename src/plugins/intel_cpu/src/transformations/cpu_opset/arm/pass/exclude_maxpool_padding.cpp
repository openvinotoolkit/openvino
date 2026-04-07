// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "exclude_maxpool_padding.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using ov::pass::pattern::Matcher;
using ov::pass::pattern::wrap_type;

namespace {

bool has_non_zero_padding(const ov::Shape& pads_begin, const ov::Shape& pads_end) {
    return std::any_of(pads_begin.begin(), pads_begin.end(), [](size_t value) {
               return value != 0;
           }) ||
           std::any_of(pads_end.begin(), pads_end.end(), [](size_t value) {
               return value != 0;
           });
}

std::vector<int64_t> extend_pads(const ov::Shape& pads) {
    std::vector<int64_t> extended_pads(pads.size() + 2, 0);
    std::transform(pads.begin(), pads.end(), extended_pads.begin() + 2, [](size_t value) {
        return static_cast<int64_t>(value);
    });
    return extended_pads;
}

bool has_indices_consumer(const std::shared_ptr<const ov::Node>& max_pool) {
    if (ov::is_type_any_of<ov::op::v8::MaxPool, ov::op::v14::MaxPool>(max_pool)) {
        return !max_pool->output(1).get_target_inputs().empty();
    }
    return false;
}

}  // namespace

ov::intel_cpu::ExcludeMaxPoolPadding::ExcludeMaxPoolPadding() {
    const auto max_pool_pattern = wrap_type<ov::op::v1::MaxPool, ov::op::v8::MaxPool, ov::op::v14::MaxPool>();

    const ov::matcher_pass_callback callback = [this](Matcher& m) {
        using ov::op::v0::Constant;
        using ov::op::v1::ConvertLike;
        using ov::op::v12::Pad;

        const auto max_pool = ov::as_type_ptr<ov::op::util::MaxPoolBase>(m.get_match_root());
        if (!max_pool || transformation_callback(max_pool) || max_pool->get_auto_pad() != ov::op::PadType::EXPLICIT ||
            !has_non_zero_padding(max_pool->get_pads_begin(), max_pool->get_pads_end()) || has_indices_consumer(max_pool)) {
            return false;
        }

        const auto pads_begin = extend_pads(max_pool->get_pads_begin());
        const auto pads_end = extend_pads(max_pool->get_pads_end());
        const auto pads_begin_node = Constant::create(ov::element::i64, ov::Shape{pads_begin.size()}, pads_begin);
        const auto pads_end_node = Constant::create(ov::element::i64, ov::Shape{pads_end.size()}, pads_end);
        const auto minus_inf = Constant::create(ov::element::f32,
                                                ov::Shape{},
                                                {-std::numeric_limits<float>::infinity()});
        const auto pad_value = std::make_shared<ConvertLike>(minus_inf, max_pool->input_value(0));
        const auto pad_node = std::make_shared<Pad>(max_pool->input_value(0),
                                                    pads_begin_node,
                                                    pads_end_node,
                                                    pad_value,
                                                    ov::op::PadMode::CONSTANT);

        auto replacement = max_pool->clone_with_new_inputs({pad_node});
        const auto replacement_max_pool = ov::as_type_ptr<ov::op::util::MaxPoolBase>(replacement);
        if (!replacement_max_pool) {
            return false;
        }

        const ov::Shape zero_pads(max_pool->get_pads_begin().size(), 0);
        replacement_max_pool->set_pads_begin(zero_pads);
        replacement_max_pool->set_pads_end(zero_pads);
        replacement_max_pool->set_auto_pad(ov::op::PadType::VALID);
        replacement_max_pool->validate_and_infer_types();

        replacement->set_friendly_name(max_pool->get_friendly_name());
        ov::copy_runtime_info(ov::NodeVector{max_pool},
                              ov::NodeVector{pads_begin_node, pads_end_node, minus_inf, pad_value, pad_node, replacement});
        ov::replace_node(max_pool, replacement);
        max_pool->clear_control_dependencies();

        return true;
    };

    register_matcher(std::make_shared<Matcher>(max_pool_pattern, "ExcludeMaxPoolPadding"), callback);
}