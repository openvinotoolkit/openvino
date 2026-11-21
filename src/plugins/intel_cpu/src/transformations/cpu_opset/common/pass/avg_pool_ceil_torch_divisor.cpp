// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "avg_pool_ceil_torch_divisor.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/op/util/avg_pool_base.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace {

using ov::op::RoundingType;

// Pooling attributes of one spatial axis.
struct Axis {
    int64_t kernel;
    int64_t stride;
    int64_t dilation;
    int64_t pad_begin;
    int64_t pad_end;
};

// Number of CEIL_TORCH windows along an axis: ceil rounding, minus a last window that would start in the
// right padding.
int64_t ceil_torch_outputs(int64_t size, const Axis& axis) {
    const int64_t span = (axis.kernel - 1) * axis.dilation + 1;
    const int64_t room = size + axis.pad_begin + axis.pad_end - span;
    const int64_t outputs = (room + axis.stride - 1) / axis.stride + 1;
    const bool last_starts_in_right_padding = (outputs - 1) * axis.stride >= size + axis.pad_begin;
    return last_starts_in_right_padding ? outputs - 1 : outputs;
}

// Real input positions / padded positions covered by the window starting at `start`.
// Positions beyond `size + pad_end` (the ceil overhang) belong to neither.
float window_ratio(int64_t start, int64_t size, const Axis& axis) {
    int64_t real = 0;
    int64_t padded = 0;
    for (int64_t tap = 0; tap < axis.kernel; tap++) {
        const int64_t position = start + tap * axis.dilation;
        real += (position >= 0 && position < size) ? 1 : 0;
        padded += (position < size + axis.pad_end) ? 1 : 0;
    }
    return static_cast<float>(real) / static_cast<float>(padded);
}

// Per-window ratios along one static axis, and whether any window reaches past the explicit padding.
struct AxisRatios {
    std::vector<float> ratios;
    bool overhang = false;
};

AxisRatios axis_ratios(int64_t size, const Axis& axis) {
    const int64_t span = (axis.kernel - 1) * axis.dilation + 1;
    const int64_t outputs = ceil_torch_outputs(size, axis);
    AxisRatios result;
    for (int64_t index = 0; index < outputs; index++) {
        const int64_t start = index * axis.stride - axis.pad_begin;
        result.ratios.push_back(window_ratio(start, size, axis));
        result.overhang |= start + span > size + axis.pad_end;
    }
    return result;
}

// Flattened outer product: the ratio of a window is the product of its per-axis ratios.
std::vector<float> outer_product(const std::vector<float>& outer, const std::vector<float>& inner) {
    std::vector<float> product(outer.size() * inner.size());
    for (size_t index = 0; index < product.size(); index++) {
        product[index] = outer[index / inner.size()] * inner[index % inner.size()];
    }
    return product;
}

std::vector<Axis> pooling_axes(const ov::op::util::AvgPoolBase& pool, const ov::Strides& dilations) {
    std::vector<Axis> axes;
    for (size_t axis = 0; axis < pool.get_kernel().size(); axis++) {
        axes.push_back(Axis{static_cast<int64_t>(pool.get_kernel()[axis]),
                            static_cast<int64_t>(pool.get_strides()[axis]),
                            static_cast<int64_t>(dilations[axis]),
                            static_cast<int64_t>(pool.get_pads_begin()[axis]),
                            static_cast<int64_t>(pool.get_pads_end()[axis])});
    }
    return axes;
}

// Static spatial shape: the ratio as a [1, 1, out_spatial...] constant, or nothing when no window overhangs
// (then the original pool is already correct on every executor).
std::optional<std::shared_ptr<ov::Node>> static_ratio(const ov::PartialShape& data_shape,
                                                      const std::vector<Axis>& axes,
                                                      const ov::element::Type& type) {
    std::vector<float> ratios{1.0F};
    ov::Shape ratio_shape{1, 1};
    bool overhang = false;
    for (size_t axis = 0; axis < axes.size(); axis++) {
        const auto per_axis = axis_ratios(data_shape[axis + 2].get_length(), axes[axis]);
        ratios = outer_product(ratios, per_axis.ratios);
        ratio_shape.push_back(per_axis.ratios.size());
        overhang |= per_axis.overhang;
    }
    if (!overhang) {
        return std::nullopt;
    }
    return ov::op::v0::Constant::create(type, ratio_shape, ratios);
}

std::vector<int64_t> with_batch_and_channel(const ov::Shape& spatial) {
    std::vector<int64_t> values{0, 0};
    values.insert(values.end(), spatial.begin(), spatial.end());
    return values;
}

// Dynamic spatial shape: compute the ratio in the graph by pooling a [1, 1, spatial] tensor of ones that is
// zero-padded by the explicit pads, with exclude_pad (so only the ceil overhang is left out of the divisor).
std::shared_ptr<ov::Node> dynamic_ratio(const ov::Output<ov::Node>& data,
                                        const ov::Output<ov::Node>& exclude_pad_pool,
                                        const ov::op::util::AvgPoolBase& pool,
                                        const ov::Strides& dilations,
                                        ov::NodeVector& new_nodes) {
    using ov::op::v0::Constant;
    const auto i64 = ov::element::i64;
    const auto rank = static_cast<int64_t>(data.get_partial_shape().rank().get_length());
    const ov::Shape no_pads(pool.get_kernel().size(), 0);

    const auto data_shape = std::make_shared<ov::op::v3::ShapeOf>(data, i64);
    const auto spatial_shape = std::make_shared<ov::op::v8::Slice>(data_shape,
                                                                   Constant::create(i64, {1}, {2}),
                                                                   Constant::create(i64, {1}, {rank}),
                                                                   Constant::create(i64, {1}, {1}));
    const auto ones_shape =
        std::make_shared<ov::op::v0::Concat>(ov::OutputVector{Constant::create(i64, {2}, {1, 1}), spatial_shape}, 0);
    const auto ones =
        std::make_shared<ov::op::v3::Broadcast>(Constant::create(ov::element::f32, {}, {1.0F}), ones_shape);
    const auto pads_begin = with_batch_and_channel(pool.get_pads_begin());
    const auto pads_end = with_batch_and_channel(pool.get_pads_end());
    const auto padded_ones = std::make_shared<ov::op::v1::Pad>(ones,
                                                               Constant::create(i64, {pads_begin.size()}, pads_begin),
                                                               Constant::create(i64, {pads_end.size()}, pads_end),
                                                               Constant::create(ov::element::f32, {}, {0.0F}),
                                                               ov::op::PadMode::CONSTANT);
    const auto ratio_pool = std::make_shared<ov::op::v16::AvgPool>(padded_ones,
                                                                   pool.get_strides(),
                                                                   dilations,
                                                                   no_pads,
                                                                   no_pads,
                                                                   pool.get_kernel(),
                                                                   true,
                                                                   RoundingType::CEIL_TORCH,
                                                                   ov::op::PadType::EXPLICIT);
    // on the padded tensor CEIL_TORCH may keep one extra last window; cut to the real output size
    const auto ratio = std::make_shared<ov::op::v8::Slice>(
        ratio_pool,
        Constant::create(i64, {static_cast<size_t>(rank)}, std::vector<int64_t>(rank, 0)),
        std::make_shared<ov::op::v3::ShapeOf>(exclude_pad_pool, i64),
        Constant::create(i64, {static_cast<size_t>(rank)}, std::vector<int64_t>(rank, 1)));
    // plain Convert: ConvertLike is lowered earlier in the pipeline and has no CPU implementation
    const auto ratio_like_data = std::make_shared<ov::op::v0::Convert>(ratio, pool.get_output_element_type(0));
    new_nodes.insert(new_nodes.end(),
                     {data_shape, spatial_shape, ones_shape, ones, padded_ones, ratio_pool, ratio, ratio_like_data});
    return ratio_like_data;
}

bool has_non_zero_padding(const ov::op::util::AvgPoolBase& pool) {
    const auto non_zero = [](size_t pad) {
        return pad != 0;
    };
    return std::any_of(pool.get_pads_begin().begin(), pool.get_pads_begin().end(), non_zero) ||
           std::any_of(pool.get_pads_end().begin(), pool.get_pads_end().end(), non_zero);
}

// AvgPool with exclude_pad requires every padding to be smaller than the dilated kernel span, so the rewrite
// is only valid when it holds.
// shortcut: pools with padding >= kernel keep the old divisor; handle them if a real model needs it.
bool padding_smaller_than_kernel(const std::vector<Axis>& axes) {
    return std::all_of(axes.begin(), axes.end(), [](const Axis& axis) {
        const int64_t span = (axis.kernel - 1) * axis.dilation + 1;
        return span > axis.pad_begin && span > axis.pad_end;
    });
}

bool spatial_dims_static(const ov::PartialShape& shape) {
    return std::all_of(shape.begin() + 2, shape.end(), [](const ov::Dimension& dim) {
        return dim.is_static();
    });
}

}  // namespace

ov::intel_cpu::AvgPoolCeilTorchDivisor::AvgPoolCeilTorchDivisor() {
    MATCHER_SCOPE(AvgPoolCeilTorchDivisor);
    auto pool_pattern = ov::pass::pattern::wrap_type<ov::op::v14::AvgPool, ov::op::v16::AvgPool>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        const auto pool = ov::as_type_ptr<ov::op::util::AvgPoolBase>(m.get_match_root());
        if (!pool || pool->get_exclude_pad() || pool->get_rounding_type() != RoundingType::CEIL_TORCH ||
            pool->get_auto_pad() != ov::op::PadType::EXPLICIT || !has_non_zero_padding(*pool) ||
            !pool->get_input_element_type(0).is_real() || pool->get_input_partial_shape(0).rank().is_dynamic()) {
            return false;
        }

        const auto pool_v16 = ov::as_type_ptr<ov::op::v16::AvgPool>(pool);
        ov::Strides dilations = pool_v16 ? pool_v16->get_dilations() : ov::Strides{};
        // empty dilations are valid for AvgPool-16 and mean "no dilation"
        dilations.resize(pool->get_kernel().size(), 1);
        if (!padding_smaller_than_kernel(pooling_axes(*pool, dilations))) {
            return false;
        }

        const auto data = pool->input_value(0);
        const auto& data_shape = pool->get_input_partial_shape(0);
        ov::NodeVector new_nodes;

        std::shared_ptr<ov::Node> static_ratio_constant;
        if (spatial_dims_static(data_shape)) {
            const auto ratio =
                static_ratio(data_shape, pooling_axes(*pool, dilations), pool->get_output_element_type(0));
            if (!ratio) {
                return false;  // no window reaches past the explicit padding
            }
            static_ratio_constant = ratio.value();
        }

        // average over real input positions only: every executor agrees on exclude_pad semantics
        const auto exclude_pad_pool = std::make_shared<ov::op::v16::AvgPool>(data,
                                                                             pool->get_strides(),
                                                                             dilations,
                                                                             pool->get_pads_begin(),
                                                                             pool->get_pads_end(),
                                                                             pool->get_kernel(),
                                                                             true,
                                                                             RoundingType::CEIL_TORCH,
                                                                             ov::op::PadType::EXPLICIT);
        const auto ratio = static_ratio_constant ? static_ratio_constant
                                                 : dynamic_ratio(data, exclude_pad_pool, *pool, dilations, new_nodes);
        const auto result = std::make_shared<ov::op::v1::Multiply>(exclude_pad_pool, ratio);

        new_nodes.insert(new_nodes.end(), {exclude_pad_pool, ratio, result});
        result->set_friendly_name(pool->get_friendly_name());
        ov::copy_runtime_info(pool, new_nodes);
        ov::replace_node(pool, result);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(pool_pattern, matcher_name);
    this->register_matcher(m, callback);
}
