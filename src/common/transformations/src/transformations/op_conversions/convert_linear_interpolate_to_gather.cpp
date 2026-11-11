// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_linear_interpolate_to_gather.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "itt.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/pass/node_registry.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/op_conversions/interpolate_gather_utils.hpp"

using ov::pass::pattern::Matcher;

namespace ov {
namespace pass {
namespace {

using InterpolateBase = ov::op::util::InterpolateBase;
using namespace ov::pass::interpolate_gather;

bool is_linear_static_interpolate(const ov::Output<ov::Node>& output) {
    const auto node = std::dynamic_pointer_cast<InterpolateBase>(output.get_node_shared_ptr());
    if (!node) {
        return false;
    }
    const auto& attrs = node->get_attrs();
    const auto& output_pshape = output.get_partial_shape();
    const auto& input_pshape = node->get_input_partial_shape(0);
    const auto elem_type = node->get_input_element_type(0);
    const bool supported_type =
        elem_type == ov::element::f32 || elem_type == ov::element::f16 || elem_type == ov::element::bf16;
    return attrs.mode == InterpolateBase::InterpolateMode::LINEAR_ONNX && !attrs.antialias &&
           !has_non_zero_padding(attrs.pads_begin) && !has_non_zero_padding(attrs.pads_end) && supported_type &&
           output_pshape.is_static() && output_pshape.rank().get_length() >= 2 && input_pshape.is_static() &&
           input_pshape.rank() == output_pshape.rank();
}

}  // namespace

ConvertLinearInterpolateToGather::ConvertLinearInterpolateToGather() {
    MATCHER_SCOPE(ConvertLinearInterpolateToGather);

    const auto interpolate_pattern =
        ov::pass::pattern::wrap_type<ov::op::v11::Interpolate, ov::op::v4::Interpolate>(is_linear_static_interpolate);

    const matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto node_handle = m.get_pattern_value_map().at(interpolate_pattern).get_node_shared_ptr();
        const auto node = std::dynamic_pointer_cast<InterpolateBase>(node_handle);
        OPENVINO_ASSERT(node, "ConvertLinearInterpolateToGather: matched node is expected to be Interpolate-4/11");

        if (transformation_callback(node_handle)) {
            return false;
        }

        const auto& attrs = node->get_attrs();
        const auto input_shape = node->get_input_partial_shape(0).to_shape();
        const auto output_shape = node->get_output_partial_shape(0).to_shape();
        const size_t rank = input_shape.size();
        const auto is_scales_mode = attrs.shape_calculation_mode == InterpolateBase::ShapeCalcMode::SCALES;

        // v11 has a single `scales_or_sizes` input (port 1) and an optional `axes` input (port 2); v4 keeps the
        // legacy layout: `output_shape` (port 1), `scales` (port 2) and an optional `axes` input (port 3).
        const auto is_v11 = std::dynamic_pointer_cast<ov::op::v11::Interpolate>(node_handle) != nullptr;
        OPENVINO_ASSERT(is_v11 || std::dynamic_pointer_cast<ov::op::v4::Interpolate>(node_handle),
                        "ConvertLinearInterpolateToGather: matched node is expected to be Interpolate-4/11");
        const size_t values_port = is_v11 ? 1 : (is_scales_mode ? 2 : 1);
        const size_t axes_port = is_v11 ? 2 : 3;

        std::vector<int64_t> axes;
        if (node->get_input_size() > axes_port) {
            const auto axes_constant = ov::util::get_constant_from_source(node->input_value(axes_port));
            if (!axes_constant) {
                return false;
            }
            axes = axes_constant->cast_vector<int64_t>();
        }
        if (axes.empty()) {
            axes.resize(rank);
            std::iota(axes.begin(), axes.end(), int64_t{0});
        }

        const auto scales_or_sizes = ov::util::get_constant_from_source(node->input_value(values_port));
        if (!scales_or_sizes) {
            return false;
        }
        const auto scales = is_scales_mode ? scales_or_sizes->cast_vector<float>() : std::vector<float>{};
        const auto sizes = is_scales_mode ? std::vector<int64_t>{} : scales_or_sizes->cast_vector<int64_t>();
        if ((is_scales_mode ? scales.size() : sizes.size()) != axes.size()) {
            return false;
        }

        ov::pass::NodeRegistry registry;
        const auto original_type = node->get_input_element_type(0);
        ov::Output<ov::Node> result = node->input_value(0);
        if (original_type != ov::element::f32) {
            result = registry.make<ov::op::v0::Convert>(result, ov::element::f32);
        }
        bool rewritten = false;

        for (size_t i = 0; i < axes.size(); ++i) {
            const auto axis = axes[i] < 0 ? axes[i] + static_cast<int64_t>(rank) : axes[i];
            OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < rank,
                            "ConvertLinearInterpolateToGather: axis is out of range");
            const auto in = input_shape[axis];
            const auto out = output_shape[axis];
            if (in == out) {
                continue;
            }
            const auto scale = is_scales_mode ? scales[i] : static_cast<float>(out) / static_cast<float>(in);

            std::vector<int64_t> lower_index(out);
            std::vector<int64_t> upper_index(out);
            std::vector<float> lower_weight(out);
            std::vector<float> upper_weight(out);
            for (int64_t out_idx = 0; out_idx < static_cast<int64_t>(out); ++out_idx) {
                auto coord = source_coordinate(attrs.coordinate_transformation_mode,
                                               static_cast<float>(out_idx),
                                               scale,
                                               static_cast<int64_t>(out),
                                               static_cast<int64_t>(in));
                coord = std::max(0.0f, std::min(coord, static_cast<float>(in - 1)));
                const auto lower = std::min(static_cast<int64_t>(coord), static_cast<int64_t>(in) - 1);
                const auto upper = std::min(lower + 1, static_cast<int64_t>(in) - 1);
                auto d1 = std::fabs(coord - static_cast<float>(lower));
                auto d2 = std::fabs(coord - static_cast<float>(upper));
                if (lower == upper) {
                    d1 = 0.5f;
                    d2 = 0.5f;
                }
                lower_index[out_idx] = lower;
                upper_index[out_idx] = upper;
                lower_weight[out_idx] = d2;
                upper_weight[out_idx] = d1;
            }

            std::vector<size_t> weight_shape(rank, 1);
            weight_shape[axis] = out;
            const auto axis_constant = registry.make<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, axis);
            const auto lower_index_constant =
                registry.make<ov::op::v0::Constant>(ov::element::i64, ov::Shape{out}, lower_index);
            const auto upper_index_constant =
                registry.make<ov::op::v0::Constant>(ov::element::i64, ov::Shape{out}, upper_index);
            const auto lower_weight_constant =
                registry.make<ov::op::v0::Constant>(ov::element::f32, weight_shape, lower_weight);
            const auto upper_weight_constant =
                registry.make<ov::op::v0::Constant>(ov::element::f32, weight_shape, upper_weight);

            const auto lower = registry.make<ov::op::v8::Gather>(result, lower_index_constant, axis_constant);
            const auto upper = registry.make<ov::op::v8::Gather>(result, upper_index_constant, axis_constant);
            const auto lower_scaled = registry.make<ov::op::v1::Multiply>(lower, lower_weight_constant);
            const auto upper_scaled = registry.make<ov::op::v1::Multiply>(upper, upper_weight_constant);
            result = registry.make<ov::op::v1::Add>(lower_scaled, upper_scaled);
            rewritten = true;
        }

        if (!rewritten) {
            return false;
        }
        if (original_type != ov::element::f32) {
            result = registry.make<ov::op::v0::Convert>(result, original_type);
        }

        result.get_node_shared_ptr()->set_friendly_name(node->get_friendly_name());
        copy_runtime_info(node_handle, registry.get());
        replace_node(node_handle, result.get_node_shared_ptr());
        return true;
    };

    register_matcher(std::make_shared<Matcher>(interpolate_pattern, matcher_name), callback);
}

}  // namespace pass
}  // namespace ov
