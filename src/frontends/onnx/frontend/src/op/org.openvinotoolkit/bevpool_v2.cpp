// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include <string>

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/bevpool_v2.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/transpose.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace org_openvinotoolkit {
namespace opset_1 {

namespace {
uint32_t to_u32_checked(const ov::frontend::onnx::Node& node, const std::string& name, int64_t value) {
    CHECK_VALID_NODE(node,
                     value >= 0 && value <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                     "Attribute '",
                     name,
                     "' is out of uint32 range: ",
                     value);
    return static_cast<uint32_t>(value);
}

uint32_t get_u32_attr_alias(const ov::frontend::onnx::Node& node,
                            const std::initializer_list<std::string>& names,
                            uint32_t default_value) {
    for (const auto& name : names) {
        if (node.has_attribute(name)) {
            const auto value = node.get_attribute_value<int64_t>(name, static_cast<int64_t>(default_value));
            return to_u32_checked(node, name, value);
        }
    }
    return default_value;
}

uint32_t get_static_dim_or_default(const ov::Output<ov::Node>& input, size_t dim_idx, uint32_t default_value) {
    const auto& pshape = input.get_partial_shape();
    if (!pshape.rank().is_static()) {
        return default_value;
    }

    const auto rank_len = pshape.rank().get_length();
    if (rank_len <= static_cast<int64_t>(dim_idx)) {
        return default_value;
    }

    const auto& dim = pshape[dim_idx];
    if (!dim.is_static()) {
        return default_value;
    }

    const auto dim_len = dim.get_length();
    if (dim_len < 0 || dim_len > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        return default_value;
    }

    return static_cast<uint32_t>(dim_len);
}

ov::op::v15::Bound get_bound_attr(const ov::frontend::onnx::Node& node, const std::string& prefix) {
    ov::op::v15::Bound bound;
    bound.min = static_cast<float>(node.get_attribute_value<double>(prefix + "_min", 0.0));
    bound.max = static_cast<float>(node.get_attribute_value<double>(prefix + "_max", 0.0));
    bound.step = static_cast<float>(node.get_attribute_value<double>(prefix + "_step", 1.0));

    CHECK_VALID_NODE(node,
                     bound.step != 0.0f,
                     "Attribute '",
                     prefix,
                     "_step' must not be zero.");
    return bound;
}
}  // namespace

ov::OutputVector bevpool_v2(const ov::frontend::onnx::Node& node) {
    auto inputs = node.get_ov_inputs();
    CHECK_VALID_NODE(node, !inputs.empty(), "BevPoolV2 expects at least one input.");
    CHECK_VALID_NODE(node,
                     inputs.size() == 4,
                     "BevPoolV2 expects exactly 4 inputs: feat, depth, indices, intervals. Got: ",
                     inputs.size());

    // 4-input ONNX schema is fixed to: [feat, depth, indices, intervals]
    // Internal BevPoolV2 constructor expects: [cf, dw, idx, itv], where cf is NHWC.
    auto cf_input = inputs[0];
    const auto dw_input = inputs[1];
    const auto idx_input = inputs[2];
    const auto itv_input = inputs[3];

    // Frontend layout adaptation for feat:
    // - preferred layout: NHWC
    // - if feat comes in NCHW, transpose to NHWC
    // Optional string attr `feat_layout` can force behavior: "NCHW" or "NHWC".
    const auto feat_layout = node.get_attribute_value<std::string>("feat_layout", "AUTO");
    bool convert_feat_nchw_to_nhwc = false;

    if (feat_layout == "NCHW") {
        convert_feat_nchw_to_nhwc = true;
    } else if (feat_layout == "AUTO") {
        const auto feat_ps = cf_input.get_partial_shape();
        const auto dw_ps = dw_input.get_partial_shape();

        if (feat_ps.rank().is_static() && dw_ps.rank().is_static() &&
            feat_ps.rank().get_length() == 4 && dw_ps.rank().get_length() == 4 &&
            feat_ps[2].is_static() && feat_ps[3].is_static() &&
            dw_ps[2].is_static() && dw_ps[3].is_static()) {
            // Detect NCHW by matching feat H/W against depth H/W at dims [2]/[3].
            const auto feat_h = feat_ps[2].get_length();
            const auto feat_w = feat_ps[3].get_length();
            const auto dw_h = dw_ps[2].get_length();
            const auto dw_w = dw_ps[3].get_length();
            if (feat_h == dw_h && feat_w == dw_w) {
                convert_feat_nchw_to_nhwc = true;
            }
        }
    }

    if (convert_feat_nchw_to_nhwc) {
        cf_input = std::make_shared<ov::op::v1::Transpose>(
            cf_input,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1}));
    }

    const auto input_channels = get_u32_attr_alias(
        node,
        {"input_channels", "in_channels", "channels"},
        get_static_dim_or_default(cf_input, 3, 1));

    const auto output_channels = get_u32_attr_alias(
        node,
        {"output_channels", "out_channels", "channels_out", "bev_channels"},
        get_static_dim_or_default(dw_input, 1, input_channels));

    const auto image_width = get_u32_attr_alias(node,
                                                {"image_width", "in_width", "width"},
                                                get_static_dim_or_default(cf_input, 2, 1));

    const auto image_height = get_u32_attr_alias(node,
                                                 {"image_height", "in_height", "height"},
                                                 get_static_dim_or_default(cf_input, 1, 1));

    const auto feature_width = get_u32_attr_alias(node,
                                                  {"feature_width", "out_width", "bev_width"},
                                                  1);

    const auto feature_height = get_u32_attr_alias(node,
                                                   {"feature_height", "out_height", "bev_height"},
                                                   1);

    CHECK_VALID_NODE(node, output_channels > 0, "Attribute 'output_channels' must be greater than zero.");
    CHECK_VALID_NODE(node, feature_width > 0, "Attribute 'feature_width' must be greater than zero.");
    CHECK_VALID_NODE(node, feature_height > 0, "Attribute 'feature_height' must be greater than zero.");

    const auto x_bound = get_bound_attr(node, "x_bound");
    const auto y_bound = get_bound_attr(node, "y_bound");
    const auto z_bound = get_bound_attr(node, "z_bound");
    auto d_bound = get_bound_attr(node, "d_bound");
    if (!node.has_attribute("d_bound_min") && !node.has_attribute("d_bound_max") && !node.has_attribute("d_bound_step")) {
        const auto depth_bins = get_static_dim_or_default(dw_input, 1, 1);
        d_bound.min = 0.0f;
        d_bound.max = static_cast<float>(depth_bins);
        d_bound.step = 1.0f;
    }

    return {std::make_shared<ov::op::v15::BevPoolV2>(ov::OutputVector{cf_input, dw_input, idx_input, itv_input},
                                                     input_channels,
                                                     output_channels,
                                                     image_width,
                                                     image_height,
                                                     feature_width,
                                                     feature_height,
                                                     x_bound,
                                                     y_bound,
                                                     z_bound,
                                                     d_bound)};
}

bool register_multiple_translators(void) {
    ONNX_OP_M("BevPoolV2", OPSET_SINCE(1), org_openvinotoolkit::opset_1::bevpool_v2, OPENVINO_ONNX_DOMAIN);
    ONNX_OP_M("BevPoolV2", OPSET_SINCE(1), org_openvinotoolkit::opset_1::bevpool_v2, "com.intel.bevpool");
    return true;
}

static bool registered = register_multiple_translators();

}  // namespace opset_1
}  // namespace org_openvinotoolkit
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
