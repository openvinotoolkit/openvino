// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_to_power_static.hpp"

#include <cstdint>
#include <memory>
#include <type_traits>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/normalize_l2.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/fully_connected.hpp"
#include "transformations/cpu_opset/common/op/power_static.hpp"
#include "transformations/rt_info/dequantization_node.hpp"
#include "utils/general_utils.h"

namespace {

int getConstPort(const std::shared_ptr<ov::Node>& node) {
    const auto const1 = ov::as_type_ptr<ov::op::v0::Constant>(node->get_input_node_shared_ptr(0));
    const auto const2 = ov::as_type_ptr<ov::op::v0::Constant>(node->get_input_node_shared_ptr(1));
    int constPort = -1;
    if (const2) {
        constPort = 1;
    } else if (const1) {
        constPort = 0;
    }
    return constPort;
}

template <class BaseOp>
bool isConvertableToPowerStatic(const std::shared_ptr<BaseOp>& node) {
    const int constPort = getConstPort(node);
    if ((!node->get_input_element_type(0).is_real() && !node->get_input_element_type(1).is_real()) ||
        !node->get_output_element_type(0).is_real() || constPort == -1) {
        return false;
    }

    const int nonConstPort = 1 - constPort;
    auto input_rank = node->get_input_partial_shape(nonConstPort).rank();
    if (input_rank.is_dynamic()) {
        return false;
    }
    auto const_shape = node->get_input_shape(constPort);
    return ov::shape_size(const_shape) == 1 && input_rank.get_length() >= static_cast<int64_t>(const_shape.size()) &&
           !ov::intel_cpu::any_of(node->get_input_node_shared_ptr(nonConstPort)->get_type_info(),
                                  ov::op::v0::NormalizeL2::get_type_info_static(),
                                  ov::op::v0::Interpolate::get_type_info_static(),
                                  ov::op::v1::Convolution::get_type_info_static(),
                                  ov::op::v1::GroupConvolution::get_type_info_static(),
                                  ov::op::v1::ConvolutionBackpropData::get_type_info_static(),
                                  ov::op::v1::GroupConvolutionBackpropData::get_type_info_static(),
                                  ov::op::v0::MatMul::get_type_info_static(),
                                  ov::op::internal::FullyConnected::get_type_info_static(),
                                  ov::op::v0::MVN::get_type_info_static(),
                                  ov::op::v6::MVN::get_type_info_static());
}

template <>
bool isConvertableToPowerStatic(const std::shared_ptr<ov::op::v1::Power>& node) {
    auto input_rank = node->get_input_partial_shape(0).rank();
    if (input_rank.is_dynamic()) {
        return false;
    }
    auto const_node = ov::as_type_ptr<ov::op::v0::Constant>(node->get_input_node_shared_ptr(1));
    return const_node &&
           input_rank.get_length() >= static_cast<ov::Dimension::value_type>(const_node->get_shape().size()) &&
           ov::shape_size(const_node->get_shape()) == 1;
}

template <class BaseOp>
std::shared_ptr<ov::Node> convert(const std::shared_ptr<BaseOp>& node) {
    const int constPort = getConstPort(node);
    const int nonConstPort = 1 - constPort;
    std::shared_ptr<ov::op::v0::Constant> powerNode =
        ov::as_type_ptr<ov::op::v0::Constant>(node->get_input_node_shared_ptr(constPort));
    const float value = powerNode->cast_vector<float>()[0];
    if (std::is_same_v<BaseOp, ov::op::v1::Power>) {
        return std::make_shared<ov::intel_cpu::PowerStaticNode>(node->input(nonConstPort).get_source_output(),
                                                                value,
                                                                1.0F,
                                                                0.0F,
                                                                node->output(0).get_element_type());
    }
    if (std::is_same_v<BaseOp, ov::op::v1::Add>) {
        return std::make_shared<ov::intel_cpu::PowerStaticNode>(node->input(nonConstPort).get_source_output(),
                                                                1.0F,
                                                                1.0F,
                                                                value,
                                                                node->output(0).get_element_type());
    }
    if (std::is_same_v<BaseOp, ov::op::v1::Subtract>) {
        float scale = 1.0F;
        float shift = value;
        if (constPort == 0) {
            scale *= -1.0F;
        } else {
            shift *= -1.0F;
        }
        return std::make_shared<ov::intel_cpu::PowerStaticNode>(node->input(nonConstPort).get_source_output(),
                                                                1.0F,
                                                                scale,
                                                                shift,
                                                                node->output(0).get_element_type());
    }
    if (std::is_same_v<BaseOp, ov::op::v1::Multiply>) {
        return std::make_shared<ov::intel_cpu::PowerStaticNode>(node->input(nonConstPort).get_source_output(),
                                                                1.F,
                                                                value,
                                                                0.0F,
                                                                node->output(0).get_element_type());
    }
    OPENVINO_THROW("ConvertToPowerStatic: op type is not supported");
}

}  // namespace

ov::intel_cpu::ConvertToPowerStatic::ConvertToPowerStatic() {
    MATCHER_SCOPE(ConvertToPowerStatic);
    ov::OutputVector twoInputs = {ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank()),
                                  ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank())};
    // Decompression/dequantization nodes are not converted to PowerStatic because
    // they are always fused into other operations or constant folded in CPU graph optimizations
    // If these constants converted into PowerStatic, we would have to handle these specific cases in plugin fusings
    auto not_dequantization_or_decompression = [](const ov::Output<ov::Node>& output) {
        return !ov::is_dequantization_node(output.get_node_shared_ptr());
    };
    auto power = ov::pass::pattern::wrap_type<ov::op::v1::Power>(twoInputs, not_dequantization_or_decompression);
    auto add = ov::pass::pattern::wrap_type<ov::op::v1::Add>(twoInputs, not_dequantization_or_decompression);
    auto sub = ov::pass::pattern::wrap_type<ov::op::v1::Subtract>(twoInputs, not_dequantization_or_decompression);
    auto mult = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>(twoInputs, not_dequantization_or_decompression);
    const auto candidate = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{power, add, sub, mult});

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto node = m.get_match_root();

        std::shared_ptr<ov::Node> toReplace = node;
        if (auto power = ov::as_type_ptr<ov::op::v1::Power>(node)) {
            if (!isConvertableToPowerStatic(power)) {
                return false;
            }
            toReplace = convert(power);
        } else if (auto add = ov::as_type_ptr<ov::op::v1::Add>(node)) {
            if (!isConvertableToPowerStatic(add)) {
                return false;
            }
            toReplace = convert(add);
        } else if (auto sub = ov::as_type_ptr<ov::op::v1::Subtract>(node)) {
            if (!isConvertableToPowerStatic(sub)) {
                return false;
            }
            toReplace = convert(sub);
        } else if (auto mult = ov::as_type_ptr<ov::op::v1::Multiply>(node)) {
            if (!isConvertableToPowerStatic(mult)) {
                return false;
            }
            toReplace = convert(mult);
        } else {
            OPENVINO_THROW("ConvertToPowerStatic: op type is not supported");
        }
        toReplace->set_friendly_name(node->get_friendly_name());
        ov::copy_runtime_info(node, toReplace);
        ov::replace_node(node, toReplace);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(candidate, matcher_name);
    this->register_matcher(m, callback);
}
