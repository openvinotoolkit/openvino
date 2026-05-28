// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <memory>
#include <string>
#include <vector>

#if defined(OV_CPU_WITH_ACL)
#    include "nodes/executors/acl/acl_pooling.hpp"
#    include "nodes/executors/acl/acl_utils.hpp"
#endif
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "utils/general_utils.h"

using namespace ov::pass::pattern;

namespace ov::intel_cpu {

namespace {

#if defined(OV_CPU_WITH_ACL)
void copy_pool_attributes(std::vector<ptrdiff_t>& internal_attribute, const std::vector<size_t>& external_attribute) {
    for (const auto attribute : external_attribute) {
        internal_attribute.push_back(static_cast<ptrdiff_t>(attribute));
    }
}

std::optional<PoolingAttrs> get_avg_pooling_attrs(const std::shared_ptr<const ov::Node>& node) {
    const auto avg_pool = ov::as_type_ptr<const ov::op::util::AvgPoolBase>(node);
    if (!avg_pool) {
        return std::nullopt;
    }

    PoolingAttrs pooling_attrs;
    pooling_attrs.algorithm = Algorithm::PoolingAvg;
    pooling_attrs.exclude_pad = avg_pool->get_exclude_pad();
    pooling_attrs.rounding = avg_pool->get_rounding_type();
    copy_pool_attributes(pooling_attrs.kernel, avg_pool->get_kernel());
    copy_pool_attributes(pooling_attrs.stride, avg_pool->get_strides());
    if (const auto avg_pool_v16 = ov::as_type_ptr<const ov::op::v16::AvgPool>(node)) {
        copy_pool_attributes(pooling_attrs.dilation, avg_pool_v16->get_dilations());
    } else {
        pooling_attrs.dilation.resize(pooling_attrs.kernel.size(), 1);
    }
    copy_pool_attributes(pooling_attrs.data_pad_begin, avg_pool->get_pads_begin());
    copy_pool_attributes(pooling_attrs.data_pad_end, avg_pool->get_pads_end());
    pooling_attrs.auto_pad = any_of(avg_pool->get_auto_pad(), ov::op::PadType::SAME_LOWER, ov::op::PadType::SAME_UPPER);
    return pooling_attrs;
}

std::optional<std::pair<std::vector<float>, std::vector<float>>> get_fq_input_quant_params(
    const std::shared_ptr<const ov::op::v0::FakeQuantize>& fq) {
    const auto input_low = ov::util::get_constant_from_source(fq->input_value(1));
    const auto input_high = ov::util::get_constant_from_source(fq->input_value(2));
    if (!input_low || !input_high) {
        return std::nullopt;
    }

    const auto input_low_values = input_low->cast_vector<float>();
    const auto input_high_values = input_high->cast_vector<float>();
    if (input_low_values.empty() || input_high_values.empty()) {
        return std::nullopt;
    }

    const size_t channels = std::max(input_low_values.size(), input_high_values.size());
    std::vector<float> input_scale(channels);
    std::vector<float> input_shift(channels);
    const float levels = static_cast<float>(fq->get_levels() - 1);

    for (size_t index = 0; index < channels; ++index) {
        const auto input_low_value = input_low_values[input_low_values.size() == 1 ? 0 : index];
        const auto input_high_value = input_high_values[input_high_values.size() == 1 ? 0 : index];
        const auto range = input_high_value - input_low_value;
        if (range == 0.0f) {
            return std::nullopt;
        }
        input_scale[index] = levels / range;
        input_shift[index] = -input_low_value * levels / range;
    }

    return std::make_pair(std::move(input_scale), std::move(input_shift));
}

bool is_acl_supported_avg_pooling_fq_chain(const std::shared_ptr<const ov::Node>& node) {
    const auto fq = ov::as_type_ptr<const ov::op::v0::FakeQuantize>(node);
    if (!fq || fq->get_input_size() == 0) {
        return false;
    }

    const auto avg_pool = fq->get_input_node_shared_ptr(0);
    if (!avg_pool || avg_pool->get_input_partial_shape(0).is_dynamic() || avg_pool->get_output_partial_shape(0).is_dynamic()) {
        return false;
    }

    const auto pooling_attrs = get_avg_pooling_attrs(avg_pool);
    if (!pooling_attrs) {
        return false;
    }

    const auto& input_shape = avg_pool->get_input_shape(0);
    const auto& output_shape = avg_pool->get_output_shape(0);
    if (input_shape.size() == 5U) {
        // Pooling::getSupportedDescriptors() still forces 5D pooling away from ACL.
        return false;
    }

    arm_compute::DataLayout data_layout =
        (input_shape.size() == 5U) ? arm_compute::DataLayout::NDHWC : arm_compute::DataLayout::NCHW;
    arm_compute::TensorInfo src_tensor_info(shapeCast(input_shape),
                                            1,
                                            convertToQuantizedType(precisionToAclDataType(avg_pool->get_input_element_type(0))),
                                            data_layout);
    arm_compute::TensorInfo dst_tensor_info(shapeCast(output_shape),
                                            1,
                                            convertToQuantizedType(precisionToAclDataType(node->get_output_element_type(0))),
                                            data_layout);

    if (any_of(avg_pool->get_input_element_type(0), ov::element::Type_t::u8, ov::element::Type_t::i8) ||
        any_of(node->get_output_element_type(0), ov::element::Type_t::u8, ov::element::Type_t::i8)) {
        const auto fq_quant_params = get_fq_input_quant_params(fq);
        if (!fq_quant_params) {
            return false;
        }

        src_tensor_info.set_quantization_info(arm_compute::QuantizationInfo(1.0F));
        dst_tensor_info.set_quantization_info(
            getDstQuantizationInfo(fq_quant_params->first, fq_quant_params->second, node->get_output_element_type(0)));
    }

    arm_compute::PoolingLayerInfo pool_info;
    return AclPoolingExecutor::isSupported(src_tensor_info,
                                           dst_tensor_info,
                                           *pooling_attrs,
                                           input_shape.size(),
                                           1,
                                           data_layout,
                                           nullptr,
                                           &pool_info,
                                           nullptr,
                                           false);
}
#endif

}  // namespace

bool match_fq_mul_conv_bias_same_types(const std::shared_ptr<const ov::Node>& node, FQMulAddPattern pattern) {
    auto convMulAdd_conv = wrap_type<ov::op::v1::Convolution>();
    auto convMulAdd_mul = wrap_type<ov::op::v1::Multiply>({convMulAdd_conv, any_input()});
    auto convMulAdd_add = wrap_type<ov::op::v1::Add>({convMulAdd_mul, any_input()});
    auto convMulAdd_fq =
        wrap_type<ov::op::v0::FakeQuantize>({convMulAdd_add, any_input(), any_input(), any_input(), any_input()});
    Matcher convMulAdd_matcher(convMulAdd_fq);
    auto convAddMul_conv = wrap_type<ov::op::v1::Convolution>();
    auto convAddMul_add = wrap_type<ov::op::v1::Add>({convAddMul_conv, any_input()});
    auto convAddMul_mul = wrap_type<ov::op::v1::Multiply>({convAddMul_add, any_input()});
    auto convAddMul_fq =
        wrap_type<ov::op::v0::FakeQuantize>({convAddMul_mul, any_input(), any_input(), any_input(), any_input()});
    Matcher convAddMul_matcher(convAddMul_fq);
    auto matcher = (pattern == FQMulAddPattern::ConvMulAdd) ? convMulAdd_matcher : convAddMul_matcher;
    if (!matcher.match(std::const_pointer_cast<ov::Node>(node))) {
        return false;
    }
    const auto& pattern_map = matcher.get_pattern_value_map();
    auto conv = pattern_map.at((pattern == FQMulAddPattern::ConvMulAdd) ? convMulAdd_conv : convAddMul_conv);

    return conv.get_node_shared_ptr()->get_input_element_type(0) == node->get_output_element_type(0);
}

bool match_conv_fq_same_types(const std::shared_ptr<const ov::Node>& node) {
    auto conv = wrap_type<ov::op::v1::Convolution>();
    auto fq = wrap_type<ov::op::v0::FakeQuantize>({conv, any_input(), any_input(), any_input(), any_input()});
    Matcher matcher(fq);
    if (!matcher.match(std::const_pointer_cast<ov::Node>(node))) {
        return false;
    }

    const auto& pattern_map = matcher.get_pattern_value_map();
    const auto conv_node = pattern_map.at(conv).get_node_shared_ptr();

    return conv_node->get_input_element_type(0) == node->get_output_element_type(0);
}

bool match_acl_int8_pooling_fq_chain(const std::shared_ptr<const ov::Node>& node) {
    if (!node || !ov::is_type<const ov::op::v0::FakeQuantize>(node) || node->get_input_size() == 0) {
        return false;
    }

    if (!any_of(node->get_output_element_type(0), ov::element::Type_t::u8, ov::element::Type_t::i8)) {
        return false;
    }

    const auto pool = node->get_input_node_shared_ptr(0);
    // returns true if Pooling-FQ chain will be fused into int8 pooling and handled by ACL executor
    const bool isMaxPool = ov::is_type_any_of<ov::op::v1::MaxPool, ov::op::v8::MaxPool, ov::op::v14::MaxPool>(pool);
#if defined(OV_CPU_WITH_ACL)
    return isMaxPool || is_acl_supported_avg_pooling_fq_chain(node);
#else
    return isMaxPool;
#endif
}

bool match_acl_int8_conv_fq_chain(const std::shared_ptr<const ov::Node>& node) {
    if (!node) {
        return false;
    }
    // returns true if Conv-Add-Mul-FQ chain will be fused into int8 convolution and handled by ACL executor
    // int8 ACL Convolution executor supports only same activation and FQ output types
    return ov::is_type<const ov::op::v0::FakeQuantize>(node) &&
           any_of(node->get_output_element_type(0), ov::element::Type_t::u8, ov::element::Type_t::i8) &&
           (match_conv_fq_same_types(node) || match_fq_mul_conv_bias_same_types(node, FQMulAddPattern::ConvAddMul));
}

bool match_conv_stride_oc_ic_limit(const std::shared_ptr<const ov::Node>& node,
                                   const std::vector<int64_t>& strides,
                                   const ov::Shape& kernel_shape,
                                   size_t oc_ic_limit) {
    const auto weights_shape = "OC, IC, " + std::to_string(kernel_shape[0]) + ", " + std::to_string(kernel_shape[1]);
    const auto weights_m = any_input(has_static_shape() && shape_matches(weights_shape));
    const auto conv_m = wrap_type<ov::op::v1::Convolution>({any_input(), weights_m}, {{"strides", strides}});
    Matcher matcher(conv_m);
    if (!matcher.match(std::const_pointer_cast<ov::Node>(node))) {
        return false;
    }

    const auto& symbols = matcher.get_symbols();
    const auto oc = symbols.at("OC").i();
    const auto ic = symbols.at("IC").i();
    return (oc >= 0 && static_cast<size_t>(oc) < oc_ic_limit) || (ic >= 0 && static_cast<size_t>(ic) < oc_ic_limit);
}

}  // namespace ov::intel_cpu
