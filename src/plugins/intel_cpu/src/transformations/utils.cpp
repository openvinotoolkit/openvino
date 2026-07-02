// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "low_precision/network_helper.hpp"
#include "low_precision/resolve_precision_attribute.hpp"
#include "openvino/core/descriptor/output.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/op/util/avg_pool_base.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils/general_utils.h"

using namespace ov::pass::pattern;

namespace ov::intel_cpu {

namespace {

std::shared_ptr<ov::Node> get_consumer(const ov::Output<const ov::Node>& output) {
    const auto& consumers = output.get_target_inputs();
    if (consumers.size() != 1) {
        return nullptr;
    }

    return consumers.begin()->get_node()->shared_from_this();
}

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

    const auto pool = node->get_input_node_shared_ptr(0);
    if (pool->output(0).get_target_inputs().size() != 1) {
        return false;
    }

    const auto avg_pool = ov::as_type_ptr<const ov::op::util::AvgPoolBase>(pool);
    if (avg_pool && avg_pool->get_rounding_type() == ov::op::RoundingType::CEIL) {
        return false;
    }

    // returns true if Pooling-FQ chain will be fused into int8 pooling and handled by ACL executor
    return any_of(node->get_output_element_type(0), ov::element::Type_t::u8, ov::element::Type_t::i8) &&
           (ov::is_type_any_of<ov::op::v1::AvgPool, ov::op::v14::AvgPool, ov::op::v16::AvgPool>(pool) ||
            ov::is_type_any_of<ov::op::v1::MaxPool, ov::op::v8::MaxPool, ov::op::v14::MaxPool>(pool));
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

bool is_acl_int8_avg_pool_lpt_skipped(const std::shared_ptr<const ov::Node>& node,
                                      const std::vector<ov::element::Type>& defaultPrecisions) {
    const auto avg_pool = ov::as_type_ptr<const ov::op::util::AvgPoolBase>(node);
    if (!avg_pool) {
        return true;
    }

    const auto& input_pshape = avg_pool->get_input_partial_shape(0);
    const auto input_rank = input_pshape.rank();
    if (input_rank.is_dynamic() || input_rank.get_length() == 5) {
        return true;
    }

    const auto dequantization = ov::pass::low_precision::NetworkHelper::getDequantization(avg_pool, defaultPrecisions);
    if (dequantization.empty() ||
        !any_of(dequantization.data.get_element_type(), ov::element::Type_t::u8, ov::element::Type_t::i8)) {
        return true;
    }

    // ACL rejects NCHW AvgPool with CEIL rounding in the executor wrapper.
    if (avg_pool->get_rounding_type() == op::RoundingType::CEIL) {
        return true;
    }

    const auto fq_consumer = get_consumer(avg_pool->output(0));
    if (!fq_consumer) {
        return true;
    }

    const auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(fq_consumer);
    if (!fq) {
        return true;
    }

    const auto resolved_precision = ov::pass::low_precision::ResolvePrecisionAttribute::getDataPrecision(fq);
    return resolved_precision.empty() ||
           !any_of(resolved_precision.precision, ov::element::Type_t::u8, ov::element::Type_t::i8) ||
           dequantization.data.get_element_type() != resolved_precision.precision;
}

bool match_acl_int8_conv_add_multiply_chain(const std::shared_ptr<const ov::Node>& node) {
    const auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(node);
    if (!conv) {
        return false;
    }

    const auto first_consumer = get_consumer(conv->output(0));
    if (!first_consumer) {
        return false;
    }

    if (ov::is_type<ov::op::v0::FakeQuantize>(first_consumer)) {
        return true;
    }

    const auto add = ov::as_type_ptr<const ov::op::v1::Add>(first_consumer);
    if (!add) {
        return false;
    }

    // Accept Conv->FQ and Conv->Add->FQ only.
    // Activations between bias and FQ are not supported here yet, some of them will be enabled later
    const auto second_consumer = get_consumer(add->output(0));
    if (!second_consumer) {
        return false;
    }

    return ov::is_type<ov::op::v0::FakeQuantize>(second_consumer);
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
