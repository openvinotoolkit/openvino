// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>

#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/predicate.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::intel_cpu {

template <class T>
bool match_conv_mul_add_fq(const std::shared_ptr<const ov::Node>& node) {
    static_assert(std::is_same_v<T, ov::op::v1::Subtract> || std::is_same_v<T, ov::op::v1::Multiply>,
                  "match_conv_mul_add_fq supports only Subtract and Multiply");

    using namespace ov::pass::pattern;

    auto conv_m = wrap_type<ov::op::v1::Convolution>(
        {any_input(type_matches_any({ov::element::i8, ov::element::u8})), any_input()});
    auto mul0_m = wrap_type<ov::op::v1::Multiply>({conv_m, any_input()});
    auto add_m = wrap_type<ov::op::v1::Add>({mul0_m, any_input()});
    auto fq_m = wrap_type<ov::op::v0::FakeQuantize>({add_m, any_input(), any_input(), any_input(), any_input()},
                                                    type_matches_any({ov::element::i8, ov::element::u8}));
    auto final_m = wrap_type<T>({fq_m, any_input()});

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(final_m);
    if (!matcher->match(std::const_pointer_cast<ov::Node>(node))) {
        return false;
    }

    const auto& pattern_map = matcher->get_pattern_value_map();
    const auto fq = pattern_map.at(fq_m).get_node_shared_ptr();
    const auto conv = pattern_map.at(conv_m).get_node_shared_ptr();

    return conv->get_input_element_type(0) == fq->get_output_element_type(0);
}

enum class FQMulAddPattern : std::uint8_t { ConvMulAdd, ConvAddMul };

// Shared skeleton for the ACL int8 "bare" chain: TGemm -> FakeQuantize.
// Matches the pattern and requires the GEMM activation input type to equal the FakeQuantize output type
// (the ACL int8 executors support only same activation and FQ output types).
template <class TGemm>
bool match_gemm_fq_same_types(const std::shared_ptr<const ov::Node>& node) {
    using namespace ov::pass::pattern;

    auto gemm_m = wrap_type<TGemm>();
    auto fq_m = wrap_type<ov::op::v0::FakeQuantize>({gemm_m, any_input(), any_input(), any_input(), any_input()});
    Matcher matcher(fq_m);
    if (!matcher.match(std::const_pointer_cast<ov::Node>(node))) {
        return false;
    }

    const auto& pattern_map = matcher.get_pattern_value_map();
    const auto gemm_node = pattern_map.at(gemm_m).get_node_shared_ptr();

    return gemm_node->get_input_element_type(0) == node->get_output_element_type(0);
}

// Shared skeleton for the ACL int8 dequantization tail feeding a FakeQuantize:
//   FQMulAddPattern::ConvMulAdd -> TGemm -> Multiply -> Add -> FakeQuantize
//   FQMulAddPattern::ConvAddMul -> TGemm -> Add -> Multiply -> FakeQuantize
// Requires the GEMM activation input type to equal the FakeQuantize output type.
//
// Per-op differences are expressed via parameters, keeping conv/matmul semantics intact:
//   - `act_pred`: optional predicate pinning the GEMM activation input(0) inside the pattern
//     (matmul pins it to i8/u8; conv leaves it unconstrained and relies on the same-type check + the
//     u8/i8 output guard done by its caller).
//   - `extra`: optional post-match constraint on the matched FakeQuantize node
//     (matmul additionally requires per-tensor scalar FQ-scale constants; conv has no such constraint).
template <class TGemm>
bool match_gemm_bias_fq_same_types(const std::shared_ptr<const ov::Node>& node,
                                   FQMulAddPattern pattern,
                                   const ov::pass::pattern::op::Predicate& act_pred = {},
                                   const std::function<bool(const std::shared_ptr<const ov::Node>&)>& extra = {}) {
    using namespace ov::pass::pattern;

    auto mulAdd_gemm = wrap_type<TGemm>({any_input(act_pred), any_input()});
    auto mulAdd_mul = wrap_type<ov::op::v1::Multiply>({mulAdd_gemm, any_input()});
    auto mulAdd_add = wrap_type<ov::op::v1::Add>({mulAdd_mul, any_input()});
    auto mulAdd_fq =
        wrap_type<ov::op::v0::FakeQuantize>({mulAdd_add, any_input(), any_input(), any_input(), any_input()});
    Matcher mulAdd_matcher(mulAdd_fq);

    auto addMul_gemm = wrap_type<TGemm>({any_input(act_pred), any_input()});
    auto addMul_add = wrap_type<ov::op::v1::Add>({addMul_gemm, any_input()});
    auto addMul_mul = wrap_type<ov::op::v1::Multiply>({addMul_add, any_input()});
    auto addMul_fq =
        wrap_type<ov::op::v0::FakeQuantize>({addMul_mul, any_input(), any_input(), any_input(), any_input()});
    Matcher addMul_matcher(addMul_fq);

    const bool is_mul_add = (pattern == FQMulAddPattern::ConvMulAdd);
    auto& matcher = is_mul_add ? mulAdd_matcher : addMul_matcher;
    const auto& gemm_m = is_mul_add ? mulAdd_gemm : addMul_gemm;
    if (!matcher.match(std::const_pointer_cast<ov::Node>(node))) {
        return false;
    }

    const auto& pattern_map = matcher.get_pattern_value_map();
    const auto gemm_node = pattern_map.at(gemm_m).get_node_shared_ptr();
    if (gemm_node->get_input_element_type(0) != node->get_output_element_type(0)) {
        return false;
    }

    return !extra || extra(node);
}

bool match_fq_mul_conv_bias_same_types(const std::shared_ptr<const ov::Node>& node, FQMulAddPattern pattern);

bool match_conv_fq_same_types(const std::shared_ptr<const ov::Node>& node);

bool match_acl_int8_conv_fq_chain(const std::shared_ptr<const ov::Node>& node);

bool match_acl_int8_matmul_fq_chain(const std::shared_ptr<const ov::Node>& node);

bool match_acl_int8_pooling_fq_chain(const std::shared_ptr<const ov::Node>& node);

bool is_acl_int8_avg_pool_lpt_skipped(const std::shared_ptr<const ov::Node>& node,
                                      const std::vector<ov::element::Type>& defaultPrecisions);

bool match_acl_int8_conv_add_multiply_chain(const std::shared_ptr<const ov::Node>& node);

bool match_conv_stride_oc_ic_limit(const std::shared_ptr<const ov::Node>& node,
                                   const std::vector<int64_t>& strides,
                                   const ov::Shape& kernel_shape,
                                   size_t oc_ic_limit);

}  // namespace ov::intel_cpu
