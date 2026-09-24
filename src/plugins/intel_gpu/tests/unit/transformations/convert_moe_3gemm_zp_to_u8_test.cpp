// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/convert_moe_3gemm_zp_to_u8.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "ov_ops/moe_compressed.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov::test::intel_gpu {

namespace {

using MOECompressed = ov::op::internal::MOECompressed;

constexpr size_t num_experts = 4, hidden_size = 256, inter_size = 128, top_k = 2, group_size = 128;

MOECompressed::Config gemm3_config() {
    MOECompressed::Config c;
    c.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
    c.activation_type = ov::op::internal::MOE::Activation_type::SWIGLU;
    c.hidden_size = hidden_size;
    c.inter_size = inter_size;
    c.num_expert = num_experts;
    c.num_shared_expert = 0;
    c.top_k = top_k;
    c.group_size = group_size;
    c.has_zp = true;
    c.out_type = ov::element::f16;
    return c;
}

// Grouped weights [E, N, K/G, G] and scales [E, N, K/G, 1] for gate/up/down
ov::Shape wei_shape(size_t n, size_t k) {
    return {num_experts, n, k / group_size, group_size};
}
ov::Shape scale_shape(size_t n, size_t k) {
    return {num_experts, n, k / group_size, 1};
}

// make_zp(n, k) returns the zp input of the GEMM with N=n, K=k
template <typename ZpFactory>
std::shared_ptr<ov::Model> make_model(ov::element::Type wei_type, ZpFactory make_zp) {
    using C = ov::op::v0::Constant;
    auto hidden = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, hidden_size});
    auto routing = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, top_k});
    auto topk_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, top_k});
    ov::OutputVector inputs{hidden, routing, topk_idx};
    for (auto [n, k] : {std::pair{inter_size, hidden_size}, std::pair{inter_size, hidden_size}, std::pair{hidden_size, inter_size}}) {
        inputs.push_back(C::create(wei_type, wei_shape(n, k), std::vector<uint8_t>(ov::shape_size(wei_shape(n, k)), 1)));
        inputs.push_back(C::create(ov::element::f16, scale_shape(n, k), std::vector<float>(ov::shape_size(scale_shape(n, k)), 1.0f)));
        inputs.push_back(make_zp(n, k));
    }
    auto moe = std::make_shared<MOECompressed>(inputs, gemm3_config());
    return std::make_shared<ov::Model>(ov::OutputVector{moe}, ov::ParameterVector{hidden, routing, topk_idx});
}

}  // namespace

// NNCF int3 layout: u3 weights with one i8 zp (= 4) for all experts, expanded to the scale shape
TEST_F(TransformationTestsF, ConvertMOE3GemmZpToU8_ScalarZpBroadcast) {
    using C = ov::op::v0::Constant;
    model = make_model(ov::element::u3, [](size_t, size_t) {
        return C::create(ov::element::i8, ov::Shape{}, {4});
    });
    manager.register_pass<ConvertMOE3GemmZpToU8>();

    model_ref = make_model(ov::element::u3, [](size_t n, size_t k) {
        return C::create(ov::element::i8, scale_shape(n, k), std::vector<int8_t>(ov::shape_size(scale_shape(n, k)), 4));
    });
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

// Per-group u3 zp is converted to u8 (byte-addressed reorders / OneDNN zp buffers)
TEST_F(TransformationTestsF, ConvertMOE3GemmZpToU8_U3ZpToU8) {
    using C = ov::op::v0::Constant;
    model = make_model(ov::element::u3, [](size_t n, size_t k) {
        return C::create(ov::element::u3, scale_shape(n, k), std::vector<uint8_t>(ov::shape_size(scale_shape(n, k)), 5));
    });
    manager.register_pass<ConvertMOE3GemmZpToU8>();

    model_ref = make_model(ov::element::u3, [](size_t n, size_t k) {
        return C::create(ov::element::u8, scale_shape(n, k), std::vector<uint8_t>(ov::shape_size(scale_shape(n, k)), 5));
    });
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

// Scalar u3 zp: converted to u8, then broadcast
TEST_F(TransformationTestsF, ConvertMOE3GemmZpToU8_ScalarU3Zp) {
    using C = ov::op::v0::Constant;
    model = make_model(ov::element::u3, [](size_t, size_t) {
        return C::create(ov::element::u3, ov::Shape{1, 1, 1, 1}, {4});
    });
    manager.register_pass<ConvertMOE3GemmZpToU8>();

    model_ref = make_model(ov::element::u3, [](size_t n, size_t k) {
        return C::create(ov::element::u8, scale_shape(n, k), std::vector<uint8_t>(ov::shape_size(scale_shape(n, k)), 4));
    });
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

// Sub-byte zp other than u3 is not touched (u4 scalar zp stays scalar, per-group u4 zp stays u4)
TEST_F(TransformationTestsF, ConvertMOE3GemmZpToU8_U4ZpUnchanged) {
    using C = ov::op::v0::Constant;
    model = make_model(ov::element::u4, [](size_t, size_t) {
        return C::create(ov::element::u4, ov::Shape{}, {8});
    });
    manager.register_pass<ConvertMOE3GemmZpToU8>();
}

}  // namespace ov::test::intel_gpu
