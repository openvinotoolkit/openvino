// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/fuse_hadamard_fc.hpp"

#include <gtest/gtest.h>

#include <bitset>
#include <cmath>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace {

constexpr size_t N = 32;

// Normalised Sylvester H_b, scaled by `scale` (1 for the real transform).
std::shared_ptr<ov::op::v0::Constant> hadamard(size_t b, float scale = 1.0f) {
    std::vector<float> h(b * b);
    const float v = scale / std::sqrt(static_cast<float>(b));
    for (size_t i = 0; i < b; ++i) {
        for (size_t j = 0; j < b; ++j)
            h[i * b + j] = (std::bitset<32>(i & j).count() & 1) ? -v : v;
    }
    return ov::op::v0::Constant::create(ov::element::f16, ov::Shape{b, b}, h);
}

std::vector<int8_t> make_signs(size_t k) {
    std::vector<int8_t> s(k);
    for (size_t i = 0; i < k; ++i)
        s[i] = (i * 7 + 3) % 5 < 2 ? -1 : 1;
    return s;
}

std::shared_ptr<op::FullyConnectedCompressed> make_fc(const ov::Output<ov::Node>& x, size_t k) {
    auto weights = ov::op::v0::Constant::create(ov::element::u2, ov::Shape{N, k}, {1});
    auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{N, k / 128}, {1});
    auto zp = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{1}, {1});
    return std::make_shared<op::FullyConnectedCompressed>(x, weights, std::make_shared<op::Placeholder>(), scale, zp);
}

// x -> [Multiply(signs)] -> [Reshape(-1, K/B, B)] -> MatMul(H) -> [Reshape(-1, K)] -> FC. The Reshapes are absent when K == B.
std::shared_ptr<ov::Model> getModel(size_t k, size_t b, bool with_signs, const std::shared_ptr<ov::op::v0::Constant>& h) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, static_cast<int64_t>(k)});
    ov::Output<ov::Node> x = input;
    if (with_signs) {
        const auto s = make_signs(k);
        x = std::make_shared<ov::op::v1::Multiply>(x, ov::op::v0::Constant::create(ov::element::f16, ov::Shape{k}, std::vector<float>(s.begin(), s.end())));
    }
    if (k != b) {
        auto shape =
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{-1, static_cast<int64_t>(k / b), static_cast<int64_t>(b)});
        x = std::make_shared<ov::op::v1::Reshape>(x, shape, false);
    }
    x = std::make_shared<ov::op::v0::MatMul>(x, h);
    if (k != b) {
        auto shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(k)});
        x = std::make_shared<ov::op::v1::Reshape>(x, shape, false);
    }
    return std::make_shared<ov::Model>(ov::OutputVector{make_fc(x, k)}, ov::ParameterVector{input});
}

std::shared_ptr<ov::Model> getModelRef(size_t k, size_t b, bool with_signs) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, static_cast<int64_t>(k)});
    auto fc = make_fc(input, k);
    fc->get_rt_info()[int2_hadamard_block_key] = static_cast<int64_t>(b);
    fc->get_rt_info()[int2_hadamard_signs_key] = with_signs ? make_signs(k) : std::vector<int8_t>{};
    return std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});
}

// K, block, explicit sign Multiply
using FuseHadamardParams = std::tuple<size_t, size_t, bool>;

class FuseHadamardIntoFCTests : public testing::WithParamInterface<FuseHadamardParams>, public TransformationTestsF {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FuseHadamardParams>& info) {
        const auto& [k, b, with_signs] = info.param;
        return "K" + std::to_string(k) + "_B" + std::to_string(b) + (with_signs ? "_signs" : "_folded");
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        const auto& [k, b, with_signs] = GetParam();
        model = getModel(k, b, with_signs, hadamard(b));
        model_ref = getModelRef(k, b, with_signs);
        manager.register_pass<FuseHadamardIntoFC>();
    }
};

TEST_P(FuseHadamardIntoFCTests, Fused) {}

INSTANTIATE_TEST_SUITE_P(smoke,
                         FuseHadamardIntoFCTests,
                         ::testing::Values(FuseHadamardParams{1024, 256, true},
                                           FuseHadamardParams{2560, 512, true},
                                           FuseHadamardParams{5120, 1024, true},
                                           FuseHadamardParams{4096, 2048, false},
                                           FuseHadamardParams{8192, 4096, true},
                                           FuseHadamardParams{4096, 4096, true}),
                         FuseHadamardIntoFCTests::getTestCaseName);

TEST_F(TransformationTestsF, FuseHadamardIntoFC_UnsupportedBlockNotFused) {
    model = getModel(1024, 128, true, hadamard(128));
    manager.register_pass<FuseHadamardIntoFC>();
}

TEST_F(TransformationTestsF, FuseHadamardIntoFC_NonHadamardConstantNotFused) {
    model = getModel(1024, 512, true, hadamard(512, 2.0f));
    manager.register_pass<FuseHadamardIntoFC>();
}

// As emitted by bonsai2_gguf_to_ir.py: [?, ?, K] -> Multiply(signs) -> Reshape([0, 0, K/B, B]) -> MatMul(H) -> Reshape([0, 0, K]) -> FC.
TEST_F(TransformationTestsF, FuseHadamardIntoFC_Rank3SpecialZeroFused) {
    constexpr size_t k = 2048, b = 512;
    const auto s = make_signs(k);
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, static_cast<int64_t>(k)});
        auto mul =
            std::make_shared<ov::op::v1::Multiply>(input,
                                                   ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1, k}, std::vector<float>(s.begin(), s.end())));
        auto split = std::make_shared<ov::op::v1::Reshape>(
            mul,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, static_cast<int64_t>(k / b), static_cast<int64_t>(b)}),
            true);
        auto mm = std::make_shared<ov::op::v0::MatMul>(split, hadamard(b));
        auto merge = std::make_shared<ov::op::v1::Reshape>(
            mm,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 0, static_cast<int64_t>(k)}),
            true);
        model = std::make_shared<ov::Model>(ov::OutputVector{make_fc(merge, k)}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, static_cast<int64_t>(k)});
        auto fc = make_fc(input, k);
        fc->get_rt_info()[int2_hadamard_block_key] = static_cast<int64_t>(b);
        fc->get_rt_info()[int2_hadamard_signs_key] = s;
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});
    }
    manager.register_pass<FuseHadamardIntoFC>();
}

// K == B with the Reshape pair kept, and a source whose last dim is not known yet, as the RMS output in Bonsai 2 9B:
// [?, 1, ?] -> Reshape([0, 0, 1, B]) -> MatMul(H) -> Reshape([0, 0, K]) -> FC.
TEST_F(TransformationTestsF, FuseHadamardIntoFC_DynamicSourceLastDimFused) {
    constexpr size_t k = 1024;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 1, -1});
        auto split =
            std::make_shared<ov::op::v1::Reshape>(input, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 1, k}), true);
        auto mm = std::make_shared<ov::op::v0::MatMul>(split, hadamard(k));
        auto merge =
            std::make_shared<ov::op::v1::Reshape>(mm, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 0, k}), true);
        model = std::make_shared<ov::Model>(ov::OutputVector{make_fc(merge, k)}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 1, -1});
        auto fc = make_fc(input, k);
        fc->get_rt_info()[int2_hadamard_block_key] = static_cast<int64_t>(k);
        fc->get_rt_info()[int2_hadamard_signs_key] = std::vector<int8_t>{};
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});
    }
    manager.register_pass<FuseHadamardIntoFC>();
}

// A leading -1 could absorb part of a dynamic last dim: [?, ?] -> Reshape([-1, 1, 256]) -> MatMul(H) -> Reshape([-1, 256]) -> FC
// turns a runtime [2, 128] into [1, 256]; bypassing the Reshapes would hand [2, 128] to the FC.
TEST_F(TransformationTestsF, FuseHadamardIntoFC_InferredDimWithDynamicSourceNotFused) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1});
    auto split =
        std::make_shared<ov::op::v1::Reshape>(input, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{-1, 1, 256}), false);
    auto mm = std::make_shared<ov::op::v0::MatMul>(split, hadamard(256));
    auto merge = std::make_shared<ov::op::v1::Reshape>(mm, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{-1, 256}), false);
    model = std::make_shared<ov::Model>(ov::OutputVector{make_fc(merge, 256)}, ov::ParameterVector{input});
    manager.register_pass<FuseHadamardIntoFC>();
}

// A sign Multiply on data with a dynamic last dim may broadcast it (a runtime [2, 1] becomes [2, 256]), so it stays in
// the graph and only the rotation is fused: [?, ?] -> Multiply(signs) -> MatMul(H) -> FC.
TEST_F(TransformationTestsF, FuseHadamardIntoFC_BroadcastingSignsKept) {
    constexpr size_t k = 256;
    const auto s = make_signs(k);
    auto make_mul = [&](const ov::Output<ov::Node>& in) {
        return std::make_shared<ov::op::v1::Multiply>(in, ov::op::v0::Constant::create(ov::element::f16, ov::Shape{k}, std::vector<float>(s.begin(), s.end())));
    };
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1});
        auto mm = std::make_shared<ov::op::v0::MatMul>(make_mul(input), hadamard(k));
        model = std::make_shared<ov::Model>(ov::OutputVector{make_fc(mm, k)}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1});
        auto fc = make_fc(make_mul(input), k);
        fc->get_rt_info()[int2_hadamard_block_key] = static_cast<int64_t>(k);
        fc->get_rt_info()[int2_hadamard_signs_key] = std::vector<int8_t>{};
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});
    }
    manager.register_pass<FuseHadamardIntoFC>();
}

// K == B with an unpaired flatten in front of the MatMul: [2, 3, 256] -> Reshape([6, 256]) -> MatMul(H) -> FC.
// Bypassing the Reshape would turn the FC output [6, N] into [2, 3, N].
TEST_F(TransformationTestsF, FuseHadamardIntoFC_UnpairedReshapeNotFused) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{2, 3, 256});
    auto flatten =
        std::make_shared<ov::op::v1::Reshape>(input, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{6, 256}), false);
    auto mm = std::make_shared<ov::op::v0::MatMul>(flatten, hadamard(256));
    model = std::make_shared<ov::Model>(ov::OutputVector{make_fc(mm, 256)}, ov::ParameterVector{input});
    manager.register_pass<FuseHadamardIntoFC>();
}

// K > B, but the trailing Reshape flattens instead of restoring the source shape:
// [2, 3, 1024] -> Reshape([-1, 4, 256]) -> MatMul(H) -> Reshape([-1, 1024]) -> FC.
TEST_F(TransformationTestsF, FuseHadamardIntoFC_ShapeChangingReshapeNotFused) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{2, 3, 1024});
    auto split =
        std::make_shared<ov::op::v1::Reshape>(input, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{-1, 4, 256}), false);
    auto mm = std::make_shared<ov::op::v0::MatMul>(split, hadamard(256));
    auto merge = std::make_shared<ov::op::v1::Reshape>(mm, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{-1, 1024}), false);
    model = std::make_shared<ov::Model>(ov::OutputVector{make_fc(merge, 1024)}, ov::ParameterVector{input});
    manager.register_pass<FuseHadamardIntoFC>();
}

}  // namespace
