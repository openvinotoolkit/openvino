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

}  // namespace
