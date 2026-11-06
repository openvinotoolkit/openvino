// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <random>

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/runtime/properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {
// Sequence length, hidden size
typedef std::tuple<size_t, size_t> FCMulGateViewParams;

// adaLN-style block (DiT, LTX-2): a modulation table [1, S, 6, D] is split along an inner axis into shift / scale /
// gate slices, so every slice is a strided view into one buffer. FullyConnected outputs are multiplied by the gate
// slices while the other slices are still read afterwards. The FC must not write its output in place into a gate
// view (binary_mul_inplace), or the remaining slices of the table get overwritten.
class FCMulGateView : public testing::WithParamInterface<FCMulGateViewParams>, virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FCMulGateViewParams>& params) {
        const auto& [seq, hidden] = params.param;
        std::ostringstream result;
        result << "S=" << seq << "_D=" << hidden;
        return result.str();
    }

protected:
    std::shared_ptr<ov::Node> fc(const ov::Output<ov::Node>& x, size_t k_in, size_t k_out, std::mt19937& gen) {
        std::normal_distribution<float> dist(0.f, 1.f / std::sqrt(static_cast<float>(k_in)));
        std::vector<float> w(k_out * k_in), b(k_out);
        for (auto& v : w)
            v = dist(gen);
        for (auto& v : b)
            v = dist(gen);
        auto weights = ov::op::v0::Constant::create(type, ov::Shape{k_out, k_in}, w);
        auto bias = ov::op::v0::Constant::create(type, ov::Shape{k_out}, b);
        return std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v0::MatMul>(x, weights, false, true), bias);
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        // f32 compute keeps accumulation noise small (~1e-3 abs); the aliasing bug causes O(1) errors at any precision
        configuration.insert({ov::hint::inference_precision.name(), ov::element::f32});
        abs_threshold = 2e-2;
        rel_threshold = 1e-2;
        const auto& [seq, hidden] = GetParam();
        const auto S = static_cast<int64_t>(seq);
        const auto D = static_cast<int64_t>(hidden);
        std::mt19937 gen(42);

        auto x = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{1, seq, hidden});
        auto cond = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{1, seq, hidden});
        auto table_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, S, 6, D});
        auto table = std::make_shared<ov::op::v1::Reshape>(fc(cond, hidden, 6 * hidden, gen), table_shape, false);
        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
        auto split = std::make_shared<ov::op::v1::Split>(table, axis, 6);
        auto squeeze_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        ov::OutputVector mod;
        for (size_t i = 0; i < 6; ++i)
            mod.push_back(std::make_shared<ov::op::v0::Squeeze>(split->output(i), squeeze_axis));
        auto one = ov::op::v0::Constant::create(type, ov::Shape{}, std::vector<float>{1.f});

        // shift = mod[0], scale = mod[1], gate = mod[2] for the first sub-block; mod[3..5] for the second one
        auto modulate = [&](const ov::Output<ov::Node>& h, size_t base) {
            auto scaled = std::make_shared<ov::op::v1::Multiply>(h, std::make_shared<ov::op::v1::Add>(mod[base + 1], one));
            return std::make_shared<ov::op::v1::Add>(scaled, mod[base]);
        };
        std::shared_ptr<ov::Node> h = x;
        for (size_t base : {0, 3}) {
            auto gated = std::make_shared<ov::op::v1::Multiply>(fc(modulate(h, base), hidden, hidden, gen), mod[base + 2]);
            h = std::make_shared<ov::op::v1::Add>(h, gated);
        }
        auto out = fc(h, hidden, hidden, gen);
        function = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{x, cond});
    }

    const ov::element::Type type = ov::element::f32;
};

TEST_P(FCMulGateView, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_GPU_FCMulGateView,
                         FCMulGateView,
                         ::testing::Values(FCMulGateViewParams{192, 256}, FCMulGateViewParams{64, 128}),
                         FCMulGateView::getTestCaseName);

}  // namespace
