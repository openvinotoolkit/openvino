// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/l2_norm_to_rms.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sqrt.hpp"
#include "ov_ops/rms.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace {

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v4 = ov::op::v4;

// ---------------------------------------------------------------------------------------------------------------------
// Last-axis form
// ---------------------------------------------------------------------------------------------------------------------

constexpr size_t hidden = 64;
constexpr float last_axis_eps = 1e-6f;

// x / sqrt(ReduceSum(x^2, axis) + eps) * gamma, as an ONNX LpNormalization(p=2) followed by a scale exports it.
std::shared_ptr<ov::Node> l2_norm(const ov::Output<ov::Node>& x,
                                  const ov::Output<ov::Node>& gamma,
                                  int64_t axis = -1,
                                  bool keep_dims = true,
                                  float eps = last_axis_eps,
                                  const ov::op::AutoBroadcastSpec& gamma_broadcast = ov::op::AutoBroadcastType::NUMPY) {
    auto power = std::make_shared<v1::Power>(x, v0::Constant::create(x.get_element_type(), {}, {2.f}));
    auto axes = v0::Constant::create(ov::element::i64, {1}, {axis});
    auto sum = std::make_shared<v1::ReduceSum>(power, axes, keep_dims);
    auto add = std::make_shared<v1::Add>(sum, v0::Constant::create(x.get_element_type(), {}, {eps}));
    auto sqrt = std::make_shared<v0::Sqrt>(add);
    auto div = std::make_shared<v1::Divide>(x, sqrt);
    return std::make_shared<v1::Multiply>(div, gamma, gamma_broadcast);
}

std::shared_ptr<ov::Node> last_axis_rms(const ov::Output<ov::Node>& x, const ov::Output<ov::Node>& gamma, ov::element::Type out_type) {
    auto inv_sqrt_n = v0::Constant::create(gamma.get_element_type(), {}, {1.0 / std::sqrt(static_cast<double>(hidden))});
    auto scaled_gamma = std::make_shared<v1::Multiply>(gamma, inv_sqrt_n);
    return std::make_shared<ov::op::internal::RMS>(x, scaled_gamma, last_axis_eps / static_cast<float>(hidden), out_type);
}

// ---------------------------------------------------------------------------------------------------------------------
// Channel-axis form
// ---------------------------------------------------------------------------------------------------------------------

constexpr size_t channels = 32;
constexpr float scale_value = 2.5f;

// The pass fuses only an eps <= denorm_min(type) / 1024: 1e-12, torch's F.normalize default, qualifies for f16
// (limit ~5.8e-11), while for f32 practically only 0 does.
double negligible_eps(ov::element::Type type) {
    return type == ov::element::f16 ? 1e-12 : 0.0;
}

enum class Floor { Clamp, MaxMin, Max };
enum class Inverse { Divide, Power };

std::ostream& operator<<(std::ostream& os, Floor floor) {
    return os << (floor == Floor::Clamp ? "Clamp" : floor == Floor::MaxMin ? "MaxMin" : "Max");
}
std::ostream& operator<<(std::ostream& os, Inverse inverse) {
    return os << (inverse == Inverse::Divide ? "Divide" : "PowerMinusOne");
}

double type_max(ov::element::Type type) {
    return type == ov::element::f16 ? 65504.0 : static_cast<double>(std::numeric_limits<float>::max());
}

ov::PartialShape channel_shape(size_t rank) {
    return rank == 4 ? ov::PartialShape{2, channels, 6, 10} : ov::PartialShape{1, channels, 3, 6, 10};
}

// {1, C, 1, ...} for a rank-`rank` tensor.
ov::Shape per_channel_shape(size_t rank) {
    ov::Shape shape(rank, 1);
    shape[1] = channels;
    return shape;
}

// Non-uniform gamma 0.5 + 0.01 * c, rounded to `type` the way a Constant of that type stores it.
std::vector<float> gamma_values(ov::element::Type type) {
    std::vector<float> values(channels);
    for (size_t c = 0; c < channels; ++c) {
        const float value = 0.5f + 0.01f * static_cast<float>(c);
        values[c] = type == ov::element::f16 ? static_cast<float>(ov::float16(value)) : value;
    }
    return values;
}

// A Constant of `type`, or a f16 Constant behind a decompression Convert to `type`.
ov::Output<ov::Node> make_const(ov::element::Type type, const ov::Shape& shape, const std::vector<float>& values, bool decompress) {
    if (!decompress) {
        return v0::Constant::create(type, shape, values);
    }
    return std::make_shared<v0::Convert>(v0::Constant::create(ov::element::f16, shape, values), type);
}

struct ChannelL2Norm {
    Floor floor = Floor::Clamp;
    Inverse inverse = Inverse::Divide;
    std::vector<int64_t> axes = {1};
    bool keep_dims = true;
    bool broadcast = true;
    std::optional<double> clip;  // largest finite value of the element type by default
    std::optional<double> eps;   // negligible_eps of the element type by default
};

// eps as the pass reads it: a Clamp keeps the double attribute, a Maximum Constant rounds it to its element type
// (1e-12 underflows to 0 in f16).
double floor_eps(ov::element::Type type, Floor floor, double eps) {
    return floor == Floor::Clamp ? eps : v0::Constant::create(type, {}, {static_cast<float>(eps)})->cast_vector<double>()[0];
}

// x / Broadcast(max(ReduceL2(x, axis 1), eps) [clipped]), as torch's F.normalize(x, dim=1) exports it.
ov::Output<ov::Node> l2_norm_channel(const ov::Output<ov::Node>& x, const ChannelL2Norm& cfg = {}) {
    const auto type = x.get_element_type();
    const double clip = cfg.clip.value_or(type_max(type));
    const double eps = cfg.eps.value_or(negligible_eps(type));
    auto axes = v0::Constant::create(ov::element::i64, {cfg.axes.size()}, cfg.axes);
    ov::Output<ov::Node> norm = std::make_shared<v4::ReduceL2>(x, axes, cfg.keep_dims);
    if (cfg.floor == Floor::Clamp) {
        norm = std::make_shared<v0::Clamp>(norm, eps, clip);
    } else {
        norm = std::make_shared<v1::Maximum>(norm, v0::Constant::create(type, {}, {static_cast<float>(eps)}));
        if (cfg.floor == Floor::MaxMin) {
            norm = std::make_shared<v1::Minimum>(norm, v0::Constant::create(type, {}, {clip}));
        }
    }
    if (cfg.broadcast) {
        // ShapeOf(x) rather than a constant shape, as torch exports it; this also works for dynamic shapes.
        norm = std::make_shared<v3::Broadcast>(norm, std::make_shared<v3::ShapeOf>(x));
    }
    if (cfg.inverse == Inverse::Power) {
        return std::make_shared<v1::Multiply>(x, std::make_shared<v1::Power>(norm, v0::Constant::create(type, {}, {-1.f})));
    }
    return std::make_shared<v1::Divide>(x, norm);
}

// gamma' = 1/sqrt(C) [* scale] [* gamma], in the f32 operation order the pass folds it in.
std::vector<float> fused_gamma(std::optional<float> scale, const std::optional<std::vector<float>>& gamma) {
    float base = static_cast<float>(1.0 / std::sqrt(static_cast<double>(channels)));
    if (scale) {
        base *= *scale;
    }
    std::vector<float> values(channels, base);
    if (gamma) {
        for (size_t c = 0; c < channels; ++c) {
            values[c] = base * (*gamma)[c];
        }
    }
    return values;
}

std::shared_ptr<ov::Node> channel_rms(const ov::Output<ov::Node>& x, const ov::Output<ov::Node>& gamma, std::optional<double> eps = std::nullopt) {
    const double value = eps.value_or(negligible_eps(x.get_element_type()));
    return std::make_shared<ov::op::internal::RMS>(x, gamma, value * value / static_cast<double>(channels), x.get_element_type(), 1);
}

std::shared_ptr<ov::Node> channel_rms(const ov::Output<ov::Node>& x, const std::vector<float>& gamma, std::optional<double> eps = std::nullopt) {
    return channel_rms(x, v0::Constant::create(x.get_element_type(), ov::Shape{channels}, gamma), eps);
}

}  // namespace

// Fixture for single-case tests: comparator also checks constant values (fused gamma) and attributes (eps, axis).
class L2NormToRMSTest : public TransformationTestsF {
public:
    L2NormToRMSTest() {
        comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
        comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<L2NormToRMS>(1024);
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// Last-axis tests
// ---------------------------------------------------------------------------------------------------------------------

TEST_F(L2NormToRMSTest, LastAxis) {
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, hidden});
        auto gamma = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{hidden});
        model = std::make_shared<ov::Model>(ov::OutputVector{l2_norm(x, gamma)}, ov::ParameterVector{x, gamma});
    }
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, hidden});
        auto gamma = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{hidden});
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{last_axis_rms(x, gamma, ov::element::f32)}, ov::ParameterVector{x, gamma});
    }
}

// f16 activations normalized in f32 and converted back: RMS takes the f16 input and writes f16 directly.
TEST_F(L2NormToRMSTest, LastAxisWithConverts) {
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, hidden});
        auto gamma = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{hidden});
        auto x_f32 = std::make_shared<v0::Convert>(x, ov::element::f32);
        auto out = std::make_shared<v0::Convert>(l2_norm(x_f32, gamma), ov::element::f16);
        model = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{x, gamma});
    }
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, hidden});
        auto gamma = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{hidden});
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{last_axis_rms(x, gamma, ov::element::f16)}, ov::ParameterVector{x, gamma});
    }
}

// The Multiply also feeds something else than the trailing Convert: RMS replaces the Multiply and the Convert stays.
TEST_F(L2NormToRMSTest, LastAxisConvertNotSoleConsumer) {
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, hidden});
        auto gamma = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{hidden});
        auto norm = l2_norm(x, gamma);
        auto out = std::make_shared<v0::Convert>(norm, ov::element::f16);
        model = std::make_shared<ov::Model>(ov::OutputVector{out, norm}, ov::ParameterVector{x, gamma});
    }
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, hidden});
        auto gamma = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{hidden});
        auto rms = last_axis_rms(x, gamma, ov::element::f32);
        auto out = std::make_shared<v0::Convert>(rms, ov::element::f16);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{out, rms}, ov::ParameterVector{x, gamma});
    }
}

// gamma {1, 1, n} still scales the normalized axis only: fused.
TEST_F(L2NormToRMSTest, LastAxisGammaWithLeadingOnes) {
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, hidden});
        auto gamma = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 1, hidden});
        model = std::make_shared<ov::Model>(ov::OutputVector{l2_norm(x, gamma)}, ov::ParameterVector{x, gamma});
    }
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, hidden});
        auto gamma = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 1, hidden});
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{last_axis_rms(x, gamma, ov::element::f32)}, ov::ParameterVector{x, gamma});
    }
}

// Not normalized over the last axis: left alone.
TEST_F(L2NormToRMSTest, LastAxisNotLastAxis) {
    auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, hidden, hidden});
    auto gamma = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{hidden, 1});
    model = std::make_shared<ov::Model>(ov::OutputVector{l2_norm(x, gamma, 1)}, ov::ParameterVector{x, gamma});
}

// Without keep_dims the reduced tensor does not broadcast against x the way RMS needs: left alone.
TEST_F(L2NormToRMSTest, LastAxisNoKeepDims) {
    auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{hidden});
    auto gamma = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{hidden});
    model = std::make_shared<ov::Model>(ov::OutputVector{l2_norm(x, gamma, -1, false)}, ov::ParameterVector{x, gamma});
}

// gamma {n, 1} has n elements but scales each row, not the normalized axis: left alone.
TEST_F(L2NormToRMSTest, LastAxisGammaColumn) {
    auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, hidden, hidden});
    auto gamma = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{hidden, 1});
    model = std::make_shared<ov::Model>(ov::OutputVector{l2_norm(x, gamma)}, ov::ParameterVector{x, gamma});
}

// gamma with more dims than x would change the output rank: left alone.
TEST_F(L2NormToRMSTest, LastAxisGammaHigherRank) {
    auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, hidden});
    auto gamma = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 1, hidden});
    model = std::make_shared<ov::Model>(ov::OutputVector{l2_norm(x, gamma)}, ov::ParameterVector{x, gamma});
}

// Rank 6 has no RMS implementation on GPU: left alone.
TEST_F(L2NormToRMSTest, LastAxisRankSix) {
    auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 1, 1, 4, hidden});
    auto gamma = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{hidden});
    model = std::make_shared<ov::Model>(ov::OutputVector{l2_norm(x, gamma)}, ov::ParameterVector{x, gamma});
}

// A row too long for one work group of the RMS kernel (hidden / 8 > max work group size): left alone.
TEST_F(TransformationTestsF, L2NormToRMSLastAxisRowTooLong) {
    auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, hidden});
    auto gamma = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{hidden});
    model = std::make_shared<ov::Model>(ov::OutputVector{l2_norm(x, gamma)}, ov::ParameterVector{x, gamma});
    manager.register_pass<L2NormToRMS>(4);
}

// ---------------------------------------------------------------------------------------------------------------------
// Channel-axis tests: the full chain x / norm * s * gamma + 0 over ranks, precisions and op variants
// ---------------------------------------------------------------------------------------------------------------------

using ChannelParams = std::tuple<size_t /* rank */, ov::element::Type, Floor, Inverse>;

class L2NormToRMSChannelTest : public testing::WithParamInterface<ChannelParams>, public TransformationTestsF {
public:
    L2NormToRMSChannelTest() : TransformationTestsF() {
        comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
        comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ChannelParams>& info) {
        const auto& [rank, type, floor, inverse] = info.param;
        std::ostringstream ss;
        ss << "rank" << rank << "_" << type << "_" << floor << "_" << inverse;
        return ss.str();
    }

    static std::shared_ptr<ov::Model> getModel(size_t rank, ov::element::Type type, Floor floor, Inverse inverse) {
        auto x = std::make_shared<v0::Parameter>(type, channel_shape(rank));
        ChannelL2Norm cfg;
        cfg.floor = floor;
        cfg.inverse = inverse;
        auto normalized = l2_norm_channel(x, cfg);
        auto scaled = std::make_shared<v1::Multiply>(normalized, v0::Constant::create(type, {}, {scale_value}));
        auto weighted = std::make_shared<v1::Multiply>(scaled, v0::Constant::create(type, per_channel_shape(rank), gamma_values(type)));
        auto out = std::make_shared<v1::Add>(weighted, v0::Constant::create(type, {}, {0.f}));
        return std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{x});
    }

    static std::shared_ptr<ov::Model> getModelRef(size_t rank, ov::element::Type type, Floor floor) {
        auto x = std::make_shared<v0::Parameter>(type, channel_shape(rank));
        auto rms = channel_rms(x, fused_gamma(scale_value, gamma_values(type)), floor_eps(type, floor, negligible_eps(type)));
        return std::make_shared<ov::Model>(ov::OutputVector{rms}, ov::ParameterVector{x});
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        const auto& [rank, type, floor, inverse] = GetParam();
        model = getModel(rank, type, floor, inverse);
        model_ref = getModelRef(rank, type, floor);
        manager.register_pass<L2NormToRMS>(1024);
    }
};

TEST_P(L2NormToRMSChannelTest, FusesFullChain) {}

INSTANTIATE_TEST_SUITE_P(L2NormToRMSChannel,
                         L2NormToRMSChannelTest,
                         ::testing::Combine(::testing::Values(4, 5),
                                            ::testing::Values(ov::element::f32, ov::element::f16),
                                            ::testing::Values(Floor::Clamp, Floor::MaxMin, Floor::Max),
                                            ::testing::Values(Inverse::Divide, Inverse::Power)),
                         L2NormToRMSChannelTest::getTestCaseName);

// ---------------------------------------------------------------------------------------------------------------------
// Channel-axis tests: optional parts of the chain and corner cases
// ---------------------------------------------------------------------------------------------------------------------
TEST_F(L2NormToRMSTest, ChannelNoScaleNoGamma) {
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, channel_shape(4));
        model = std::make_shared<ov::Model>(ov::OutputVector{l2_norm_channel(x)}, ov::ParameterVector{x});
    }
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, channel_shape(4));
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{channel_rms(x, fused_gamma(std::nullopt, std::nullopt))}, ov::ParameterVector{x});
    }
}

TEST_F(L2NormToRMSTest, ChannelGammaWithoutLeadingOne) {
    const auto gamma = gamma_values(ov::element::f32);
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, channel_shape(4));
        auto out = std::make_shared<v1::Multiply>(l2_norm_channel(x), v0::Constant::create(ov::element::f32, {channels, 1, 1}, gamma));
        model = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{x});
    }
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, channel_shape(4));
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{channel_rms(x, fused_gamma(std::nullopt, gamma))}, ov::ParameterVector{x});
    }
}

TEST_F(L2NormToRMSTest, ChannelDecompressedConstants) {
    const auto gamma = gamma_values(ov::element::f16);
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, channel_shape(4));
        ChannelL2Norm cfg;
        cfg.floor = Floor::MaxMin;
        auto scaled = std::make_shared<v1::Multiply>(l2_norm_channel(x, cfg), make_const(ov::element::f32, {}, {scale_value}, true));
        auto weighted = std::make_shared<v1::Multiply>(scaled, make_const(ov::element::f32, per_channel_shape(4), gamma, true));
        auto out = std::make_shared<v1::Add>(weighted, make_const(ov::element::f32, {}, {0.f}, true));
        model = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{x});
    }
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, channel_shape(4));
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{channel_rms(x, fused_gamma(scale_value, gamma))}, ov::ParameterVector{x});
    }
}

TEST_F(L2NormToRMSTest, ChannelNonConstGamma) {
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, channel_shape(4));
        auto gamma = std::make_shared<v0::Parameter>(ov::element::f32, per_channel_shape(4));
        auto scaled = std::make_shared<v1::Multiply>(l2_norm_channel(x), v0::Constant::create(ov::element::f32, {}, {scale_value}));
        auto out = std::make_shared<v1::Multiply>(scaled, gamma);
        model = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{x, gamma});
    }
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, channel_shape(4));
        auto gamma = std::make_shared<v0::Parameter>(ov::element::f32, per_channel_shape(4));
        auto scale = v0::Constant::create(ov::element::f32, {}, {fused_gamma(scale_value, std::nullopt)[0]});
        auto fused = std::make_shared<v1::Reshape>(std::make_shared<v1::Multiply>(scale, gamma), v0::Constant::create(ov::element::i64, {1}, {-1}), false);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{channel_rms(x, fused->output(0))}, ov::ParameterVector{x, gamma});
    }
}

TEST_F(L2NormToRMSTest, ChannelNegativeAxis) {
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, channel_shape(4));
        ChannelL2Norm cfg;
        cfg.axes = {-3};
        model = std::make_shared<ov::Model>(ov::OutputVector{l2_norm_channel(x, cfg)}, ov::ParameterVector{x});
    }
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, channel_shape(4));
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{channel_rms(x, fused_gamma(std::nullopt, std::nullopt))}, ov::ParameterVector{x});
    }
}

TEST_F(L2NormToRMSTest, ChannelDynamicSpatial) {
    const auto gamma = gamma_values(ov::element::f16);
    const ov::PartialShape shape{-1, channels, -1, -1};
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f16, shape);
        auto out = std::make_shared<v1::Multiply>(l2_norm_channel(x), v0::Constant::create(ov::element::f16, per_channel_shape(4), gamma));
        model = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{x});
    }
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f16, shape);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{channel_rms(x, fused_gamma(std::nullopt, gamma))}, ov::ParameterVector{x});
    }
}

TEST_F(L2NormToRMSTest, ChannelWithoutBroadcast) {
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, channel_shape(5));
        ChannelL2Norm cfg;
        cfg.broadcast = false;
        model = std::make_shared<ov::Model>(ov::OutputVector{l2_norm_channel(x, cfg)}, ov::ParameterVector{x});
    }
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, channel_shape(5));
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{channel_rms(x, fused_gamma(std::nullopt, std::nullopt))}, ov::ParameterVector{x});
    }
}

TEST_F(L2NormToRMSTest, ChannelV1BroadcastConstantTarget) {
    const auto shape = channel_shape(4).to_shape();
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto norm = std::make_shared<v4::ReduceL2>(x, v0::Constant::create(ov::element::i64, {1}, {1}), true);
        auto clamp = std::make_shared<v0::Clamp>(norm, negligible_eps(ov::element::f32), type_max(ov::element::f32));
        auto bcast = std::make_shared<v1::Broadcast>(clamp, v0::Constant::create(ov::element::i64, {shape.size()}, shape));
        model = std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<v1::Divide>(x, bcast)}, ov::ParameterVector{x});
    }
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{channel_rms(x, fused_gamma(std::nullopt, std::nullopt))}, ov::ParameterVector{x});
    }
}

TEST_F(L2NormToRMSTest, ChannelNonZeroBiasStaysOutside) {
    const auto gamma = gamma_values(ov::element::f32);
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, channel_shape(4));
        auto weighted = std::make_shared<v1::Multiply>(l2_norm_channel(x), v0::Constant::create(ov::element::f32, per_channel_shape(4), gamma));
        auto out = std::make_shared<v1::Add>(weighted, v0::Constant::create(ov::element::f32, {}, {0.1f}));
        model = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{x});
    }
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, channel_shape(4));
        auto out = std::make_shared<v1::Add>(channel_rms(x, fused_gamma(std::nullopt, gamma)), v0::Constant::create(ov::element::f32, {}, {0.1f}));
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{x});
    }
}

TEST_F(L2NormToRMSTest, ChannelSpatialFactorStaysOutside) {
    const auto shape = channel_shape(4).to_shape();
    const ov::Shape factor_shape{1, 1, shape[2], shape[3]};
    const std::vector<float> factor(ov::shape_size(factor_shape), 2.f);
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto out = std::make_shared<v1::Multiply>(l2_norm_channel(x), v0::Constant::create(ov::element::f32, factor_shape, factor));
        model = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{x});
    }
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto rms = channel_rms(x, fused_gamma(std::nullopt, std::nullopt));
        auto out = std::make_shared<v1::Multiply>(rms, v0::Constant::create(ov::element::f32, factor_shape, factor));
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{x});
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Channel-axis negative tests
// ---------------------------------------------------------------------------------------------------------------------

// A clip below the largest finite value of the element type can change the result, so it must stay.
TEST_F(L2NormToRMSTest, ChannelClipBelowTypeMax) {
    auto x = std::make_shared<v0::Parameter>(ov::element::f16, channel_shape(4));
    ChannelL2Norm cfg;
    cfg.clip = 32000.0;
    model = std::make_shared<ov::Model>(ov::OutputVector{l2_norm_channel(x, cfg)}, ov::ParameterVector{x});
}

// Regression: a f16-maximum clip is not a no-op for a f32 norm, which can exceed 65504.
TEST_F(L2NormToRMSTest, ChannelF16MaxClampOnF32) {
    auto x = std::make_shared<v0::Parameter>(ov::element::f32, channel_shape(4));
    ChannelL2Norm cfg;
    cfg.clip = 65504.0;
    model = std::make_shared<ov::Model>(ov::OutputVector{l2_norm_channel(x, cfg)}, ov::ParameterVector{x});
}

TEST_F(L2NormToRMSTest, ChannelF16MaxMinimumOnF32) {
    auto x = std::make_shared<v0::Parameter>(ov::element::f32, channel_shape(4));
    ChannelL2Norm cfg;
    cfg.floor = Floor::MaxMin;
    cfg.clip = 65504.0;
    model = std::make_shared<ov::Model>(ov::OutputVector{l2_norm_channel(x, cfg)}, ov::ParameterVector{x});
}

// Same, with the f16 clip decompressed to f32 as a compressed f32 model carries it.
TEST_F(L2NormToRMSTest, ChannelDecompressedF16MaxMinimumOnF32) {
    auto x = std::make_shared<v0::Parameter>(ov::element::f32, channel_shape(4));
    auto norm = std::make_shared<v4::ReduceL2>(x, v0::Constant::create(ov::element::i64, {1}, {1}), true);
    auto max = std::make_shared<v1::Maximum>(norm, v0::Constant::create(ov::element::f32, {}, {negligible_eps(ov::element::f32)}));
    auto min = std::make_shared<v1::Minimum>(max, make_const(ov::element::f32, {}, {65504.f}, true));
    model = std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<v1::Divide>(x, min)}, ov::ParameterVector{x});
}

// RMS maps axis 1 to the feature axis for rank-4/5 tensors only.
TEST_F(L2NormToRMSTest, ChannelRankThree) {
    auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, channels, 28});
    model = std::make_shared<ov::Model>(ov::OutputVector{l2_norm_channel(x)}, ov::ParameterVector{x});
}

// C goes into gamma and eps, so it must be static.
TEST_F(L2NormToRMSTest, ChannelDynamicChannels) {
    auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 8, 8});
    model = std::make_shared<ov::Model>(ov::OutputVector{l2_norm_channel(x)}, ov::ParameterVector{x});
}

// Without keep_dims the norm is not a per-pixel divisor of the RMS form.
TEST_F(L2NormToRMSTest, ChannelNoKeepDims) {
    // Batch 1, so the reduced {1, H, W} still numpy-broadcasts to {1, C, H, W}.
    const ov::Shape shape{1, channels, 6, 10};
    auto x = std::make_shared<v0::Parameter>(ov::element::f32, shape);
    auto norm = std::make_shared<v4::ReduceL2>(x, v0::Constant::create(ov::element::i64, {1}, {1}), false);
    auto clamp = std::make_shared<v0::Clamp>(norm, negligible_eps(ov::element::f32), type_max(ov::element::f32));
    auto bcast = std::make_shared<v1::Broadcast>(clamp, v0::Constant::create(ov::element::i64, {shape.size()}, shape));
    model = std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<v1::Divide>(x, bcast)}, ov::ParameterVector{x});
}

// Normalized over a spatial axis: not RMS on axis 1.
TEST_F(L2NormToRMSTest, ChannelSpatialAxis) {
    auto x = std::make_shared<v0::Parameter>(ov::element::f32, channel_shape(4));
    ChannelL2Norm cfg;
    cfg.axes = {2};
    model = std::make_shared<ov::Model>(ov::OutputVector{l2_norm_channel(x, cfg)}, ov::ParameterVector{x});
}

// Normalized over the channel and a spatial axis: not RMS on axis 1.
TEST_F(L2NormToRMSTest, ChannelAndSpatialAxes) {
    auto x = std::make_shared<v0::Parameter>(ov::element::f32, channel_shape(4));
    ChannelL2Norm cfg;
    cfg.axes = {1, 2};
    model = std::make_shared<ov::Model>(ov::OutputVector{l2_norm_channel(x, cfg)}, ov::ParameterVector{x});
}

// The norm is used elsewhere too: fusing would not remove the reduction.
TEST_F(L2NormToRMSTest, ChannelNormHasOtherConsumer) {
    auto x = std::make_shared<v0::Parameter>(ov::element::f32, channel_shape(4));
    auto norm = std::make_shared<v4::ReduceL2>(x, v0::Constant::create(ov::element::i64, {1}, {1}), true);
    auto clamp = std::make_shared<v0::Clamp>(norm, negligible_eps(ov::element::f32), type_max(ov::element::f32));
    auto div = std::make_shared<v1::Divide>(x, clamp);
    model = std::make_shared<ov::Model>(ov::OutputVector{div, norm}, ov::ParameterVector{x});
}

// The normalized x is used elsewhere too: the gamma Multiply stays outside, and RMS replaces the normalization.
TEST_F(L2NormToRMSTest, ChannelNormalizedHasOtherConsumer) {
    const auto gamma = gamma_values(ov::element::f32);
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, channel_shape(4));
        auto normalized = l2_norm_channel(x);
        auto weighted = std::make_shared<v1::Multiply>(normalized, v0::Constant::create(ov::element::f32, per_channel_shape(4), gamma));
        model = std::make_shared<ov::Model>(ov::OutputVector{weighted, normalized}, ov::ParameterVector{x});
    }
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, channel_shape(4));
        auto rms = channel_rms(x, fused_gamma(std::nullopt, std::nullopt));
        auto weighted = std::make_shared<v1::Multiply>(rms, v0::Constant::create(ov::element::f32, per_channel_shape(4), gamma));
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{weighted, rms}, ov::ParameterVector{x});
    }
}

// Only the root may have several consumers.
TEST_F(L2NormToRMSTest, ChannelRootHasTwoConsumers) {
    const auto gamma = gamma_values(ov::element::f32);
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, channel_shape(4));
        auto weighted = std::make_shared<v1::Multiply>(l2_norm_channel(x), v0::Constant::create(ov::element::f32, per_channel_shape(4), gamma));
        model = std::make_shared<ov::Model>(ov::OutputVector{weighted, weighted}, ov::ParameterVector{x});
    }
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, channel_shape(4));
        auto rms = channel_rms(x, fused_gamma(std::nullopt, gamma));
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rms, rms}, ov::ParameterVector{x});
    }
}

// The Broadcast expands x itself, so the result does not have the shape of RMS(x).
TEST_F(L2NormToRMSTest, ChannelBroadcastExpandsX) {
    const ov::Shape x_shape{1, channels, 1, 10};
    const ov::Shape target{1, channels, 6, 10};
    auto x = std::make_shared<v0::Parameter>(ov::element::f32, x_shape);
    auto norm = std::make_shared<v4::ReduceL2>(x, v0::Constant::create(ov::element::i64, {1}, {1}), true);
    auto clamp = std::make_shared<v0::Clamp>(norm, negligible_eps(ov::element::f32), type_max(ov::element::f32));
    auto bcast = std::make_shared<v3::Broadcast>(clamp, v0::Constant::create(ov::element::i64, {target.size()}, target));
    model = std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<v1::Divide>(x, bcast)}, ov::ParameterVector{x});
}

// ---------------------------------------------------------------------------------------------------------------------
// eps gate: the floor max(||x||_2, eps) is fused only for an eps negligible in the element type of the norm
// ---------------------------------------------------------------------------------------------------------------------

// The f16 model form: a f32 eps Constant decompressed to the f16 Maximum. The pass reads the f32 value (1e-12).
TEST_F(L2NormToRMSTest, ChannelF16DecompressedF32Eps) {
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f16, channel_shape(4));
        auto norm = std::make_shared<v4::ReduceL2>(x, v0::Constant::create(ov::element::i64, {1}, {1}), true);
        auto eps = std::make_shared<v0::Convert>(v0::Constant::create(ov::element::f32, {}, {1e-12f}), ov::element::f16);
        auto max = std::make_shared<v1::Maximum>(norm, eps);
        model = std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<v1::Divide>(x, max)}, ov::ParameterVector{x});
    }
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f16, channel_shape(4));
        auto rms = channel_rms(x, fused_gamma(std::nullopt, std::nullopt), static_cast<double>(1e-12f));
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rms}, ov::ParameterVector{x});
    }
}

// eps exactly at the f16 limit, denorm_min(f16) / 1024 = 2^-34: fused.
TEST_F(L2NormToRMSTest, ChannelF16EpsAtLimit) {
    const double eps = std::ldexp(1.0, -34);
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f16, channel_shape(4));
        ChannelL2Norm cfg;
        cfg.eps = eps;
        model = std::make_shared<ov::Model>(ov::OutputVector{l2_norm_channel(x, cfg)}, ov::ParameterVector{x});
    }
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f16, channel_shape(4));
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{channel_rms(x, fused_gamma(std::nullopt, std::nullopt), eps)}, ov::ParameterVector{x});
    }
}

struct EpsNotNegligibleCase {
    std::string name;
    ov::element::Type type;
    Floor floor;
    double eps;
};

class L2NormToRMSChannelEpsNotNegligibleTest : public testing::WithParamInterface<EpsNotNegligibleCase>, public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        const auto& p = GetParam();
        auto x = std::make_shared<v0::Parameter>(p.type, channel_shape(4));
        ChannelL2Norm cfg;
        cfg.floor = p.floor;
        cfg.eps = p.eps;
        model = std::make_shared<ov::Model>(ov::OutputVector{l2_norm_channel(x, cfg)}, ov::ParameterVector{x});
        manager.register_pass<L2NormToRMS>(1024);
    }
};

// The floor would differ from RMS's addend by more than the element type resolves: left alone.
TEST_P(L2NormToRMSChannelEpsNotNegligibleTest, LeftAlone) {}

INSTANTIATE_TEST_SUITE_P(L2NormToRMSChannel,
                         L2NormToRMSChannelEpsNotNegligibleTest,
                         ::testing::Values(EpsNotNegligibleCase{"f32_Clamp_1em12", ov::element::f32, Floor::Clamp, 1e-12},
                                           EpsNotNegligibleCase{"f32_Max_1em12", ov::element::f32, Floor::Max, 1e-12},
                                           EpsNotNegligibleCase{"f32_MaxMin_1em12", ov::element::f32, Floor::MaxMin, 1e-12},
                                           EpsNotNegligibleCase{"f16_Clamp_just_above_limit", ov::element::f16, Floor::Clamp, std::ldexp(1.0, -33)},
                                           EpsNotNegligibleCase{"f16_Clamp_1em4", ov::element::f16, Floor::Clamp, 1e-4},
                                           EpsNotNegligibleCase{"f16_Max_1em4", ov::element::f16, Floor::Max, 1e-4},
                                           EpsNotNegligibleCase{"f32_Clamp_negative", ov::element::f32, Floor::Clamp, -1.0},
                                           EpsNotNegligibleCase{"f32_Max_inf", ov::element::f32, Floor::Max, std::numeric_limits<double>::infinity()}),
                         [](const testing::TestParamInfo<EpsNotNegligibleCase>& info) {
                             return info.param.name;
                         });

// ---------------------------------------------------------------------------------------------------------------------
// auto_broadcast: only numpy-broadcasting eltwise ops are taken in
// ---------------------------------------------------------------------------------------------------------------------

// A PDPD-broadcast gamma Multiply is not folded into gamma; the numpy chain below it still is.
TEST_F(L2NormToRMSTest, ChannelPdpdGammaStaysOutside) {
    const auto gamma = gamma_values(ov::element::f32);
    const ov::op::AutoBroadcastSpec pdpd(ov::op::AutoBroadcastType::PDPD);
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, channel_shape(4));
        auto out = std::make_shared<v1::Multiply>(l2_norm_channel(x), v0::Constant::create(ov::element::f32, per_channel_shape(4), gamma), pdpd);
        model = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{x});
    }
    {
        auto x = std::make_shared<v0::Parameter>(ov::element::f32, channel_shape(4));
        auto rms = channel_rms(x, fused_gamma(std::nullopt, std::nullopt));
        auto out = std::make_shared<v1::Multiply>(rms, v0::Constant::create(ov::element::f32, per_channel_shape(4), gamma), pdpd);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{x});
    }
}

// A PDPD-broadcast gamma Multiply in the last-axis form: left alone.
TEST_F(L2NormToRMSTest, LastAxisPdpdGamma) {
    auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, 4, hidden});
    auto gamma = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{hidden});
    auto out = l2_norm(x, gamma, -1, true, last_axis_eps, ov::op::AutoBroadcastType::PDPD);
    model = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{x, gamma});
}

// Regression: an inf eps is outside the range the callback accepts, so the pattern must not match it (it used to
// assert in the callback).
TEST_F(L2NormToRMSTest, LastAxisInfiniteEps) {
    auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, hidden});
    auto gamma = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{hidden});
    auto out = l2_norm(x, gamma, -1, true, std::numeric_limits<float>::infinity());
    model = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{x, gamma});
}
