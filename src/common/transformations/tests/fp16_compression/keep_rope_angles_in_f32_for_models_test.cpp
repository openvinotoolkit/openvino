// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/fp16_compression/keep_rope_angles_in_f32_for_models.hpp"

#include <gtest/gtest.h>

#include <cmath>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

using namespace ov;
using RopeModel = ov::pass::KeepRopeAnglesInF32ForModels::Model;

namespace {

constexpr float pi = 3.14159265358979323846f;
constexpr size_t tokens = 8;
constexpr size_t axes = 3;
constexpr size_t bands = 4;

std::shared_ptr<op::v0::Constant> f32(const Shape& shape, const std::vector<float>& values) {
    return op::v0::Constant::create(element::f32, shape, values);
}

std::shared_ptr<op::v0::Constant> i64(const std::vector<int64_t>& values) {
    return op::v0::Constant::create(element::i64, Shape{values.size()}, values);
}

enum class Freqs { Graph, Folded, NotLtx };

// diffusers LTX2AudioVideoRotaryPosEmbed (split): coords / max_positions -> (grid * 2 - 1) * freqs
// -> transpose(-1, -2) + flatten -> Sin/Cos, applied to a projection. `mark` builds the expected markup.
std::shared_ptr<ov::Model> ltx2_rope(Freqs freqs_kind, bool mark) {
    auto coords = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, tokens, axes});
    auto hidden = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, tokens, axes * bands});
    NodeVector chain;
    auto max_positions = f32({1, 1, axes}, {20.0f, 2048.0f, 2048.0f});
    auto grid = std::make_shared<op::v1::Divide>(coords, max_positions);
    auto axis = i64({-1});
    auto unsqueeze = std::make_shared<op::v0::Unsqueeze>(grid, axis);
    // scalar constants stored in f16 as in a compressed IR
    auto two_f16 = op::v0::Constant::create(element::f16, Shape{1, 1, 1, 1}, {2.0f});
    auto two = std::make_shared<op::v0::Convert>(two_f16, element::f32);
    mark_as_decompression(two);  // not constant-foldable, as in a real compressed IR
    auto scaled = std::make_shared<op::v1::Multiply>(unsqueeze, two);
    auto minus_one = f32({1, 1, 1, 1}, {-1.0f});
    auto centered = std::make_shared<op::v1::Add>(scaled, minus_one);
    chain.insert(chain.end(), {max_positions, grid, axis, unsqueeze, two_f16, two, scaled, minus_one, centered});

    std::shared_ptr<Node> freqs;
    if (freqs_kind == Freqs::Graph) {
        auto theta = f32({}, {10000.0f});
        auto linspace = f32({bands}, {0.0f, 1.0f / 3, 2.0f / 3, 1.0f});
        auto power = std::make_shared<op::v1::Power>(theta, linspace);
        auto pi_c = f32({1}, {pi});
        auto mul = std::make_shared<op::v1::Multiply>(power, pi_c);
        auto half = f32({1}, {2.0f});
        auto div = std::make_shared<op::v1::Divide>(mul, half);
        freqs = div;
        chain.insert(chain.end(), {theta, linspace, power, pi_c, mul, half, div});
    } else {
        std::vector<float> table(bands);
        for (size_t i = 0; i < bands; ++i) {
            const float first = freqs_kind == Freqs::Folded ? pi / 2.0f : 1.0f;
            table[i] = first * std::pow(10000.0f, static_cast<float>(i) / (bands - 1));
        }
        freqs = f32({bands}, table);
        chain.push_back(freqs);
    }
    auto angles = std::make_shared<op::v1::Multiply>(centered, freqs);
    auto order = i64({0, 1, 3, 2});
    auto transpose = std::make_shared<op::v1::Transpose>(angles, order);
    auto flat_shape = i64({0, 0, -1});
    auto flat = std::make_shared<op::v1::Reshape>(transpose, flat_shape, true);
    auto cos = std::make_shared<op::v0::Cos>(flat);
    auto sin = std::make_shared<op::v0::Sin>(flat);
    chain.insert(chain.end(), {angles, order, transpose, flat_shape, flat, cos, sin});

    auto rotated = std::make_shared<op::v1::Add>(std::make_shared<op::v1::Multiply>(hidden, cos),
                                                 std::make_shared<op::v1::Multiply>(hidden, sin));
    if (mark) {
        for (const auto& node : chain)
            disable_conversion(node, element::f16);
        disable_conversion(sin, element::f32, element::bf16);
        disable_conversion(cos, element::f32, element::bf16);
    }
    return std::make_shared<ov::Model>(OutputVector{rotated}, ParameterVector{coords, hidden});
}

// Generic RoPE (positions * inv_freq -> Sin/Cos): not the LTX-2 signature
std::shared_ptr<ov::Model> generic_rope() {
    auto positions = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, tokens, 1});
    auto inv_freq = f32({1, 1, bands}, {1.0f, 0.1f, 0.01f, 0.001f});
    auto angles = std::make_shared<op::v1::Multiply>(positions, inv_freq);
    auto out =
        std::make_shared<op::v1::Add>(std::make_shared<op::v0::Sin>(angles), std::make_shared<op::v0::Cos>(angles));
    return std::make_shared<ov::Model>(OutputVector{out}, ParameterVector{positions});
}

}  // namespace

TEST_F(TransformationTestsF, KeepRopeAnglesInF32ForModels_LTX2_FreqsSubgraph) {
    model = ltx2_rope(Freqs::Graph, false);
    manager.register_pass<pass::KeepRopeAnglesInF32ForModels>(std::vector<RopeModel>{RopeModel::LTX2});
    model_ref = ltx2_rope(Freqs::Graph, true);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}

TEST_F(TransformationTestsF, KeepRopeAnglesInF32ForModels_LTX2_FreqsFolded) {
    model = ltx2_rope(Freqs::Folded, false);
    manager.register_pass<pass::KeepRopeAnglesInF32ForModels>(std::vector<RopeModel>{RopeModel::LTX2});
    model_ref = ltx2_rope(Freqs::Folded, true);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}

TEST_F(TransformationTestsF, KeepRopeAnglesInF32ForModels_ModelNotListed) {
    model = ltx2_rope(Freqs::Graph, false);
    manager.register_pass<pass::KeepRopeAnglesInF32ForModels>(std::vector<RopeModel>{});
    model_ref = ltx2_rope(Freqs::Graph, false);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}

TEST_F(TransformationTestsF, KeepRopeAnglesInF32ForModels_LTX2_OtherFrequenciesNotMatched) {
    model = ltx2_rope(Freqs::NotLtx, false);
    manager.register_pass<pass::KeepRopeAnglesInF32ForModels>(std::vector<RopeModel>{RopeModel::LTX2});
    model_ref = ltx2_rope(Freqs::NotLtx, false);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}

TEST_F(TransformationTestsF, KeepRopeAnglesInF32ForModels_LTX2_GenericRopeNotMatched) {
    model = generic_rope();
    manager.register_pass<pass::KeepRopeAnglesInF32ForModels>(std::vector<RopeModel>{RopeModel::LTX2});
    model_ref = generic_rope();
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}
