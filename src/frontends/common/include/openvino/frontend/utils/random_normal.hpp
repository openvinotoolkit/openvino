// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/reshape.hpp"
#include "ngraph/output_vector.hpp"
#include "openvino/frontend/visibility.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#define _USE_MATH_DEFINES
#include <cmath>

namespace ov {
namespace frontend {

/// \brief Creates a random normal tensor with the given shape and type.
/// \details Uses Box-Mueller algorithm to generate random numbers from a Gauassian distribution
/// \param sizes Shape of the output tensor
/// \param target_type Type of the output tensor
/// \param mean Mean of the distribution
/// \param scale Standard deviation of the distribution
/// \param seed Seed for the random number generator
inline std::pair<OutputVector, NodeVector> make_random_normal(const Output<Node>& sizes,
                                                              element::Type target_type,
                                                              const Output<Node>& mean,
                                                              const Output<Node>& scale,
                                                              float seed) {
    namespace default_opset = ov::opset12;
    // We start by generating two random series from a uniform distribution
    const uint64_t global_seed = 0;

    // ONNX specifies the seed as a float, but OpenVINO uses uint64_t
    const auto op_seed = static_cast<uint64_t>(seed * 1000);

    // We need to use two op_seeds to make sure we get different results for two RandomUniform series
    // But we also have to keep original logic and pass "0" (auto-generated seed) to RandomUniform
    const uint64_t seed_1 = op_seed;
    const uint64_t seed_2 = (op_seed == 0 ? op_seed : op_seed + 10000);

    ov::pass::NodeRegistry reg;

    auto min_val = reg.make<ov::op::v0::Constant>(target_type, Shape{1}, std::numeric_limits<float>::min());
    auto max_val = reg.make<default_opset::Constant>(target_type, Shape{1}, 1);

    auto uniform_1 = reg.make<default_opset::RandomUniform>(sizes, min_val, max_val, target_type, global_seed, seed_1);
    auto uniform_2 = reg.make<default_opset::RandomUniform>(sizes, min_val, max_val, target_type, global_seed, seed_2);

    // Compute Box–Muller transform
    // random_normal = scale * sqrt(-2.0 * log(uniform_1)) * cos(2.0 * pi * uniform_2) + mean
    auto pi = reg.make<default_opset::Constant>(target_type, Shape{1}, M_PI);
    auto minus_two = reg.make<default_opset::Constant>(target_type, Shape{1}, -2.0);
    auto two = reg.make<default_opset::Constant>(target_type, Shape{1}, 2.0);

    auto log = reg.make<default_opset::Log>(uniform_1);
    auto multiply_minus_two_log = reg.make<default_opset::Multiply>(log, minus_two);
    auto sqrt = reg.make<default_opset::Sqrt>(multiply_minus_two_log);

    auto multiply_2pi = reg.make<default_opset::Multiply>(two, pi);
    auto multiply_2pi_uniform_2 = reg.make<default_opset::Multiply>(multiply_2pi, uniform_2);
    auto cos = reg.make<default_opset::Cos>(multiply_2pi_uniform_2);

    auto sqrt_x_cos = reg.make<default_opset::Multiply>(sqrt, cos);
    auto product = reg.make<default_opset::Multiply>(scale, sqrt_x_cos);
    auto sum = reg.make<default_opset::Add>(product, mean);

    // if we don't disable down-casting then log(float32_min) gives -inf
    disable_fp16_compression(uniform_1);
    disable_fp16_compression(log);

    return std::make_pair(sum->outputs(), reg.get());
}

}  // namespace frontend
}  // namespace ov
