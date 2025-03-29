// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define _USE_MATH_DEFINES
#include "openvino/frontend/common/random_normal_helper.hpp"

#include <math.h>

#include "openvino/core/node_vector.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"

namespace ov {
namespace frontend {

OutputVector make_random_normal(pass::NodeRegistry& registry,
                                const Output<Node>& sizes,
                                element::Type target_type,
                                const Output<Node>& mean,
                                const Output<Node>& scale,
                                float seed) {
    // We start by generating two random series from a uniform distribution
    const uint64_t global_seed = 0;

    // ONNX specifies the seed as a float, but OpenVINO uses uint64_t
    // OpenVINO supports only uint64 seeds with a meaningful 0 value (seed will be auto-generated).
    // Because we use a seed as a just meaningful identifier we may
    // just interpret its value as a 32-bit value (float zero value is same with
    // uint32 zero value).
    // Float -0 value will be interpreted as a valid uint32 value.
    const void* seed_ptr = &seed;  // To prevent strict-aliasing error
    const uint64_t op_seed = static_cast<const uint64_t>(*static_cast<const uint32_t*>(seed_ptr));

    // We need to use two op_seeds to make sure we get different results for two RandomUniform series
    // But we also have to keep original logic and pass "0" (auto-generated seed) to RandomUniform
    const uint64_t seed_1 = op_seed;
    const uint64_t seed_2 = (op_seed == 0 ? op_seed : op_seed + 10000);

    auto min_val = registry.make<op::v0::Constant>(target_type, Shape{1}, std::numeric_limits<float>::min());
    auto max_val = registry.make<op::v0::Constant>(target_type, Shape{1}, 1);

    auto uniform_1 = registry.make<op::v8::RandomUniform>(sizes, min_val, max_val, target_type, global_seed, seed_1);
    auto uniform_2 = registry.make<op::v8::RandomUniform>(sizes, min_val, max_val, target_type, global_seed, seed_2);

    // Compute Box–Muller transform
    // random_normal = scale * sqrt(-2.0 * log(uniform_1)) * cos(2.0 * pi * uniform_2) + mean
    auto pi = registry.make<op::v0::Constant>(target_type, Shape{1}, M_PI);
    auto minus_two = registry.make<op::v0::Constant>(target_type, Shape{1}, -2.0);
    auto two = registry.make<op::v0::Constant>(target_type, Shape{1}, 2.0);

    auto log = registry.make<op::v0::Log>(uniform_1);
    auto multiply_minus_two_log = registry.make<op::v1::Multiply>(log, minus_two);
    auto sqrt = registry.make<op::v0::Sqrt>(multiply_minus_two_log);

    auto multiply_2pi = registry.make<op::v1::Multiply>(two, pi);
    auto multiply_2pi_uniform_2 = registry.make<op::v1::Multiply>(multiply_2pi, uniform_2);
    auto cos = registry.make<op::v0::Cos>(multiply_2pi_uniform_2);

    auto sqrt_x_cos = registry.make<op::v1::Multiply>(sqrt, cos);
    auto product = registry.make<op::v1::Multiply>(scale, sqrt_x_cos);
    auto sum = registry.make<op::v1::Add>(product, mean);

    // if we don't disable down-casting then log(float32_min) gives -inf
    disable_fp16_compression(uniform_1);
    disable_fp16_compression(log);

    return {sum};
}

std::pair<OutputVector, pass::NodeRegistry> make_random_normal(const Output<Node>& sizes,
                                                               element::Type target_type,
                                                               const Output<Node>& mean,
                                                               const Output<Node>& scale,
                                                               float seed) {
    pass::NodeRegistry registry;
    OutputVector res = make_random_normal(registry, sizes, target_type, mean, scale, seed);
    return std::make_pair(res, registry);
}

}  // namespace frontend
}  // namespace ov
