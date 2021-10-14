// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "random_normal.hpp"
#include "default_opset.hpp"
#include "ngraph/opsets/opset8.hpp"


namespace ngraph {
namespace onnx_import {
namespace detail {



OutputVector make_random_normal(const Output<ngraph::Node>& shape, element::Type type, float mean, float scale, float seed)
{
    // ONNX spefifies the seed as a float, but ngraph uses uint64_t
    uint64_t seed_uint64 = static_cast<uint64_t>(seed*1000);
    return box_muller(shape, type, mean, scale, seed_uint64);
}

OutputVector make_random_normal(const Output<ngraph::Node>& shape, element::Type type, float mean, float scale)
{
    return box_muller(shape, type, mean, scale);
}

OutputVector box_muller(const Output<ngraph::Node>& shape, element::Type target_type, float mean, float scale, uint64_t op_seed, uint64_t global_seed)
{
    // We need to use two op_seeds to make sure we get different results for two RandomUniform series
    const uint64_t seed_1 = (op_seed == 0 ? rand() % 10000 : op_seed);
    const uint64_t seed_2 = (op_seed == 0 ? rand() % 10000 : op_seed + 10000);

    const auto min_val = default_opset::Constant::create(target_type, Shape{1}, {0});
    const auto max_val = default_opset::Constant::create(target_type, Shape{1}, {1});

    const auto uniform_1 = std::make_shared<ngraph::opset8::RandomUniform>(shape,
                                                            min_val,
                                                            max_val,
                                                            target_type,
                                                            global_seed,
                                                            seed_1);
    const auto uniform_2 = std::make_shared<ngraph::opset8::RandomUniform>(shape,
                                                            min_val,
                                                            max_val,
                                                            target_type,
                                                            global_seed,
                                                            seed_2);

    // Compute Box–Muller transform
    // random_normal = scale * ng.sqrt(-2.0 * ng.log(uniform_1)) * ng.cos(2.0 * np.pi * uniform_2) + mean
    const auto pi = default_opset::Constant::create(target_type, Shape{1}, {3.141592653589793});
    const auto minus_two = default_opset::Constant::create(target_type, Shape{1}, {-2.0});
    const auto two = default_opset::Constant::create(target_type, Shape{1}, {2.0});

    const auto log = std::make_shared<default_opset::Log>(uniform_1);
    const auto multiply_minus_two_log = std::make_shared<default_opset::Multiply>(log, minus_two);
    const auto sqrt = std::make_shared<default_opset::Sqrt>(multiply_minus_two_log);

    const auto multiply_two_pi = std::make_shared<default_opset::Multiply>(uniform_2, pi);
    const auto multiply_two_pi_uniform_2 = std::make_shared<default_opset::Multiply>(multiply_two_pi, uniform_2);
    auto const cos = std::make_shared<default_opset::Cos>(multiply_two_pi_uniform_2);

    auto const scale_const = default_opset::Constant::create(target_type, Shape{1}, {scale});
    auto const mean_const = default_opset::Constant::create(target_type, Shape{1}, {mean});
    auto const product = std::make_shared<default_opset::Multiply>(scale_const, std::make_shared<default_opset::Multiply>(sqrt, cos));
    auto const sum = std::make_shared<default_opset::Add>(product, mean_const);

    return {sum};
}


}  // namespace detail
}  // namespace onnx_import
}  // namespace ngraph
