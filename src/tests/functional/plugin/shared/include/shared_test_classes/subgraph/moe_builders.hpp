// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <memory>
#include <optional>
#include "openvino/core/model.hpp"
#include "openvino/core/type/element_type.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/subgraph/weights_decompression_params.hpp"
#include "common_test_utils/subgraph_builders/weights_decompression_builders.hpp"

namespace ov {
namespace test {

struct MoePatternParams {
    ov::test::InputShape data_shape;
    size_t topk;
    size_t number_of_experts;
    size_t intermediate_size;
};

std::shared_ptr<ov::Model> initMoE2GeMMSubgraph(const MoePatternParams& moe_params,
                                                 const ov::element::Type data_precision,
                                                 const ov::element::Type weights_precision,
                                                 const bool use_weight_decompression = false,
                                                 const std::optional<ov::element::Type> decompression_precision = std::nullopt,
                                                 const std::optional<ov::element::Type> scale_precision = std::nullopt,
                                                 const std::optional<ov::test::utils::DecompressionType> decompression_multiply_type = std::nullopt,
                                                 const std::optional<ov::test::utils::DecompressionType> decompression_subtract_type = std::nullopt,
                                                 const std::optional<bool> reshape_on_decompression = std::nullopt,
                                                 const std::optional<int> decompression_group_size = std::nullopt);

std::shared_ptr<ov::Model> initMoE3GeMMSubgraph(const MoePatternParams& moe_params,
                                                 const ov::element::Type data_precision,
                                                 const ov::element::Type weights_precision,
                                                 const bool use_weight_decompression = false,
                                                 const std::optional<ov::element::Type> decompression_precision = std::nullopt,
                                                 const std::optional<ov::element::Type> scale_precision = std::nullopt,
                                                 const std::optional<ov::test::utils::DecompressionType> decompression_multiply_type = std::nullopt,
                                                 const std::optional<ov::test::utils::DecompressionType> decompression_subtract_type = std::nullopt,
                                                 const std::optional<bool> reshape_on_decompression = std::nullopt,
                                                 const std::optional<int> decompression_group_size = std::nullopt);

} // namespace test
} // namespace ov