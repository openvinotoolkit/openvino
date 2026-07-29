// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <tuple>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "ov_ops/fully_connected.hpp"
#include "transformations/pattern_blocks/compressed_weights_block.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertFullyConnectedToFullyConnectedCompressed;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertFullyConnectedToFullyConnectedCompressed : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertFullyConnectedToFullyConnectedCompressed");

    using SupportsPredicate =
        std::function<bool(const std::shared_ptr<ov::op::internal::FullyConnected>&, size_t, size_t, size_t)>;

    ConvertFullyConnectedToFullyConnectedCompressed(const std::vector<ov::element::Type>& supported_activation_types,
                                                    const std::vector<ov::element::Type>& supported_weights_types,
                                                    SupportsPredicate supports_config = nullptr,
                                                    bool convert_u4zp_to_u8 = false);

    /**
     * @brief Processes compressed weights from a pattern block and prepares them for compressed operations.
     *
     * @param weights_block The CompressedWeightsBlock pattern containing the weight compression graph
     * @param pattern_map The pattern value map from the matcher containing matched nodes
     * @param convert_u4zp_to_u8 Flag indicating whether to convert u4 zero points to u8
     * @param has_transpose Flag indicating whether the weights require transpose operation
     * @param grouped Flag indicating whether the compression uses grouped quantization
     * @param batched_weights Flag indicating whether the weights have a batch dimension
     * @param result_nodes Output vector to collect intermediate nodes created during processing
     *
     * @return A tuple containing processed compressed weights, decompression scales, and decompression zero points.
     */
    static std::tuple<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>>
    process_compressed_weights(const std::shared_ptr<ov::pass::pattern::op::CompressedWeightsBlock>& weights_block,
                               const ov::pass::pattern::PatternValueMap& pattern_map,
                               bool convert_u4zp_to_u8,
                               bool has_transpose,
                               bool grouped,
                               bool batched_weights,
                               std::vector<std::shared_ptr<ov::Node>>& result_nodes);
};
