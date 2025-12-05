// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertWeightCompressedConv1x1ToMatmul;

class TRANSFORMATIONS_API ConvertWeightCompressedConv1x1ToMatmulMatcher;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Convert Weight Compressed Convolution with 1x1 kernel to MatMul Op.
 *
 */

class ov::pass::ConvertWeightCompressedConv1x1ToMatmulMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertWeightCompressedConv1x1ToMatmulMatcher");
    ConvertWeightCompressedConv1x1ToMatmulMatcher();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Container for all types of WeightCompressedConv1x1 to MatMul convertion.
 *
 */

class ov::pass::ConvertWeightCompressedConv1x1ToMatmul : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ConvertWeightCompressedConv1x1ToMatmul");
    ConvertWeightCompressedConv1x1ToMatmul() {
        add_matcher<ov::pass::ConvertWeightCompressedConv1x1ToMatmulMatcher>();
    }
};
