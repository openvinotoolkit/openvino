// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MarkSugraphsToKeepInMixedPrecision;
class TRANSFORMATIONS_API MarkNormalizationOps;
class TRANSFORMATIONS_API MarkExpInReduceOpPath;
class TRANSFORMATIONS_API MarkDivWithEps;

}  // namespace pass
}  // namespace ov

/*
 * @ingroup ie_transformation_common_api
 * @brief: MarkSugraphsToKeepInMixedPrecision container for marking pass including:
 * L2Normalize, MVN, Exp in ReduceOp paths and Division with small eps values
 */
class ov::pass::MarkSugraphsToKeepInMixedPrecision : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("MarkSugraphsToKeepInMixedPrecision", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

/*
 * @ingroup ie_transformation_common_api
 * @brief: MarkNormalizationOps marks MVN and NormalizeL2 to be kept in f32 precision.
 */
class ov::pass::MarkNormalizationOps : public pass::MatcherPass {
public:
    OPENVINO_RTTI("MarkNormalizationOps", "0");
    MarkNormalizationOps();
};

/*
 * @ingroup ie_transformation_common_api
 * @brief: MarkExpInReduceOpPath marks path that goes into ReduceSum and ReduceMean.
 * Values that go from Exp to ReduceSum/ReduceMean are precision
 * sensitive and should be kept in f32 precision for mixed inference.
 */
class ov::pass::MarkExpInReduceOpPath : public ov::pass::BackwardGraphRewrite {
public:
    OPENVINO_RTTI("MarkExpInReduceOpPath", "0");
    MarkExpInReduceOpPath();
};

/*
 * @ingroup ie_transformation_common_api
 * @brief: matches the patterns input_1/Maximum(input_2, eps); input_1/Add(input_2, eps);
 * and input_1*Pow(Maximum[Add](input_2, eps), -z) and marks subgraph root to be kept in fp32.
 *
 * If both input_1 and input_2 simultaneously happen to be zero to prevent from NaNs and not to loose accuracy,
 * we should calculate such patterns always in fp32 precision even if ov::Model is compressed to fp16.
 */
class ov::pass::MarkDivWithEps : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MarkDivWithEps", "0");
    MarkDivWithEps();
};
