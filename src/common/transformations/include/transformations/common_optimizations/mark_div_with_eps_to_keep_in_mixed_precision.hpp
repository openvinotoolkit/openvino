// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/pass/pattern/matcher.hpp>
#include <transformations_visibility.hpp>
#include <utility>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MarkDivWithEpsToKeepInMixedPrecision;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief: matches the patterns input_1/Maximum(input_2, eps); input_1/Add(input_2, eps);
 * and input_1*Pow(Maximum[Add](input_2, eps), -z) and marks subgraph root to be kept in fp32.
 *
 * If both input_1 and input_2 simultaneously happen to be zero to prevent from NaNs and not to loose accuracy,
 * we should calculate such patterns always in fp32 precision even if ov::Model is compressed to fp16.
 */
class ov::pass::MarkDivWithEpsToKeepInMixedPrecision : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MarkDivWithEpsToKeepInMixedPrecision", "0");
    MarkDivWithEpsToKeepInMixedPrecision();
};
