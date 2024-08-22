// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/backward_graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MarkSugraphsToKeepInMixedPrecision;

constexpr auto float16_min_normalized = float16::from_bits(0x0400);

}  // namespace pass
}  // namespace ov

/*
 * @ingroup ov_transformation_common_api
 * @brief: MarkSugraphsToKeepInMixedPrecision container for marking passes which marks subgraphs
 * to be kept in f32 for mixed precision inference. Includes passes for the following patterns:
 * L2Normalize, MVN, ShapeOf subgraphs, Exp in ReduceOp paths and Division with small eps values.
 */
class ov::pass::MarkSugraphsToKeepInMixedPrecision : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("MarkSugraphsToKeepInMixedPrecision", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};
