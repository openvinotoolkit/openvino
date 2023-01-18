// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MarkExpReduceOpToKeepInMixedPrecision;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief: MarkExpReduceOpToKeepInMixedPrecision  marks path that goes
 * into ReduceSum and ReduceMean. Values that go from Exp to ReduceSum/ReduceMean are precision
 * sensitive and such nodes should be kept in f32 precision for mixed inference.
 */
class ov::pass::MarkExpReduceOpToKeepInMixedPrecision : public ov::pass::BackwardGraphRewrite {
public:
    OPENVINO_RTTI("MarkExpReduceOpToKeepInMixedPrecision", "0");
    MarkExpReduceOpToKeepInMixedPrecision();
};
