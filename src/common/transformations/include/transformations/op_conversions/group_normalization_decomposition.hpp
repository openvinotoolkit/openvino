// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API GroupNormalizationDecomposition;

}  // namespace pass
}  // namespace ov

// This transformation expresses GroupNormalization with a sub-graph of OpenVINO operations
class ov::pass::GroupNormalizationDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GroupNormalizationDecomposition", "0");
    GroupNormalizationDecomposition();
};
