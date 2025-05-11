// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API UniqueDecomposition;

}  // namespace pass
}  // namespace ov

// This transformation expresses Unique with a sub-graph of OpenVINO operations
class ov::pass::UniqueDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("UniqueDecomposition");
    UniqueDecomposition();
};
