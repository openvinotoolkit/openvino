// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

class TRANSFORMATIONS_API VariadicSplitMerge : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("VariadicSplitMerge");
    VariadicSplitMerge();
};

}  // namespace ov::pass
