// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "lpt_visibility.hpp"
#include "openvino/core/model.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov::pass::low_precision {
class LP_TRANSFORMATIONS_API MoveFakeConvertUpThroughKVCacheConcat : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MoveFakeConvertUpThroughKVCacheConcat");
    MoveFakeConvertUpThroughKVCacheConcat();
};

}  // namespace ov::pass::low_precision
