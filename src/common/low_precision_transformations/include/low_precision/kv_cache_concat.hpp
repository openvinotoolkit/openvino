// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "lpt_visibility.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov::pass::low_precision {
class LP_TRANSFORMATIONS_API KVCacheConcat : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("KVCacheConcat");
    KVCacheConcat(const std::shared_ptr<ov::Model>& model);
};

} // namespace ov::pass::low_precision
