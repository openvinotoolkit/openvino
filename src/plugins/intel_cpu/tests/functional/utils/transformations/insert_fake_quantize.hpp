// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/pass/matcher_pass.hpp"
#include "utils/quantization_utils.hpp"

namespace CPUTestUtils {

class InsertFakeQuantize : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("InsertFakeQuantize");
    InsertFakeQuantize(size_t input_id, const QuantizationData& qinfo);
};

}  // namespace CPUTestUtils
