// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/supported_section_type_evaluator.hpp"

#include <mutex>

namespace intel_npu {

std::shared_ptr<SupportedSectionTypeEvaluator> SupportedSectionTypeEvaluator::get_instance() {
    // No lazy initialization since the object is lightweight
    static auto instance = std::make_shared<SupportedSectionTypeEvaluator>();
    return instance;
}

bool SupportedSectionTypeEvaluator::evaluate() const {
    return true;
}

}  // namespace intel_npu
