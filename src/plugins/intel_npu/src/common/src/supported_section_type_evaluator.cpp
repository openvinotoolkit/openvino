// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/supported_section_type_evaluator.hpp"

#include <mutex>

namespace {

/**
 * @brief Allows "std::make_shared" to called the protected constructor
 */
struct MakeSharedEnabler : public intel_npu::SupportedSectionTypeEvaluator {};

}  // namespace

namespace intel_npu {

std::shared_ptr<SupportedSectionTypeEvaluator> SupportedSectionTypeEvaluator::get_instance() {
    // No lazy initialization since the object is lightweight
    static std::shared_ptr<SupportedSectionTypeEvaluator> instance = std::make_shared<MakeSharedEnabler>();
    return instance;
}

bool SupportedSectionTypeEvaluator::evaluate() const {
    return true;
}

}  // namespace intel_npu
