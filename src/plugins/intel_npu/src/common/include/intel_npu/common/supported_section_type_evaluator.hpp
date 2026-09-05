// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/isection_type_evaluator.hpp"

namespace intel_npu {

class SupportedSectionTypeEvaluator final : public ISectionTypeEvaluator {
public:
    SupportedSectionTypeEvaluator(const SectionType section_type);

private:
    bool evaluate() const override;
};

}  // namespace intel_npu
