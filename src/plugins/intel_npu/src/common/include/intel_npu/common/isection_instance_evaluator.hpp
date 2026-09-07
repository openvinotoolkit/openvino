// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/isection.hpp"
#include "openvino/runtime/properties.hpp"

namespace intel_npu {

/**
 * @brief Abstract class that standardizes the evaluation of section instances.
 * @note There should be a at most one instance of such class per section type. Inherited classes are meant to instruct
 * how to evaluate the section instances of a given type based on some runtime requirements. Section types that do not
 * have special requirements per instance don't need any inherited instance evaluator.
 * @note TODO different from type evaluator
 * @see `SectionInstanceEvaluator`, the class that wraps this class and stores the evaluation result of every single
 * section instance.
 */
class ISectionInstanceEvaluator {
public:
    ISectionInstanceEvaluator() = default;

    virtual ~ISectionInstanceEvaluator() = default;

    /**
     * @brief Checks whether or not the NPU plugin supports a section instance described by the given runtime
     * requirements
     */
    virtual ov::CompatibilityCheck evaluate(std::string_view runtime_requirements) const = 0;
};

}  // namespace intel_npu
