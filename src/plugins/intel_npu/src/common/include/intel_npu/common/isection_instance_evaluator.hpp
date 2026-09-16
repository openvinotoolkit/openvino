// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/isection.hpp"
#include "openvino/runtime/properties.hpp"

namespace intel_npu {

/**
 * @brief Abstract class that standardizes the evaluation of section instances.
 * @note The purpose of these kinds of classes is different than the role of `ISectionTypeEvaluator`. The type
 * evaluators check if the SectionType is supported by the software. The instance evaluators evaluate the "compatibility
 * substring" corresponding to some instance.
 * @note There should be a at most one instance of such class per section type. I.e. there is a 1:1 or 0:1 relationshipt
 * between instance evaluators and section types. Inherited classes are meant to instruct how to evaluate the section
 * instances of a given type based on some runtime requirements. Section types that do not have special requirements per
 * instance don't need any inherited instance evaluator.
 * @see `SingleSectionInstanceEvaluator`, the class that wraps this class and stores the evaluation result of individual
 * section instances. There is a 1:1 or 0:1 relationship between one such class instance and SectionIDs.
 * @note Inherited classes may be defined as singletons if their behavior doesn't change during runtime.
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
