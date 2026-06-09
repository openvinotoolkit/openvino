// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/section_type_instance_evaluator.hpp"

namespace intel_npu {

SectionTypeInstanceEvaluator::SectionTypeInstanceEvaluator(const std::function<bool(BlobReaderInterface&)>& evaluate_fn,
                                                           BlobReaderInterface reader)
    : m_evaluate_fn(evaluate_fn),
      m_reader(std::move(reader)) {}

bool SectionTypeInstanceEvaluator::check_support() const {
    if (m_supported.has_value()) {
        return m_supported.value();
    }

    m_supported = m_evaluate_fn(m_reader);
    return m_supported.value();
}

}  // namespace intel_npu
