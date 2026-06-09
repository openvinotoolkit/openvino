// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "intel_npu/common/blob_reader_interface.hpp"
#include "intel_npu/common/isection.hpp"

namespace intel_npu {

/**
 * @brief Interface that standardizes the evaluation of section types support.
 */
class SectionTypeInstanceEvaluator {
public:
    SectionTypeInstanceEvaluator(const std::shared_ptr<ISection>& section, BlobReaderInterface reader);

    /**
     * @brief Checks whether or not the NPU plugin supports the section type instance.
     * @details After evaluation, the
     * result is stored in "m_supported" for future use.
     */
    bool check_support();

private:
    std::shared_ptr<ISection> m_section;

    BlobReaderInterface m_reader;

    /**
     * @brief If evaluation is performed, the result will be stored here for future use.
     */
    std::optional<bool> m_supported;
};

}  // namespace intel_npu
