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
    SectionTypeInstanceEvaluator(const std::function<bool(BlobReaderInterface&)>& evaluate_fn,
                                 BlobReaderInterface reader);

    /**
     * @brief Checks whether or not the NPU plugin supports the section type instance.
     * @details After evaluation, the
     * result is stored in "m_supported" for future use.
     */
    bool check_support() const;

    /**
     * @brief Tells whether or not the section type instance has already been evaluated.
     */
    bool evaluated() const;

private:
    std::function<bool(BlobReaderInterface&)> m_evaluate_fn;

    mutable BlobReaderInterface m_reader;

    /**
     * @brief If evaluation is performed, the result will be stored here for future use.
     */
    mutable std::optional<bool> m_supported;
};

}  // namespace intel_npu
