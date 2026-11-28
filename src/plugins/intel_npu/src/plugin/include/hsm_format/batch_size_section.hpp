// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "hsm_format/isection.hpp"

namespace intel_npu {

/**
 * @brief Section responsible for encoding and decoding the batch size of the original model.
 */
class BatchSizeSection final : public ISection {
public:
    /**
     * @note The "inline" flag is set to "true", since the serialized payload is shorter than the 24 bytes threshold.
     */
    BatchSizeSection(const int64_t batch_size);

    /**
     * @brief Returns its own section type as CRE token. A plugin importer has to known how to hanlde plugin batching if
     * this section exists within the blob.
     */
    std::vector<std::shared_ptr<CREToken>> get_cre_tokens(
        const std::unordered_map<SectionID, std::shared_ptr<ISection>>& all_registered_sections) const override;

    int64_t get_batch_size() const;

    /**
     * @returns A view towards the raw batch size value. This is the payload intended to be written in the blob, on
     * behalf of this section.
     */
    ov::util::MemoryView encode() const override;

    /**
     * @brief Reads the raw batch size value to create a new section instance.
     */
    static ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>> decode;

private:
    int64_t m_batch_size;

    Logger m_logger;
};

}  // namespace intel_npu
