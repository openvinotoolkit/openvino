// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "hsm_format/isection.hpp"

namespace intel_npu {

/**
 * @brief Section containing a flag that indicates whether or not the compiler schedules within the compiled models have
 * been encrypted.
 */
class EncryptedSchedulesFlagSection final : public ISection {
public:
    /**
     * @note The "inline" flag is set to "true", since the serialized payload is shorter than the 24 bytes threshold.
     */
    EncryptedSchedulesFlagSection(const bool applied_encryption);

    /**
     * @brief Returns its own section type as CRE token only when the value of the flag is "true".
     * @details A flag value of "true" means the schedules (or schedule if there's just one) have been encrypted.
     * Therefore, only in this case the section type is returned, to add as requirement the need to support decryption.
     */
    std::vector<std::shared_ptr<CREToken>> get_cre_tokens(
        const std::unordered_map<SectionID, std::shared_ptr<ISection>>& all_registered_sections) const override;

    /**
     * @return A view towards the raw flag value. This is the payload intended to be written in the blob, on
     * behalf of this section.
     */
    ov::util::MemoryView encode() const override;

    bool get_flag() const;

    void set_flag(const bool new_value);

    /**
     * @brief Reads the raw value of the flag to create a new section instance
     */
    static ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>> decode;

private:
    bool m_flag;

    Logger m_logger;
};

}  // namespace intel_npu
