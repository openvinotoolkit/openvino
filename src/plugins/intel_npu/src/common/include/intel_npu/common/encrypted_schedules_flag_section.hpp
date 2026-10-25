// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>

#include "intel_npu/common/isection.hpp"

namespace intel_npu {

/**
 * @brief Section containing a flag that indicates whether or not the compiler schedules within the compiled models have
 * been encrypted.
 */
class EncryptedSchedulesFlagSection final : public ISection {
public:
    EncryptedSchedulesFlagSection(const bool applied_encryption);

    std::vector<std::shared_ptr<CREToken>> get_compatibility_requirements_subexpression(
        const std::unordered_map<SectionID, std::shared_ptr<ISection>>& all_registered_sections) const override;

    void encode(const ov::runtime::hsm::SectionSink& sink) const override;

    bool get_flag() const;

    static ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>> decode;

private:
    bool m_flag;

    Logger m_logger;
};

}  // namespace intel_npu
