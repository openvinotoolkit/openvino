// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>

#include "intel_npu/common/isection.hpp"

namespace intel_npu {

class EncryptedSchedulesFlagSection final : public ISection {
public:
    EncryptedSchedulesFlagSection(const bool applied_encryption,
                                  const ov::log::Level log_level = ov::log::Level::WARNING);

    std::vector<CREToken> get_compatibility_requirements_subexpression(
        const std::unordered_map<SectionID, std::shared_ptr<ISection>>& all_registered_sections) const override;

    void write(BlobWriterInterface& writer) override;

    bool get_flag() const;

    static std::shared_ptr<ISection> read(BlobReaderInterface& blob_reader);

private:
    bool m_flag;

    Logger m_logger;
};

}  // namespace intel_npu
