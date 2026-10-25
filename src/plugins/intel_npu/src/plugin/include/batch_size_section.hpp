// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/isection.hpp"

namespace intel_npu {

class BatchSizeSection final : public ISection {
public:
    BatchSizeSection(const int64_t batch_size);

    std::vector<std::shared_ptr<CREToken>> get_compatibility_requirements_subexpression(
        const std::unordered_map<SectionID, std::shared_ptr<ISection>>& all_registered_sections) const override;

    int64_t get_batch_size() const;

    void encode(const ov::runtime::hsm::SectionSink& sink) const override;

    static ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>> decode;

private:
    int64_t m_batch_size;

    Logger m_logger;
};

}  // namespace intel_npu
