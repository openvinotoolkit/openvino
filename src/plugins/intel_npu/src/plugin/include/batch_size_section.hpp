// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>

#include "intel_npu/common/isection.hpp"
#include "openvino/core/layout.hpp"

namespace intel_npu {

class BatchSizeSection final : public ISection {
public:
    BatchSizeSection(const int64_t batch_size, const ov::log::Level log_level = ov::log::Level::WARNING);

    std::vector<CRE::Token> get_compatibility_requirements_subexpression(
        const std::unordered_map<SectionType, std::unordered_map<SectionTypeInstance, std::shared_ptr<ISection>>>&
            all_registered_sections) const override;

    void write(BlobWriterInterface& writer) override;

    int64_t get_batch_size() const;

    static std::shared_ptr<ISection> read(BlobReaderInterface& blob_reader);

private:
    int64_t m_batch_size;

    Logger m_logger;
};

}  // namespace intel_npu
