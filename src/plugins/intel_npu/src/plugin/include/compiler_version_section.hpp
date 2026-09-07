// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/isection.hpp"

namespace intel_npu {

class CompilerVersionSection final : public ISection {
public:
    CompilerVersionSection(const int32_t CompilerVersionSection,
                           const ov::log::Level log_level = ov::log::Level::WARNING);

    void write(BlobWriterInterface& writer) override;

    int32_t get_compiler_version() const;

    static std::shared_ptr<ISection> read(BlobReaderInterface& blob_reader);

private:
    int32_t m_compiler_version;

    Logger m_logger;
};

}  // namespace intel_npu
