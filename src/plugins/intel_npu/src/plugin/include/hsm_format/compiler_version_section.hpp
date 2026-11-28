// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "hsm_format/isection.hpp"

namespace intel_npu {

class CompilerVersionSection final : public ISection {
public:
    /**
     * @note The "inline" flag is set to "true", since the serialized payload is shorter than the 24 bytes threshold.
     */
    CompilerVersionSection(const int32_t version);

    int32_t get_compiler_version() const;

    /**
     * @returns A view towards the raw compiler version value. This is the payload intended to be written in the blob,
     * on behalf of this section.
     */
    ov::util::MemoryView encode() const override;

    /**
     * @brief Reads the raw compiler version value to create a new section instance.
     */
    static ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>> decode;

private:
    int32_t m_compiler_version;

    Logger m_logger;
};

}  // namespace intel_npu
