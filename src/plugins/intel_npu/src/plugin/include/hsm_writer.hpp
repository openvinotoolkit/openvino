// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/blob_sections_registry.hpp"
#include "intel_npu/utils/logger/logger.hpp"

namespace intel_npu::hsm {

class Writer final {
public:
    Writer(const BlobSectionsRegistry& blob_sections_registry);

    void write_to(std::ostream& stream) const;

private:
    // TODO make function instead of class
    std::reference_wrapper<const BlobSectionsRegistry> m_blob_sections_registry;

    Logger m_logger;
};

}  // namespace intel_npu::hsm
