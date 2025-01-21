// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu_memory.h>

#include <string>
#include <utility>

#include "memory_desc/dnnl_blocked_memory_desc.h"

namespace ov {
namespace intel_cpu {

/**
 * Utility class to dump blob contant in plain format.
 * Every layout information will be lost.
 *
 * In case of low precision blob it allow to store
 * with using scaling factors per channel.
 * NB! Channel is a second dimension for all blob types.
 */
class BlobDumper {
    MemoryPtr memory;

    void prepare_plain_data(const MemoryPtr& memory, std::vector<uint8_t>& data) const;

public:
    BlobDumper() = default;
    BlobDumper(const DnnlBlockedMemoryDesc& desc) {
        dnnl::engine eng(dnnl::engine::kind::cpu, 0);
        memory = std::make_shared<Memory>(eng, desc);
    }
    BlobDumper(const BlobDumper&) = default;
    BlobDumper& operator=(BlobDumper&&) = default;

    explicit BlobDumper(MemoryPtr _memory) : memory(std::move(_memory)) {}

    static BlobDumper read(const std::string& file_path);
    static BlobDumper read(std::istream& stream);

    void dump(const std::string& file_path) const;
    void dump(std::ostream& stream) const;

    void dumpAsTxt(const std::string& file_path) const;
    void dumpAsTxt(std::ostream& stream) const;

    void* getDataPtr() const {
        return memory->getData();
    }
};

}  // namespace intel_cpu
}  // namespace ov
