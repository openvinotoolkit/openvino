// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>

#include "openvino/util/common_util.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov {
namespace npuw {
namespace weights {

// A MappedMemory adapter over an already-resident host buffer (no file, no
// mmap). Used by the buffer-backed weight-sharing path (fd == -1): the weights
// pool lives in a host buffer the caller owns, so instead of mmap'ing a file we
// wrap that region and hand it to NPUW's existing weightless import, which
// consumes any MappedMemory via Weights = SharedBuffer<shared_ptr<MappedMemory>>.
//
// data() points at the pool start so per-constant descriptor offsets (which are
// pool-relative) resolve as data() + offset, exactly like the fd path.
//
// A keep-alive shared_ptr keeps the underlying bytes alive for this object's
// lifetime. It may own the bytes (a copied AlignedBuffer, "B2") or just pin the
// caller's buffer (non-owning, "B1"); either way the region must outlive the
// compiled model.
class HostRegionMemory final : public ov::MappedMemory {
public:
    HostRegionMemory(const void* data, size_t size, std::shared_ptr<void> keepalive)
        : m_data(const_cast<char*>(static_cast<const char*>(data))),
          m_size(size),
          m_keepalive(std::move(keepalive)),
          // Distinct id per region so descriptors stay distinct across models.
          m_id(ov::util::u64_hash_combine(reinterpret_cast<uint64_t>(data), {static_cast<uint64_t>(size)})) {}

    char* data() noexcept override {
        return m_data;
    }
    size_t size() const noexcept override {
        return m_size;
    }
    uint64_t get_id() const noexcept override {
        return m_id;
    }
    // Host memory is already resident; prefetch/evict hints are no-ops.
    void hint_evict(size_t /*offset*/ = 0, size_t /*size*/ = ov::auto_size) noexcept override {}
    void hint_prefetch(size_t /*offset*/ = 0, size_t /*size*/ = ov::auto_size) override {}

private:
    char* m_data = nullptr;
    size_t m_size = 0;
    std::shared_ptr<void> m_keepalive;
    uint64_t m_id = 0;
};

}  // namespace weights
}  // namespace npuw
}  // namespace ov
