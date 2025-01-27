// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/runtime/shared_buffer.hpp"

namespace intel_npu {

class BlobContainer {
public:
    /**
     * @brief Returns the address at the beginning of the blob.
     */
    virtual const void* get_ptr() const = 0;

    /**
     * @brief Size of the blob.
     */
    virtual size_t size() const = 0;

    /**
     * @brief Returns true if the blob can be deallocated from memory, false otherwise.
     */
    virtual bool release_from_memory() = 0;

    virtual ~BlobContainer() = default;
};

class BlobContainerVector : public BlobContainer {
public:
    BlobContainerVector(std::vector<uint8_t> blob) : _blob(std::move(blob)) {}

    const void* get_ptr() const override {
        return reinterpret_cast<const void*>(_blob.data());
    }

    size_t size() const override {
        return _blob.size();
    }

    bool release_from_memory() override {
        _blob.clear();
        _blob.shrink_to_fit();
        return true;
    }

private:
    std::vector<uint8_t> _blob;
};

class BlobContainerAlignedBuffer : public BlobContainer {
public:
    BlobContainerAlignedBuffer(const std::shared_ptr<ov::AlignedBuffer>& blobSO, size_t ovHeaderOffset, uint64_t size)
        : _size(size),
          _ovHeaderOffset(ovHeaderOffset),
          _blobSO(blobSO) {}

    const void* get_ptr() const override {
        return _blobSO->get_ptr(_ovHeaderOffset);
    }

    size_t size() const override {
        return _size;
    }

    bool release_from_memory() override {
        return false;
    }

private:
    uint64_t _size;
    size_t _ovHeaderOffset;
    std::shared_ptr<ov::AlignedBuffer> _blobSO;
};

}  // namespace intel_npu
