// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/runtime/shared_buffer.hpp"

namespace intel_npu {

class BlobContainer {
public:
    virtual void* get_ptr() = 0;

    virtual size_t size() const = 0;

    virtual bool release_from_memory() = 0;

    virtual ~BlobContainer() = default;
};

class BlobContainerVector : public BlobContainer {
public:
    BlobContainerVector(std::vector<uint8_t> blob) : _ownershipBlob(std::move(blob)) {}

    void* get_ptr() override {
        return reinterpret_cast<void*>(_ownershipBlob.data());
    }

    size_t size() const override {
        return _ownershipBlob.size();
    }

    bool release_from_memory() override {
        _ownershipBlob.clear();
        _ownershipBlob.shrink_to_fit();
        return true;
    }

private:
    std::vector<uint8_t> _ownershipBlob;
};

class BlobContainerAlignedBuffer : public BlobContainer {
public:
    BlobContainerAlignedBuffer(const std::shared_ptr<ov::AlignedBuffer>& blobSO,
                               size_t ovHeaderOffset,
                               uint64_t blobSize)
        : _blobSize(blobSize),
          _ovHeaderOffset(ovHeaderOffset),
          _ownershipBlob(blobSO) {}

    void* get_ptr() override {
        return _ownershipBlob->get_ptr(_ovHeaderOffset);
    }

    size_t size() const override {
        return _blobSize;
    }

    bool release_from_memory() override {
        return false;
    }

private:
    uint64_t _blobSize;
    size_t _ovHeaderOffset;
    std::shared_ptr<ov::AlignedBuffer> _ownershipBlob;
};

}  // namespace intel_npu
