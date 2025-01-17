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
    BlobContainer() = default;

    BlobContainer(std::vector<uint8_t> blob) : _blob(std::move(blob)) {}

    virtual const void* get_ptr() const {
        return _blob.data();
    }

    virtual size_t size() const {
        return _blob.size();
    }

    virtual bool release_from_memory() const {
        if (_shouldDeallocate) {
            _blob.clear();
            _blob.shrink_to_fit();
            return true;
        }
        _shouldDeallocate = true;
        return false;
    }

    virtual const std::vector<uint8_t>& get_blob() const {
        // when unerlying blob object was accessed,
        // prevent deallocation on next `release_from_memory` call
        _shouldDeallocate = false;
        return _blob;
    }

    virtual ~BlobContainer() = default;

protected:
    mutable std::vector<uint8_t> _blob;

private:
    mutable bool _shouldDeallocate = true;
};

class BlobContainerAlignedBuffer : public BlobContainer {
public:
    BlobContainerAlignedBuffer(const std::shared_ptr<ov::AlignedBuffer>& blobSO,
                               size_t ovHeaderOffset,
                               uint64_t blobSize)
        : _size(blobSize),
          _ovHeaderOffset(ovHeaderOffset),
          _blobSO(blobSO) {}

    const void* get_ptr() const override {
        return _blobSO->get_ptr(_ovHeaderOffset);
    }

    size_t size() const override {
        return _size;
    }

    bool release_from_memory() const override {
        BlobContainer::release_from_memory();
        return false;
    }

    const std::vector<uint8_t>& get_blob() const override {
        BlobContainer::release_from_memory();
        _blob.resize(_size);
        _blob.assign(reinterpret_cast<const uint8_t*>(this->get_ptr()),
                     reinterpret_cast<const uint8_t*>(this->get_ptr()) + _size);
        return _blob;
    }

private:
    uint64_t _size;
    size_t _ovHeaderOffset;
    std::shared_ptr<ov::AlignedBuffer> _blobSO;
};

}  // namespace intel_npu
