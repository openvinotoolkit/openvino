// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/runtime/shared_buffer.hpp"

namespace {

class stringbuf_helper : public std::stringbuf {
public:
    stringbuf_helper(const std::shared_ptr<ov::AlignedBuffer>& blob) : std::stringbuf(std::ios::in), _blob(blob) {
        setg(_blob->get_ptr<char>(), _blob->get_ptr<char>(), _blob->get_ptr<char>() + _blob->size());
    }
private:
    std::shared_ptr<ov::AlignedBuffer> _blob;
};

}  // anonymous-namespace

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

    virtual bool swap_stringbuf (std::ostream& stream) = 0;

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

    bool swap_stringbuf(std::ostream& stream) override {
        return false;
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

    bool swap_stringbuf(std::ostream& stream) override {
        if (auto* sstream = dynamic_cast<std::stringstream*>(&stream);
            sstream != nullptr) {
            std::stringbuf* sbh = new stringbuf_helper(_blobSO);
            sstream->rdbuf()->swap(*sbh);
            int index = std::ostream::xalloc();
            stream.pword(index) = sbh;
            stream.register_callback([](std::ios_base::event evt, std::ios_base& stream, int idx) {
                if (evt == std::ios_base::event::erase_event) {
                    stringbuf_helper* sbh = static_cast<stringbuf_helper*>(stream.pword(idx));
                    delete sbh;
                }
            }, index);
            return true;
        }
        return false;
    }

private:
    uint64_t _size;
    size_t _ovHeaderOffset;
    std::shared_ptr<ov::AlignedBuffer> _blobSO;
};

}  // namespace intel_npu
