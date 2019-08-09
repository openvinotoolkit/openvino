// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief An implementation file for CompoundBlob
 * @file ie_compound_blob.cpp
 */

#include "ie_compound_blob.h"

#include <memory>
#include <utility>
#include <vector>
#include <initializer_list>

namespace InferenceEngine {
namespace {
void verifyNV12BlobInput(const Blob::Ptr& y, const Blob::Ptr& uv) {
    // Y and UV must be valid pointers
    if (y == nullptr || uv == nullptr) {
        THROW_IE_EXCEPTION << "Y and UV planes must be valid Blob objects";
    }

    // both Y and UV must be MemoryBlob objects
    if (!y->is<MemoryBlob>() || !uv->is<MemoryBlob>()) {
        THROW_IE_EXCEPTION << "Y and UV planes must be MemoryBlob objects";
    }

    // NOTE: having Blob::Ptr (shared_ptr) and checking Blob::is() status above ensures that the
    //       cast is always successful
    auto yMemoryBlob = y->as<MemoryBlob>();
    auto uvMemoryBlob = uv->as<MemoryBlob>();
    // check Blob element size
    if (yMemoryBlob->element_size() != uvMemoryBlob->element_size()) {
        THROW_IE_EXCEPTION << "Y and UV planes have different element sizes: "
                        << yMemoryBlob->element_size() << " != " << uvMemoryBlob->element_size();
    }

    // check tensor descriptor parameters
    const auto& yDesc = yMemoryBlob->getTensorDesc();
    const auto& uvDesc = uvMemoryBlob->getTensorDesc();

    // check precision
    if (yDesc.getPrecision() != Precision::U8) {
        THROW_IE_EXCEPTION << "Y plane precision must be U8, actual: " << yDesc.getPrecision();
    }
    if (uvDesc.getPrecision() != Precision::U8) {
        THROW_IE_EXCEPTION << "UV plane precision must be U8, actual: " << uvDesc.getPrecision();
    }

    // check layout
    if (yDesc.getLayout() != Layout::NHWC) {
        THROW_IE_EXCEPTION << "Y plane layout must be NHWC, actual: " << yDesc.getLayout();
    }
    if (uvDesc.getLayout() != Layout::NHWC) {
        THROW_IE_EXCEPTION << "UV plane layout must be NHWC, actual: " << uvDesc.getLayout();
    }

    // check dimensions
    const auto& yDims = yDesc.getDims();
    const auto& uvDims = uvDesc.getDims();
    if (yDims.size() != 4 || uvDims.size() != 4) {
        THROW_IE_EXCEPTION << "Y and UV planes dimension sizes must be 4, actual: "
                        << yDims.size() << "(Y plane) and " << uvDims.size() << "(UV plane)";
    }

    // check batch size
    if (yDims[0] != uvDims[0]) {
        THROW_IE_EXCEPTION << "Y and UV planes must have the same batch size";
    }

    // check number of channels
    if (yDims[1] != 1) {
        THROW_IE_EXCEPTION << "Y plane must have 1 channel, actual: " << yDims[1];
    }
    if (uvDims[1] != 2) {
        THROW_IE_EXCEPTION << "UV plane must have 2 channels, actual: " << uvDims[1];
    }

    // check height
    if (yDims[2] != 2 * uvDims[2]) {
        THROW_IE_EXCEPTION
            << "The height of the Y plane must be equal to (2 * the height of the UV plane), actual: "
            << yDims[2] << "(Y plane) and " << uvDims[2] << "(UV plane)";
    }

    // check width
    if (yDims[3] != 2 * uvDims[3]) {
        THROW_IE_EXCEPTION
            << "The width of the Y plane must be equal to (2 * the width of the UV plane), actual: "
            << yDims[3] << "(Y plane) and " << uvDims[3] << "(UV plane)";
    }
}
}  // anonymous namespace

CompoundBlob::CompoundBlob() : Blob(TensorDesc(Precision::UNSPECIFIED, {}, Layout::ANY)) {}

CompoundBlob::CompoundBlob(const CompoundBlob& blob) : CompoundBlob() {
    this->_blobs = blob._blobs;
}

CompoundBlob::CompoundBlob(CompoundBlob&& blob) : CompoundBlob() {
     this->_blobs = std::move(blob._blobs);
}

CompoundBlob::CompoundBlob(const std::vector<Blob::Ptr>& blobs) : CompoundBlob() {
    // Cannot create a compound blob from nullptr Blob objects
    if (std::any_of(blobs.begin(), blobs.end(),
                    [] (const Blob::Ptr& blob) { return blob == nullptr; })) {
        THROW_IE_EXCEPTION << "Cannot create a compound blob from nullptr Blob objects";
    }

    // Check that none of the blobs provided is compound. If at least one of them is compound, throw
    // an exception because recursive behavior is not allowed
    if (std::any_of(blobs.begin(), blobs.end(),
                    [] (const Blob::Ptr& blob) { return blob->is<CompoundBlob>(); })) {
        THROW_IE_EXCEPTION << "Cannot create a compound blob from other compound blobs";
    }

    this->_blobs = blobs;
}

CompoundBlob::CompoundBlob(std::vector<Blob::Ptr>&& blobs) : CompoundBlob() {
    // Cannot create a compound blob from nullptr Blob objects
    if (std::any_of(blobs.begin(), blobs.end(),
                    [] (const Blob::Ptr& blob) { return blob == nullptr; })) {
        THROW_IE_EXCEPTION << "Cannot create a compound blob from nullptr Blob objects";
    }

    // Check that none of the blobs provided is compound. If at least one of them is compound, throw
    // an exception because recursive behavior is not allowed
    if (std::any_of(blobs.begin(), blobs.end(),
                    [] (const Blob::Ptr& blob) { return blob->is<CompoundBlob>(); })) {
        THROW_IE_EXCEPTION << "Cannot create a compound blob from other compound blobs";
    }

    this->_blobs = std::move(blobs);
}

size_t CompoundBlob::byteSize() const noexcept {
    return 0;
}

size_t CompoundBlob::element_size() const noexcept {
    return 0;
}

void CompoundBlob::allocate() noexcept {}

bool CompoundBlob::deallocate() noexcept {
    return false;
}

LockedMemory<void> CompoundBlob::buffer() noexcept {
    return LockedMemory<void>(nullptr, nullptr, 0);
}

LockedMemory<const void> CompoundBlob::cbuffer() const noexcept {
    return LockedMemory<const void>(nullptr, nullptr, 0);
}

size_t CompoundBlob::size() const noexcept {
    return _blobs.size();
}

Blob::Ptr CompoundBlob::getBlob(size_t i) const noexcept {
    if (i >= _blobs.size()) {
        return nullptr;
    }
    return _blobs[i];
}

const std::shared_ptr<IAllocator>& CompoundBlob::getAllocator() const noexcept {
    static std::shared_ptr<IAllocator> _allocator = nullptr;
    return _allocator;
};

void* CompoundBlob::getHandle() const noexcept {
    return nullptr;
}

NV12Blob::NV12Blob(const Blob::Ptr& y, const Blob::Ptr& uv) {
    // verify data is correct
    verifyNV12BlobInput(y, uv);
    // set blobs
    _blobs.emplace_back(y);
    _blobs.emplace_back(uv);
    tensorDesc = TensorDesc(Precision::U8, {}, Layout::NCHW);
}

NV12Blob::NV12Blob(Blob::Ptr&& y, Blob::Ptr&& uv) {
    // verify data is correct
    verifyNV12BlobInput(y, uv);
    // set blobs
    _blobs.emplace_back(std::move(y));
    _blobs.emplace_back(std::move(uv));
    tensorDesc = TensorDesc(Precision::U8, {}, Layout::NCHW);
}

Blob::Ptr& NV12Blob::y() noexcept {
    // NOTE: Y plane is a memory blob, which is checked in the constructor
    return _blobs[0];
}

const Blob::Ptr& NV12Blob::y() const noexcept {
    // NOTE: Y plane is a memory blob, which is checked in the constructor
    return _blobs[0];
}

Blob::Ptr& NV12Blob::uv() noexcept {
    // NOTE: UV plane is a memory blob, which is checked in the constructor
    return _blobs[1];
}

const Blob::Ptr& NV12Blob::uv() const noexcept {
    // NOTE: UV plane is a memory blob, which is checked in the constructor
    return _blobs[1];
}

}  // namespace InferenceEngine
