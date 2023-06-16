// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief An implementation file for CompoundBlob
 * @file ie_compound_blob.cpp
 */

#include "ie_compound_blob.h"

#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

IE_SUPPRESS_DEPRECATED_START
namespace InferenceEngine {

namespace {

TensorDesc getBlobTensorDesc(const Blob::Ptr& blob) {
    return blob->getTensorDesc();
}

TensorDesc verifyBatchedBlobInput(const std::vector<Blob::Ptr>& blobs) {
    // verify invariants
    if (blobs.empty()) {
        IE_THROW() << "BatchedBlob cannot be created from empty vector of Blob, Please, make sure vector contains at "
                      "least one Blob";
    }

    // Cannot create a compound blob from nullptr Blob objects
    if (std::any_of(blobs.begin(), blobs.end(), [](const Blob::Ptr& blob) {
            return blob == nullptr;
        })) {
        IE_THROW() << "Cannot create a compound blob from nullptr Blob objects";
    }

    const auto subBlobDesc = getBlobTensorDesc(blobs[0]);

    if (std::any_of(blobs.begin(), blobs.end(), [&subBlobDesc](const Blob::Ptr& blob) {
            return getBlobTensorDesc(blob) != subBlobDesc;
        })) {
        IE_THROW() << "All blobs tensors should be equal";
    }

    auto subBlobLayout = subBlobDesc.getLayout();

    auto blobLayout = Layout::ANY;
    SizeVector blobDims = subBlobDesc.getDims();
    switch (subBlobLayout) {
    case NCHW:
    case NHWC:
    case NCDHW:
    case NDHWC:
    case NC:
    case CN:
        blobLayout = subBlobLayout;
        if (blobDims[0] != 1) {
            IE_THROW() << "All blobs should be batch 1";
        }
        blobDims[0] = blobs.size();
        break;
    case C:
        blobLayout = NC;
        blobDims.insert(blobDims.begin(), blobs.size());
        break;
    case CHW:
        blobLayout = NCHW;
        blobDims.insert(blobDims.begin(), blobs.size());
        break;
    case HWC:
        blobLayout = NHWC;
        blobDims.insert(blobDims.begin(), blobs.size());
        break;
    default:
        IE_THROW() << "Unsupported sub-blobs layout - to be one of: [NCHW, NHWC, NCDHW, NDHWC, NC, CN, C, CHW]";
    }

    return TensorDesc{subBlobDesc.getPrecision(), blobDims, blobLayout};
}

}  // anonymous namespace

CompoundBlob::CompoundBlob(const TensorDesc& tensorDesc) : Blob(tensorDesc) {}

CompoundBlob::CompoundBlob(const std::vector<Blob::Ptr>& blobs) : CompoundBlob(TensorDesc{}) {
    // Cannot create a compound blob from nullptr Blob objects
    if (std::any_of(blobs.begin(), blobs.end(), [](const Blob::Ptr& blob) {
            return blob == nullptr;
        })) {
        IE_THROW() << "Cannot create a compound blob from nullptr Blob objects";
    }

    // Check that none of the blobs provided is compound. If at least one of them is compound, throw
    // an exception because recursive behavior is not allowed
    if (std::any_of(blobs.begin(), blobs.end(), [](const Blob::Ptr& blob) {
            return blob->is<CompoundBlob>();
        })) {
        IE_THROW() << "Cannot create a compound blob from other compound blobs";
    }

    this->_blobs = blobs;
}

CompoundBlob::CompoundBlob(std::vector<Blob::Ptr>&& blobs) : CompoundBlob(TensorDesc{}) {
    // Cannot create a compound blob from nullptr Blob objects
    if (std::any_of(blobs.begin(), blobs.end(), [](const Blob::Ptr& blob) {
            return blob == nullptr;
        })) {
        IE_THROW() << "Cannot create a compound blob from nullptr Blob objects";
    }

    // Check that none of the blobs provided is compound. If at least one of them is compound, throw
    // an exception because recursive behavior is not allowed
    if (std::any_of(blobs.begin(), blobs.end(), [](const Blob::Ptr& blob) {
            return blob->is<CompoundBlob>();
        })) {
        IE_THROW() << "Cannot create a compound blob from other compound blobs";
    }

    this->_blobs = std::move(blobs);
}

size_t CompoundBlob::byteSize() const {
    return 0;
}

size_t CompoundBlob::element_size() const {
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

Blob::Ptr CompoundBlob::createROI(const ROI& roi) const {
    std::vector<Blob::Ptr> roiBlobs;
    roiBlobs.reserve(_blobs.size());

    for (const auto& blob : _blobs) {
        roiBlobs.push_back(blob->createROI(roi));
    }

    return std::make_shared<CompoundBlob>(std::move(roiBlobs));
}

const std::shared_ptr<IAllocator>& CompoundBlob::getAllocator() const noexcept {
    static std::shared_ptr<IAllocator> _allocator = nullptr;
    return _allocator;
};

BatchedBlob::BatchedBlob(const std::vector<Blob::Ptr>& blobs) : CompoundBlob(verifyBatchedBlobInput(blobs)) {
    this->_blobs = blobs;
}

BatchedBlob::BatchedBlob(std::vector<Blob::Ptr>&& blobs) : CompoundBlob(verifyBatchedBlobInput(blobs)) {
    this->_blobs = std::move(blobs);
}

}  // namespace InferenceEngine
