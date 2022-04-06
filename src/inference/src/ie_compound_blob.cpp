// Copyright (C) 2018-2022 Intel Corporation
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

namespace InferenceEngine {

namespace {

TensorDesc verifyNV12BlobInput(const Blob::Ptr& y, const Blob::Ptr& uv) {
    // Y and UV must be valid pointers
    if (y == nullptr || uv == nullptr) {
        IE_THROW() << "Y and UV planes must be valid Blob objects";
    }

    // both Y and UV must be MemoryBlob objects
    if (!y->is<MemoryBlob>() || !uv->is<MemoryBlob>()) {
        IE_THROW() << "Y and UV planes must be MemoryBlob objects";
    }

    // NOTE: having Blob::Ptr (shared_ptr) and checking Blob::is() status above ensures that the
    //       cast is always successful
    auto yMemoryBlob = y->as<MemoryBlob>();
    auto uvMemoryBlob = uv->as<MemoryBlob>();
    // check Blob element size
    if (yMemoryBlob->element_size() != uvMemoryBlob->element_size()) {
        IE_THROW() << "Y and UV planes have different element sizes: " << yMemoryBlob->element_size()
                   << " != " << uvMemoryBlob->element_size();
    }

    // check tensor descriptor parameters
    const auto& yDesc = yMemoryBlob->getTensorDesc();
    const auto& uvDesc = uvMemoryBlob->getTensorDesc();

    // check precision
    if (yDesc.getPrecision() != Precision::U8) {
        IE_THROW() << "Y plane precision must be U8, actual: " << yDesc.getPrecision();
    }
    if (uvDesc.getPrecision() != Precision::U8) {
        IE_THROW() << "UV plane precision must be U8, actual: " << uvDesc.getPrecision();
    }

    // check layout
    if (yDesc.getLayout() != Layout::NHWC) {
        IE_THROW() << "Y plane layout must be NHWC, actual: " << yDesc.getLayout();
    }
    if (uvDesc.getLayout() != Layout::NHWC) {
        IE_THROW() << "UV plane layout must be NHWC, actual: " << uvDesc.getLayout();
    }

    // check dimensions
    const auto& yDims = yDesc.getDims();
    const auto& uvDims = uvDesc.getDims();
    if (yDims.size() != 4 || uvDims.size() != 4) {
        IE_THROW() << "Y and UV planes dimension sizes must be 4, actual: " << yDims.size() << "(Y plane) and "
                   << uvDims.size() << "(UV plane)";
    }

    // check batch size
    if (yDims[0] != uvDims[0]) {
        IE_THROW() << "Y and UV planes must have the same batch size";
    }

    // check number of channels
    if (yDims[1] != 1) {
        IE_THROW() << "Y plane must have 1 channel, actual: " << yDims[1];
    }
    if (uvDims[1] != 2) {
        IE_THROW() << "UV plane must have 2 channels, actual: " << uvDims[1];
    }

    // check height
    if (yDims[2] != 2 * uvDims[2]) {
        IE_THROW() << "The height of the Y plane must be equal to (2 * the height of the UV plane), actual: "
                   << yDims[2] << "(Y plane) and " << uvDims[2] << "(UV plane)";
    }

    // check width
    if (yDims[3] != 2 * uvDims[3]) {
        IE_THROW() << "The width of the Y plane must be equal to (2 * the width of the UV plane), actual: " << yDims[3]
                   << "(Y plane) and " << uvDims[3] << "(UV plane)";
    }

    return {Precision::U8, {}, Layout::NCHW};
}

TensorDesc verifyI420BlobInput(const Blob::Ptr& y, const Blob::Ptr& u, const Blob::Ptr& v) {
    // Y and UV must be valid pointers
    if (y == nullptr || u == nullptr || v == nullptr) {
        IE_THROW() << "Y, U and V planes must be valid Blob objects";
    }

    // both Y and UV must be MemoryBlob objects
    if (!y->is<MemoryBlob>() || !u->is<MemoryBlob>() || !v->is<MemoryBlob>()) {
        IE_THROW() << "Y, U and V planes must be MemoryBlob objects";
    }

    // NOTE: having Blob::Ptr (shared_ptr) and checking Blob::is() status above ensures that the
    //       cast is always successful
    auto yMemoryBlob = y->as<MemoryBlob>();
    auto uMemoryBlob = u->as<MemoryBlob>();
    auto vMemoryBlob = v->as<MemoryBlob>();
    // check Blob element size
    if (yMemoryBlob->element_size() != uMemoryBlob->element_size() ||
        yMemoryBlob->element_size() != vMemoryBlob->element_size()) {
        IE_THROW() << "Y and UV planes have different element sizes: " << yMemoryBlob->element_size()
                   << " != " << uMemoryBlob->element_size() << " != " << vMemoryBlob->element_size();
    }

    // check tensor descriptor parameters
    const auto& yDesc = yMemoryBlob->getTensorDesc();
    const auto& uDesc = uMemoryBlob->getTensorDesc();
    const auto& vDesc = vMemoryBlob->getTensorDesc();

    // check precision
    if (yDesc.getPrecision() != Precision::U8) {
        IE_THROW() << "Y plane precision must be U8, actual: " << yDesc.getPrecision();
    }
    if (uDesc.getPrecision() != Precision::U8) {
        IE_THROW() << "U plane precision must be U8, actual: " << uDesc.getPrecision();
    }
    if (vDesc.getPrecision() != Precision::U8) {
        IE_THROW() << "V plane precision must be U8, actual: " << vDesc.getPrecision();
    }

    // check layout
    if (yDesc.getLayout() != Layout::NHWC) {
        IE_THROW() << "Y plane layout must be NHWC, actual: " << yDesc.getLayout();
    }
    if (uDesc.getLayout() != Layout::NHWC) {
        IE_THROW() << "U plane layout must be NHWC, actual: " << uDesc.getLayout();
    }
    if (uDesc.getLayout() != Layout::NHWC) {
        IE_THROW() << "V plane layout must be NHWC, actual: " << vDesc.getLayout();
    }

    // check dimensions
    const auto& yDims = yDesc.getDims();
    const auto& uDims = uDesc.getDims();
    const auto& vDims = vDesc.getDims();

    if (yDims.size() != 4 || uDims.size() != 4 || vDims.size() != 4) {
        IE_THROW() << "Y,U and V planes dimension sizes must be 4, actual: " << yDims.size() << "(Y plane) and "
                   << uDims.size() << "(U plane) " << vDims.size() << "(V plane)";
    }

    // check batch size
    if (yDims[0] != uDims[0] || yDims[0] != vDims[0]) {
        IE_THROW() << "Y, U and U planes must have the same batch size";
    }

    // check number of channels
    if (yDims[1] != 1) {
        IE_THROW() << "Y plane must have 1 channel, actual: " << yDims[1];
    }
    if (uDims[1] != 1) {
        IE_THROW() << "U plane must have 1 channel, actual: " << uDims[1];
    }
    if (vDims[1] != 1) {
        IE_THROW() << "V plane must have 1 channel, actual: " << vDims[1];
    }

    // check height
    if (yDims[2] != 2 * uDims[2]) {
        IE_THROW() << "The height of the Y plane must be equal to (2 * the height of the U plane), actual: " << yDims[2]
                   << "(Y plane) and " << uDims[2] << "(U plane)";
    }

    if (yDims[2] != 2 * vDims[2]) {
        IE_THROW() << "The height of the Y plane must be equal to (2 * the height of the UV plane), actual: "
                   << yDims[2] << "(Y plane) and " << vDims[2] << "(V plane)";
    }

    // check width
    if (yDims[3] != 2 * uDims[3]) {
        IE_THROW() << "The width of the Y plane must be equal to (2 * the width of the UV plane), actual: " << yDims[3]
                   << "(Y plane) and " << uDims[3] << "(U plane)";
    }
    if (yDims[3] != 2 * vDims[3]) {
        IE_THROW() << "The width of the Y plane must be equal to (2 * the width of the UV plane), actual: " << yDims[3]
                   << "(Y plane) and " << vDims[3] << "(V plane)";
    }

    return {Precision::U8, {}, Layout::NCHW};
}

TensorDesc getBlobTensorDesc(const Blob::Ptr& blob) {
    if (auto nv12 = dynamic_cast<NV12Blob*>(blob.get())) {
        auto yDesc = nv12->y()->getTensorDesc();
        yDesc.getDims()[1] += 2;
        return yDesc;
    }

    if (auto i420 = dynamic_cast<I420Blob*>(blob.get())) {
        auto yDesc = i420->y()->getTensorDesc();
        yDesc.getDims()[1] += 2;
        return yDesc;
    }

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

NV12Blob::NV12Blob(const Blob::Ptr& y, const Blob::Ptr& uv) : CompoundBlob(verifyNV12BlobInput(y, uv)) {
    this->_blobs = {y, uv};
}

NV12Blob::NV12Blob(Blob::Ptr&& y, Blob::Ptr&& uv) : CompoundBlob(verifyNV12BlobInput(y, uv)) {
    this->_blobs = {std::move(y), std::move(uv)};
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

Blob::Ptr NV12Blob::createROI(const ROI& roi) const {
    auto yROI = roi;
    yROI.sizeX += yROI.sizeX % 2;
    yROI.sizeY += yROI.sizeY % 2;

    const auto uvROI = ROI(yROI.id, yROI.posX / 2, yROI.posY / 2, yROI.sizeX / 2, yROI.sizeY / 2);

    const auto yRoiBlob = y()->createROI(yROI);
    const auto uvRoiBlob = uv()->createROI(uvROI);

    return std::make_shared<NV12Blob>(yRoiBlob, uvRoiBlob);
}

I420Blob::I420Blob(const Blob::Ptr& y, const Blob::Ptr& u, const Blob::Ptr& v)
    : CompoundBlob(verifyI420BlobInput(y, u, v)) {
    this->_blobs = {y, u, v};
}

I420Blob::I420Blob(Blob::Ptr&& y, Blob::Ptr&& u, Blob::Ptr&& v) : CompoundBlob(verifyI420BlobInput(y, u, v)) {
    this->_blobs = {std::move(y), std::move(u), std::move(v)};
}

Blob::Ptr& I420Blob::y() noexcept {
    // NOTE: Y plane is a memory blob, which is checked in the constructor
    return _blobs[0];
}

const Blob::Ptr& I420Blob::y() const noexcept {
    // NOTE: Y plane is a memory blob, which is checked in the constructor
    return _blobs[0];
}

Blob::Ptr& I420Blob::u() noexcept {
    // NOTE: U plane is a memory blob, which is checked in the constructor
    return _blobs[1];
}

const Blob::Ptr& I420Blob::u() const noexcept {
    // NOTE: U plane is a memory blob, which is checked in the constructor
    return _blobs[1];
}

Blob::Ptr& I420Blob::v() noexcept {
    // NOTE: V plane is a memory blob, which is checked in the constructor
    return _blobs[2];
}

const Blob::Ptr& I420Blob::v() const noexcept {
    // NOTE: V plane is a memory blob, which is checked in the constructor
    return _blobs[2];
}

Blob::Ptr I420Blob::createROI(const ROI& roi) const {
    auto yROI = roi;
    yROI.sizeX += yROI.sizeX % 2;
    yROI.sizeY += yROI.sizeY % 2;

    const auto uvROI = ROI(yROI.id, yROI.posX / 2, yROI.posY / 2, yROI.sizeX / 2, yROI.sizeY / 2);

    const auto yRoiBlob = y()->createROI(yROI);
    const auto uRoiBlob = u()->createROI(uvROI);
    const auto vRoiBlob = v()->createROI(uvROI);

    return std::make_shared<I420Blob>(yRoiBlob, uRoiBlob, vRoiBlob);
}

BatchedBlob::BatchedBlob(const std::vector<Blob::Ptr>& blobs) : CompoundBlob(verifyBatchedBlobInput(blobs)) {
    this->_blobs = blobs;
}

BatchedBlob::BatchedBlob(std::vector<Blob::Ptr>&& blobs) : CompoundBlob(verifyBatchedBlobInput(blobs)) {
    this->_blobs = std::move(blobs);
}

}  // namespace InferenceEngine
