// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_layouts.h"

#include <algorithm>
#include <map>

using namespace InferenceEngine;

TensorDesc::TensorDesc(const Precision& precision, const SizeVector& dims, Layout layout)
    : precision(precision), blockingDesc(dims, layout) {
    this->dims = dims;
    this->layout = layout;
}

TensorDesc::TensorDesc(const Precision& precision, Layout layout): precision(precision), blockingDesc() {
    this->layout = layout;
}

TensorDesc::TensorDesc(const Precision& precision, const SizeVector& dims, const BlockingDesc& blockDesc)
    : dims(dims), precision(precision), blockingDesc(blockDesc) {
    if (dims.size() == 0 || blockingDesc.getBlockDims().size() == 0) {
        layout = Layout::SCALAR;
        return;
    }
    if (dims.size() != *std::max_element(blockDesc.getOrder().begin(), blockDesc.getOrder().end()) + 1)
        IE_THROW() << "Cannot create TensorDesc! Blocked dims are inconsistent with original dims.";

    layout = Layout::BLOCKED;
    if (dims.size() == blockingDesc.getBlockDims().size()) {
        switch (dims.size()) {
        case 1:
            layout = Layout::C;
            break;
        case 2:
            if (blockingDesc.getOrder()[0] == 0 && blockingDesc.getOrder()[1] == 1)
                layout = Layout::NC;
            else
                layout = Layout::CN;
            break;
        case 3:
            if (blockingDesc.getOrder()[0] == 0 && blockingDesc.getOrder()[1] == 1 && blockingDesc.getOrder()[2] == 2) {
                layout = Layout::CHW;
            } else if (blockingDesc.getOrder()[0] == 1 && blockingDesc.getOrder()[1] == 2 && blockingDesc.getOrder()[2] == 0) {
                layout = Layout::HWC;
            }
            break;
        case 4:
            if (blockingDesc.getOrder()[0] == 0 && blockingDesc.getOrder()[1] == 1 && blockingDesc.getOrder()[2] == 2 &&
                blockingDesc.getOrder()[3] == 3) {
                layout = Layout::NCHW;
            } else if (blockingDesc.getOrder()[0] == 0 && blockingDesc.getOrder()[1] == 2 &&
                       blockingDesc.getOrder()[2] == 3 && blockingDesc.getOrder()[3] == 1) {
                layout = Layout::NHWC;
            }
            break;
        case 5:
            if (blockingDesc.getOrder()[0] == 0 && blockingDesc.getOrder()[1] == 1 && blockingDesc.getOrder()[2] == 2 &&
                blockingDesc.getOrder()[3] == 3 && blockingDesc.getOrder()[4] == 4) {
                layout = Layout::NCDHW;
            } else if (blockingDesc.getOrder()[0] == 0 && blockingDesc.getOrder()[1] == 2 &&
                       blockingDesc.getOrder()[2] == 3 && blockingDesc.getOrder()[3] == 4 &&
                       blockingDesc.getOrder()[4] == 1) {
                layout = Layout::NDHWC;
            }
            break;
        default:
            break;
        }
    }
}

TensorDesc::TensorDesc() {
    this->layout = Layout::ANY;
    precision = Precision::UNSPECIFIED;
}

void TensorDesc::setDims(const SizeVector& dims) {
    if (layout == Layout::BLOCKED) {
        auto newDims = blockingDesc.getBlockDims();
        auto newOrder = blockingDesc.getOrder();
        if (newDims.empty()) newDims = dims;
        if (newOrder.empty()) {
            for (size_t i = 0; i < newDims.size(); i++) {
                newOrder.push_back(i);
            }
        }
        blockingDesc = BlockingDesc(newDims, newOrder);
    } else {
        if (layout == Layout::SCALAR && (dims.size() > 1 || (dims.size() == 1 && dims[0] != 1)))
            IE_THROW() << "Cannot set dimensions for SCALAR layout!";
        blockingDesc = BlockingDesc(dims, layout);
    }
    if (layout != Layout::SCALAR) this->dims = dims;
}

void TensorDesc::setLayout(Layout l) {
    bool inconsistentLayout = true;

    switch (l) {
    case Layout::SCALAR:
        inconsistentLayout = !dims.empty();
        break;
    case Layout::C:
        inconsistentLayout = dims.size() != 1;
        break;
    case Layout::BLOCKED:
    case Layout::ANY:
        inconsistentLayout = false;
        break;
    case Layout::GOIDHW:
        inconsistentLayout = dims.size() != 6;
        break;
    case Layout::NCDHW:
    case Layout::NDHWC:
    case Layout::OIDHW:
    case Layout::GOIHW:
        inconsistentLayout = dims.size() != 5;
        break;
    case Layout::OIHW:
    case Layout::NCHW:
    case Layout::NHWC:
        inconsistentLayout = dims.size() != 4;
        break;
    case Layout::CHW:
    case Layout::HWC:
        inconsistentLayout = dims.size() != 3;
        break;
    case Layout::CN:
    case Layout::NC:
    case Layout::HW:
        inconsistentLayout = dims.size() != 2;
        break;
    default:
        break;
    }

    if (inconsistentLayout) {
        IE_THROW() << "Size of dims(" << std::to_string(dims.size()) << ") and format(" << l
                           << ") are inconsistent.";
    }

    // HACK: we need to update BlockingDesc after layout change, but if it was set manually not sure how to di this properly
    const bool hasDefaultBlockingDesc =
            blockingDesc == BlockingDesc(dims, layout);

    layout = l;

    if (hasDefaultBlockingDesc) {
        blockingDesc = BlockingDesc(dims, layout);
    }
}

bool TensorDesc::operator==(const TensorDesc& rhs) const {
    return blockingDesc == rhs.blockingDesc && precision == rhs.precision && layout == rhs.layout && dims == rhs.dims;
}

bool TensorDesc::operator!=(const TensorDesc& rhs) const {
    return !(*this == rhs);
}

Layout TensorDesc::getLayoutByRank(size_t rank) {
    switch (rank) {
    case 0:
        return Layout::SCALAR;
    case 1:
        return Layout::C;
    case 2:
        return Layout::NC;
    case 3:
        return Layout::CHW;
    case 4:
        return Layout::NCHW;
    case 5:
        return Layout::NCDHW;
    default:
        return Layout::BLOCKED;
    }
}

Layout TensorDesc::getLayoutByDims(const SizeVector& dims) {
    return getLayoutByRank(dims.size());
}

size_t TensorDesc::offset(const SizeVector& v) const {
    if (layout == Layout::ANY) IE_THROW() << "Cannot calculate offset for any format!";

    if (layout == Layout::SCALAR) return blockingDesc.getOffsetPadding();

    SizeVector off_v = v;
    const SizeVector& blockedDims = blockingDesc.getBlockDims();
    const SizeVector& strides = blockingDesc.getStrides();
    const SizeVector& order = blockingDesc.getOrder();

    size_t n_blocked_dims = order.size();
    if (blockedDims.size() != n_blocked_dims || strides.size() != n_blocked_dims) {
        IE_THROW() << "Cannot calculate offset. Incorrect primitive descriptor!";
    }
    SizeVector blockedShift(n_blocked_dims);
    for (size_t i = 1; i <= n_blocked_dims; i++) {
        blockedShift[n_blocked_dims - i] = off_v[order[n_blocked_dims - i]] % blockedDims[n_blocked_dims - i];
        off_v[order[n_blocked_dims - i]] /= blockedDims[n_blocked_dims - i];
    }
    size_t offset = blockingDesc.getOffsetPadding();
    for (size_t d = 0; d < n_blocked_dims; ++d) {
        const size_t p = blockedShift[d] + blockingDesc.getOffsetPaddingToData()[d];
        offset += p * strides[d];
    }
    return offset;
}

size_t TensorDesc::offset(size_t l) const {
    size_t n_dims = dims.size();
    SizeVector pos(n_dims);
    for (size_t rd = 1; rd <= n_dims; ++rd) {
        const size_t d = n_dims - rd;
        const size_t cur_dim = dims[d];
        pos[d] = l % cur_dim;
        l /= cur_dim;
    }
    return offset(pos);
}

void TensorDesc::reshape(const SizeVector& dims, Layout layout) {
    for (auto& padd : blockingDesc.getOffsetPaddingToData()) {
        if (padd) IE_THROW() << "Cannot reshape a non-packaged blob!";
    }
    if (layout != Layout::ANY) {
        blockingDesc = BlockingDesc(dims, layout);
        this->layout = layout;
    } else {
        blockingDesc = BlockingDesc(dims, this->layout);
    }
    this->dims = dims;
}

void TensorDesc::reshape(const SizeVector& dims, const BlockingDesc& blockDesc) {
    blockingDesc = blockDesc;
    this->dims = dims;
    this->layout = Layout::BLOCKED;
}

BlockingDesc::BlockingDesc(const SizeVector& block_dims, const SizeVector& order): offsetPadding(0) {
    this->order = order;
    if (block_dims.empty() || order.empty()) return;
    fillDesc(block_dims, order);
}

BlockingDesc::BlockingDesc(): BlockingDesc({}, Layout::ANY) {}

BlockingDesc::BlockingDesc(const SizeVector& blocked_dims, const SizeVector& order, size_t offset)
    : BlockingDesc(blocked_dims, order) {
    this->offsetPadding = offset;
}

BlockingDesc::BlockingDesc(const SizeVector& blocked_dims, const SizeVector& order, size_t offset,
                           const SizeVector& dimOffsets)
    : BlockingDesc(blocked_dims, order) {
    this->offsetPadding = offset;
    if (blocked_dims.size() != dimOffsets.size())
        IE_THROW() << "Offsets are not initialized for all dimensions.";
    this->offsetPaddingToData = dimOffsets;
}

BlockingDesc::BlockingDesc(const SizeVector& blocked_dims, const SizeVector& order, size_t offset,
                           const SizeVector& dimOffsets, const SizeVector& strides)
    : BlockingDesc(blocked_dims, order) {
    this->offsetPadding = offset;
    if (blocked_dims.size() != strides.size()) IE_THROW() << "Strides are not initialized for all dimensions.";
    this->strides = strides;
    if (blocked_dims.size() != dimOffsets.size())
        IE_THROW() << "Offsets are not initialized for all dimensions.";
    this->offsetPaddingToData = dimOffsets;
}

BlockingDesc::BlockingDesc(const SizeVector& dims, Layout layout): offsetPadding(0) {
    if (dims.empty()) return;

    offsetPadding = 0;
    auto checkDims = [](size_t r_size, size_t e_size) {
        if (r_size != e_size) IE_THROW() << "Dims and format are inconsistent.";
    };
    SizeVector l_order;
    SizeVector l_dims;
    switch (layout) {
    case Layout::SCALAR:
    case Layout::ANY:
        return;
    case Layout::C:
        checkDims(dims.size(), 1);
        l_order = {0};
        l_dims = dims;
        break;
    case Layout::OIHW:
    case Layout::NCHW:
        checkDims(dims.size(), 4);
        l_order = {0, 1, 2, 3};
        l_dims = dims;
        break;
    case Layout::OIDHW:
    case Layout::GOIHW:
    case Layout::NCDHW:
        checkDims(dims.size(), 5);
        l_order = {0, 1, 2, 3, 4};
        l_dims = dims;
        break;
    case Layout::GOIDHW:
        checkDims(dims.size(), 6);
        l_order = {0, 1, 2, 3, 4, 5};
        l_dims = dims;
        break;
    case Layout::NHWC:
        checkDims(dims.size(), 4);
        l_order = {0, 2, 3, 1};
        l_dims = {dims[0], dims[2], dims[3], dims[1]};
        break;
    case Layout::NDHWC:
        checkDims(dims.size(), 5);
        l_order = {0, 2, 3, 4, 1};
        l_dims = {dims[0], dims[2], dims[3], dims[4], dims[1]};
        break;
    case Layout::CHW:
        checkDims(dims.size(), 3);
        l_order = {0, 1, 2};
        l_dims = dims;
        break;
    case Layout::HWC:
        checkDims(dims.size(), 3);
        l_order = {1, 2, 0};
        l_dims = {dims[1], dims[2], dims[0]};
        break;
    case Layout::CN:
        checkDims(dims.size(), 2);
        l_order = {1, 0};
        l_dims = {dims[1], dims[0]};
        break;
    case Layout::NC:
    case Layout::HW:
        checkDims(dims.size(), 2);
        l_order = {0, 1};
        l_dims = dims;
        break;
    case Layout::BLOCKED:
        l_order.clear();
        for (size_t i = 0; i < dims.size(); i++) l_order.push_back(i);
        l_dims = dims;
        break;
    }

    fillDesc(l_dims, l_order);
}

void BlockingDesc::fillDesc(const SizeVector& blocked_dims, const SizeVector& order) {
    if (order.size() != blocked_dims.size())
        IE_THROW() << "Cannot fill descriptor. Size of dimensions and order vector don't match.";
    if (blocked_dims.empty() || order.empty())
        IE_THROW() << "Cannot fill descriptor. Dimensions and order vector are empty.";
    this->order = order;
    this->blockedDims = blocked_dims;
    offsetPadding = 0;
    offsetPaddingToData.resize(order.size());
    strides.resize(order.size());
    strides[strides.size() - 1] = 1;
    offsetPaddingToData[offsetPaddingToData.size() - 1] = 0;
    for (size_t i = 2; i <= order.size(); i++) {
        offsetPaddingToData[offsetPaddingToData.size() - i] = 0;
        strides[strides.size() - i] = strides[strides.size() - (i - 1)] * blocked_dims[blocked_dims.size() - (i - 1)];
    }

    offsetPadding = 0;
}

bool BlockingDesc::operator==(const BlockingDesc& rhs) const {
    return blockedDims == rhs.blockedDims && strides == rhs.strides && offsetPaddingToData == rhs.offsetPaddingToData &&
           order == rhs.order && offsetPadding == rhs.offsetPadding;
}

bool BlockingDesc::operator!=(const BlockingDesc& rhs) const {
    return !(*this == rhs);
}

namespace {

struct DimSlice {
    size_t startInd = 0;
    size_t size = 0;

    DimSlice() = default;

    DimSlice(size_t startInd, size_t size) :
        startInd(startInd), size(size) {
    }
};

using TensorSlice = std::vector<DimSlice>;

void checkROI(
        const TensorDesc& origDesc,
        const TensorSlice& roi) {
    const auto numDims = origDesc.getDims().size();

    if (roi.size() != numDims) {
        IE_THROW()
            << "ROI num dims " << roi.size() <<
            " differs from original num dims " << numDims;
    }

    // TensorDesc stores dimensions in standard layout, as well as roi vector
    for (size_t dimInd = 0; dimInd < numDims; ++dimInd) {
        const auto fullSize = origDesc.getDims()[dimInd];

        const auto& roiSlice = roi[dimInd];
        const auto endInd = roiSlice.startInd + roiSlice.size;

        if (endInd > fullSize) {
            IE_THROW()
                << "ROI [" << roiSlice.startInd << ", " << endInd << ")"
                << " is out of range " << fullSize
                << " for dimension " << dimInd;
        }
    }
}

TensorDesc make_roi_desc(
        const TensorDesc& origDesc,
        const TensorSlice& roi,
        bool useOrigMemDesc) {
    const auto numDims = origDesc.getDims().size();

    checkROI(origDesc, roi);

    const auto origPrecision = origDesc.getPrecision();

    const auto& origBlkDesc = origDesc.getBlockingDesc();
    const auto& origBlkStrides = origBlkDesc.getStrides();
    const auto& origBlkOrder = origBlkDesc.getOrder();

    SizeVector roiDims(numDims);
    SizeVector roiBlkDims(numDims);
    SizeVector roiBlkDimOffsets = origBlkDesc.getOffsetPaddingToData();
    size_t roiBlkOffset = origBlkDesc.getOffsetPadding();

    IE_ASSERT(origBlkStrides.size() == numDims);
    IE_ASSERT(origBlkOrder.size() == numDims);
    IE_ASSERT(roiBlkDimOffsets.size() == numDims);

    // BlockingDesc stores dimensions in memory order, so we need to use origOrder array.
    // Offsets in `roi` relates to `origDesc` dimensions, while offsets in `BlockingDesc` relates to top parent tensor dimensions.
    for (size_t memInd = 0; memInd < numDims; ++memInd) {
        const auto dimInd = origBlkOrder[memInd];
        const auto& roiSlice = roi[dimInd];

        roiDims[dimInd] = roiSlice.size;
        roiBlkDims[memInd] = roiSlice.size;
        roiBlkDimOffsets[memInd] += roiSlice.startInd;
        roiBlkOffset += roiSlice.startInd * origBlkStrides[memInd];
    }

    const auto roiBlkDesc =
        useOrigMemDesc ?
            BlockingDesc(roiBlkDims, origBlkOrder, roiBlkOffset, roiBlkDimOffsets, origBlkStrides) :
            BlockingDesc(roiBlkDims, origBlkOrder);

    const auto roiDesc = TensorDesc(origPrecision, roiDims, roiBlkDesc);

    return roiDesc;
}

TensorSlice make_roi_slice(
        const TensorDesc& origDesc,
        const ROI& roi) {
    const auto layout = origDesc.getLayout();
    if (layout != Layout::NCHW && layout != Layout::NHWC) {
        IE_THROW()
            << "Unsupported layout " << layout;
    }

    TensorSlice roiSlice(4);
    roiSlice[0] = DimSlice {roi.id, 1};                 // N
    roiSlice[1] = DimSlice {0, origDesc.getDims()[1]};  // C
    roiSlice[2] = DimSlice {roi.posY, roi.sizeY};       // H
    roiSlice[3] = DimSlice {roi.posX, roi.sizeX};       // W

    return roiSlice;
}

}  // namespace

TensorDesc InferenceEngine::make_roi_desc(
        const TensorDesc& origDesc,
        const ROI& roi,
        bool useOrigMemDesc) {
    return make_roi_desc(origDesc, make_roi_slice(origDesc, roi), useOrigMemDesc);
}
