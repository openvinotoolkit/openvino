// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <map>

#include "ie_layouts.h"
#include <algorithm>

using namespace InferenceEngine;

IE_SUPPRESS_DEPRECATED_START

static const std::map<Layout, SizeVector> DIM_POSITIONS = {
    { NCHW, { I_W, I_H, I_C, I_N } },
    { NHWC, { I_C, I_W, I_H, I_N } }
};

LayoutOffsetCounter::LayoutOffsetCounter(Layout layout, SizeVector dims) : _layout(layout), _dims(dims), _dims_count(dims.size()), _muls(dims.size(), -1) {
    size_t mul = 1;
    for (size_t i = 0; i < _dims_count; i++) {
        size_t index = DIM_POSITIONS.at(layout)[i];
        _muls[index] = mul;
        mul *= dims[index];
    }
}

LayoutOffsetCounter::LayoutOffsetCounter(const LayoutOffsetCounter & l) {
    _layout = l._layout;
    _dims = l._dims;
    _dims_count = l._dims_count;
    _muls = l._muls;
}

LayoutOffsetCounter & LayoutOffsetCounter::operator = (const LayoutOffsetCounter & l) {
    _layout = l._layout;
    _dims = l._dims;
    _dims_count = l._dims_count;
    _muls = l._muls;

    return *this;
}

LayoutOffsetCounter::~LayoutOffsetCounter() {
}

size_t LayoutOffsetCounter::Offset(SizeVector pos) {
    size_t res = 0;
    for (size_t i = 0; i < _dims_count; i++) {
        res += pos[i] * _muls[i];
    }

    return res;
}

IE_SUPPRESS_DEPRECATED_END

TensorDesc::TensorDesc(const Precision& precision, SizeVector dims, Layout layout): blockingDesc(dims, layout),
                                                                                    precision(precision) {
    this->dims = dims;
    this->layout = layout;
}

TensorDesc::TensorDesc(const Precision& precision, Layout layout): blockingDesc(), precision(precision) {
    this->layout = layout;
}

TensorDesc::TensorDesc(const Precision &precision, SizeVector dims, const BlockingDesc &blockDesc)
        : dims(dims), blockingDesc(blockDesc), precision(precision)  {
    if (dims.size() != *std::max_element(blockDesc.getOrder().begin(), blockDesc.getOrder().end()) + 1)
        THROW_IE_EXCEPTION << "Cannot create TensorDesc! Blocked dims are inconsistent with original dims.";

    layout = Layout::BLOCKED;
    if (dims.size() == blockingDesc.getBlockDims().size()) {
        switch (dims.size()) {
            case 0:
                layout = Layout::SCALAR;
                break;
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
                if (blockingDesc.getOrder()[0] == 0 && blockingDesc.getOrder()[1] == 1 &&
                        blockingDesc.getOrder()[2] == 2) {
                    layout = Layout::CHW;
                }
                break;
            case 4:
                if (blockingDesc.getOrder()[0] == 0 && blockingDesc.getOrder()[1] == 1 &&
                        blockingDesc.getOrder()[2] == 2 && blockingDesc.getOrder()[3] == 3) {
                    layout = Layout::NCHW;
                } else if (blockingDesc.getOrder()[0] == 0 && blockingDesc.getOrder()[1] == 2 &&
                        blockingDesc.getOrder()[2] == 3 && blockingDesc.getOrder()[3] == 1) {
                    layout = Layout::NHWC;
                }
                break;
            case 5:
                if (blockingDesc.getOrder()[0] == 0 && blockingDesc.getOrder()[1] == 1 &&
                        blockingDesc.getOrder()[2] == 2 && blockingDesc.getOrder()[3] == 3 &&
                        blockingDesc.getOrder()[4] == 4) {
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

void TensorDesc::setDims(const SizeVector &dims) {
    this->dims = dims;
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
        blockingDesc = BlockingDesc(dims, layout);
    }
}

bool TensorDesc::operator==(const TensorDesc &rhs) const {
    return blockingDesc == rhs.blockingDesc &&
            precision == rhs.precision &&
            layout == rhs.layout &&
            dims == rhs.dims;
}

bool TensorDesc::operator!=(const TensorDesc &rhs) const {
    return !(*this == rhs);
}

Layout TensorDesc::getLayoutByDims(SizeVector dims) {
    switch (dims.size()) {
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

size_t TensorDesc::offset(const SizeVector& v) const {
    if (layout == Layout::ANY)
        THROW_IE_EXCEPTION << "Cannot calculate offset for any format!";

    SizeVector off_v = v;
    const SizeVector& blockedDims = blockingDesc.getBlockDims();
    const SizeVector& strides = blockingDesc.getStrides();
    const SizeVector& order = blockingDesc.getOrder();

    size_t n_blocked_dims = order.size();
    if (blockedDims.size() != n_blocked_dims || strides.size() != n_blocked_dims) {
        THROW_IE_EXCEPTION << "Cannot calculate offset. Incorrect primitive descriptor!";
    }
    SizeVector blockedShift(n_blocked_dims);
    for (size_t i = 1; i <= n_blocked_dims; i++) {
        blockedShift[n_blocked_dims - i] = off_v[order[n_blocked_dims - i]] % blockedDims[n_blocked_dims - i];
        off_v[order[n_blocked_dims - i]] /= blockedDims[n_blocked_dims - i];
    }
    size_t offset = blockingDesc.getOffsetPadding();
    for (int d = 0; d < n_blocked_dims; ++d) {
        const size_t p = blockedShift[d] + blockingDesc.getOffsetPaddingToData()[d];
        offset += p * strides[d];
    }
    return offset;
}

size_t TensorDesc::offset(size_t l) const {
    size_t n_dims = dims.size();
    SizeVector pos(n_dims);
    for (int rd = 1; rd <= n_dims; ++rd) {
        const size_t d = n_dims - rd;
        const size_t cur_dim = dims[d];
        pos[d] = l % cur_dim;
        l /= cur_dim;
    }
    return offset(pos);
}

void TensorDesc::reshape(const SizeVector &dims, Layout layout) {
    for (auto &padd : blockingDesc.getOffsetPaddingToData()) {
        if (padd)
            THROW_IE_EXCEPTION << "Cannot reshape a non-packaged blob!";
    }
    if (layout != Layout::ANY) {
        blockingDesc = BlockingDesc(dims, layout);
        this->layout = layout;
    } else {
        blockingDesc = BlockingDesc(dims, this->layout);
    }
    this->dims = dims;
}

void TensorDesc::reshape(const SizeVector &dims, const BlockingDesc &blockDesc) {
    blockingDesc = blockDesc;
    this->dims = dims;
    this->layout = Layout::BLOCKED;
}

BlockingDesc::BlockingDesc(const SizeVector& block_dims, const SizeVector & order): offsetPadding(0) {
    this->order = order;
    if (block_dims.empty() || order.empty()) return;
    fillDesc(block_dims, order);
}

BlockingDesc::BlockingDesc(): BlockingDesc({}, Layout::ANY) {}

BlockingDesc::BlockingDesc(const SizeVector &blocked_dims, const SizeVector &order,
                           size_t offset): BlockingDesc(blocked_dims, order) {
    this->offsetPadding = offset;
}

BlockingDesc::BlockingDesc(const SizeVector &blocked_dims, const SizeVector &order, size_t offset,
                           SizeVector dimOffsets): BlockingDesc(blocked_dims, order) {
    this->offsetPadding = offset;
    if (blocked_dims.size() != dimOffsets.size())
        THROW_IE_EXCEPTION << "Offsets are not initialized for all dimensions.";
    this->offsetPaddingToData = dimOffsets;
}

BlockingDesc::BlockingDesc(const SizeVector &blocked_dims, const SizeVector &order, size_t offset,
                           SizeVector dimOffsets, SizeVector strides): BlockingDesc(blocked_dims, order) {
    this->offsetPadding = offset;
    if (blocked_dims.size() != strides.size())
        THROW_IE_EXCEPTION << "Strides are not initialized for all dimensions.";
    this->strides = strides;
    if (blocked_dims.size() != dimOffsets.size())
        THROW_IE_EXCEPTION << "Offsets are not initialized for all dimensions.";
    this->offsetPaddingToData = dimOffsets;
}

BlockingDesc::BlockingDesc(const SizeVector& dims, Layout layout): offsetPadding(0) {
    if (dims.empty())
        return;

    offsetPadding = 0;
    auto checkDims = [](size_t r_size, size_t e_size) {
        if (r_size != e_size)
            THROW_IE_EXCEPTION << "Dims and format are inconsistent.";
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
        case Layout::NCDHW:
            checkDims(dims.size(), 5);
            l_order = {0, 1, 2, 3, 4};
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
            l_dims = dims;
            break;
        case Layout::CHW:
            checkDims(dims.size(), 3);
            l_order = {0, 1, 2};
            l_dims = dims;
            break;
        case Layout::CN:
            checkDims(dims.size(), 2);
            l_order = {1, 0};
            l_dims = {dims[1], dims[2]};
            break;
        case Layout::NC:
        case Layout::HW:
            checkDims(dims.size(), 2);
            l_order = {0, 1};
            l_dims = dims;
            break;
        case Layout::BLOCKED:
            l_order.clear();
            for (size_t i = 0; i < dims.size(); i++)
                l_order.push_back(i);
            l_dims = dims;
            break;
    }

    fillDesc(l_dims, l_order);
}

void BlockingDesc::fillDesc(const SizeVector& blocked_dims, const SizeVector &order) {
    if (order.size() != blocked_dims.size())
        THROW_IE_EXCEPTION << "Cannot fill descriptor. Size of dimensions and order vector don't match.";
    if (blocked_dims.empty() || order.empty())
        THROW_IE_EXCEPTION << "Cannot fill descriptor. Dimensions and order vector are empty.";
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

bool BlockingDesc::operator==(const BlockingDesc &rhs) const {
    return blockedDims == rhs.blockedDims &&
           strides == rhs.strides &&
           offsetPaddingToData == rhs.offsetPaddingToData &&
           order == rhs.order &&
           offsetPadding == rhs.offsetPadding;
}

bool BlockingDesc::operator!=(const BlockingDesc &rhs) const {
    return !(*this == rhs);
}
