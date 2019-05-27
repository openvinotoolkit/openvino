// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/model/data_desc.hpp>

#include <array>
#include <algorithm>
#include <queue>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <limits>

#include <precision_utils.h>

#include <vpu/model/edges.hpp>
#include <vpu/utils/ie_helpers.hpp>
#include <vpu/utils/numeric.hpp>

namespace vpu {

//
// DimsOrder
//

namespace {

const StorageOrder64 ORDER_MASK = static_cast<StorageOrder64>(-1ull) >> (std::numeric_limits<StorageOrder64>::digits / 4 - MAX_DIMS_64);

}  // namespace

StorageOrder64 maskOrder(StorageOrder64 fullOrder, int size) {
    StorageOrder64 mask = ~ORDER_MASK | ~(static_cast<StorageOrder64>(-1ull) << (size * 4));
    return fullOrder & mask;
}

DimsOrder DimsOrder::C = DimsOrder::fromCode(0x3);
DimsOrder DimsOrder::NC = DimsOrder::fromCode(0x43);
DimsOrder DimsOrder::CHW = DimsOrder::fromCode(0x321);
DimsOrder DimsOrder::HWC = DimsOrder::fromCode(0x213);
DimsOrder DimsOrder::HCW = DimsOrder::fromCode(0x231);
DimsOrder DimsOrder::NCHW = DimsOrder::fromCode(0x4321);
DimsOrder DimsOrder::NHWC = DimsOrder::fromCode(0x4213);
DimsOrder DimsOrder::NHCW = DimsOrder::fromCode(0x4231);

namespace {

bool isOrderCodeValid(StorageOrder64 order) {
    if (order == 0) {
        return false;
    }

    std::unordered_set<int> usedDims;

    auto orderCopy = order;

    int length = 0;

    for (int i = 0; i < MAX_DIMS_64; i++) {
        int digit = orderCopy & 0xF;
        if (digit == 0) {
            break;
        }

        --digit;

        // Dimension is used more than once
        if (usedDims.count(digit) > 0) {
            return false;
        }
        usedDims.insert(digit);

        length = i + 1;

        orderCopy >>= 4;
    }

    orderCopy = order >> (4 * length);

    // All digits on positions upper or equal to the order length should be UNDEF
    for (int i = length; i < MAX_DIMS_64; i++) {
        int digit = orderCopy & 0xF;
        if (digit != 0) {
            break;
        }

        orderCopy >>= 4;
    }

    return true;
}

}  // namespace

DimsOrder DimsOrder::fromCode(StorageOrder64 code) {
    IE_ASSERT(isOrderCodeValid(code));
    DimsOrder out;
    out._code = code;
    return out;
}

DimsOrder DimsOrder::fromNumDims(int numDims) {
    static const StorageOrder64 FULL_ORDER_DEFAULT =
            maskOrder(static_cast<StorageOrder64>(0x0fedcba987654321ull), MAX_DIMS_64);

    if (numDims == 1) {
        return DimsOrder::C;
    } else if (numDims == 2) {
        return DimsOrder::NC;
    } else {
        return DimsOrder::fromCode(maskOrder(FULL_ORDER_DEFAULT, numDims));
    }
}

DimsOrder DimsOrder::fromPermutation(const std::vector<Dim>& perm) {
    StorageOrder64 code = 0;

    for (int sh = 0, i = 0; i < perm.size(); i++, sh += 4) {
        code += (((static_cast<StorageOrder64>(perm[i]) + 1ull) & 0xFull) << sh);
    }

    return DimsOrder::fromCode(code);
}

int DimsOrder::numDims() const {
    int out = 0;

    auto code = _code;

    for (int i = 0; i < MAX_DIMS_64; i++) {
        auto digit = code & 0xF;
        if (digit == 0)
            break;

        ++out;

        code >>= 4;
    }

    return out;
}

bool DimsOrder::hasDim(Dim d) const {
    auto dimDigit = static_cast<int>(d) + 1;

    auto code = _code;

    for (int i = 0; i < MAX_DIMS_64; i++) {
        auto digit = code & 0xF;
        if (digit == 0)
            break;

        if (digit == dimDigit) {
            return true;
        }

        code >>= 4;
    }

    return false;
}

int DimsOrder::dimInd(Dim d) const {
    auto dimDigit = static_cast<int>(d) + 1;

    auto code = _code;

    for (int i = 0; i < MAX_DIMS_64; i++) {
        auto digit = code & 0xF;
        if (digit == 0)
            break;

        if (digit == dimDigit) {
            return i;
        }

        code >>= 4;
    }

    VPU_THROW_EXCEPTION << "Dim " << d << " is not avaialble in layout " << toString(*this);
}

std::vector<Dim> DimsOrder::toPermutation() const {
    std::vector<Dim> out;
    out.reserve(MAX_DIMS_64);

    auto code = _code;

    for (int i = 0; i < MAX_DIMS_64; i++) {
        auto digit = code & 0xF;
        if (digit == 0)
            break;

        auto d = static_cast<Dim>(digit - 1);
        out.emplace_back(d);

        code >>= 4;
    }

    return out;
}

DimValues DimsOrder::toIndices() const {
    DimValues out;

    auto code = _code;

    for (int i = 0; i < MAX_DIMS_64; i++) {
        auto digit = code & 0xF;
        if (digit == 0)
            break;

        auto d = static_cast<Dim>(digit - 1);
        out.set(d, i);

        code >>= 4;
    }

    return out;
}

void DimsOrder::moveDim(Dim dim, int newPos) {
    IE_ASSERT(newPos >= 0 && newPos < numDims());

    int oldPos = dimInd(dim);
    if (oldPos == newPos)
        return;

    auto step = (oldPos > newPos) ? -1 : 1;

    auto perm = toPermutation();
    IE_ASSERT(newPos < perm.size());

    for (int i = oldPos; i != newPos; i += step) {
        perm[i] = perm[i + step];
    }

    perm[newPos] = dim;

    _code = fromPermutation(perm).code();
}

DimsOrder DimsOrder::createMovedDim(Dim dim, int newPos) const {
    auto copy = *this;
    copy.moveDim(dim, newPos);
    return copy;
}

bool isOrdersCompatible(DimsOrder order1, DimsOrder order2) {
    auto vec1 = order1.toPermutation();
    auto vec2 = order2.toPermutation();

    std::sort(vec1.begin(), vec1.end());
    std::sort(vec2.begin(), vec2.end());

    return vec1 == vec2;
}

void printTo(std::ostream& os, DimsOrder order) {
    static std::unordered_map<int, char> DIM_NAMES({
        {1, 'W'},
        {2, 'H'},
        {3, 'C'},
        {4, 'N'}
    });

    auto code = order.code();

    int i = MAX_DIMS_64 - 1;

    for (; i >= 0; i--) {
        auto curDim = (code >> (i * 4)) & 0xF;

        if (curDim != 0)
            break;
    }

    for (; i >= 0; i--) {
        auto curDim = (code >> (i * 4)) & 0xF;

        auto it = DIM_NAMES.find(curDim);
        if (it != DIM_NAMES.end()) {
            os << it->second;
        } else {
            os << curDim;
        }
    }
}

//
// DataDesc
//

DataDesc::DataDesc(const ie::TensorDesc& ieDesc) {
    //
    // Parse precision
    //

    switch (ieDesc.getPrecision()) {
    case ie::Precision::U8:
        _type = DataType::U8;
        break;
    case ie::Precision::FP16:
        _type = DataType::FP16;
        break;
    case ie::Precision::FP32:
        _type = DataType::FP32;
        break;
    default:
        VPU_THROW_EXCEPTION << "Unsupported precision " << ieDesc.getPrecision().name();
    }

    //
    // Parse dimensions and layout
    //

    const auto& ieDims = ieDesc.getDims();
    IE_ASSERT(!ieDims.empty());

    _dimsOrder = DimsOrder::fromNumDims(ieDims.size());

    auto perm = _dimsOrder.toPermutation();

    for (int i = 0; i < perm.size(); ++i) {
        _dims.set(perm[i], ieDims[ieDims.size() - 1 - i]);
    }
}

DataDesc::DataDesc(DataType type, DimsOrder dimsOrder, const DimValues& dims) :
        _type(type), _dimsOrder(dimsOrder), _dims(dims) {
    IE_ASSERT(_dimsOrder.numDims() == _dims.size());
    for (const auto& p : _dims) {
        IE_ASSERT(_dimsOrder.hasDim(p.first));
    }
}

int DataDesc::elemSize() const {
    switch (_type) {
    case DataType::U8:
        return sizeof(uint8_t);
    case DataType::FP16:
        return sizeof(fp16_t);
    case DataType::FP32:
        return sizeof(float);
    default:
        VPU_THROW_EXCEPTION << "Unknown data type " << _type;
    }
}

void DataDesc::setDim(Dim d, int val) {
    IE_ASSERT(_dimsOrder.hasDim(d));
    _dims.set(d, val);
}

int DataDesc::totalDimSize() const {
    int total = 1;

    auto perm = _dimsOrder.toPermutation();
    for (auto d : perm) {
        total *= _dims[d];
    }

    return total;
}

void DataDesc::reorder(DimsOrder dimsOrder) {
    IE_ASSERT(isOrdersCompatible(_dimsOrder, dimsOrder));
    _dimsOrder = dimsOrder;
}

void printTo(std::ostream& os, const DataDesc& desc) {
    os << "[" << std::endl;

    os << "type=";
    printTo(os, desc.type());
    os << std::endl;

    os << "dimsOrder=";
    printTo(os, desc.dimsOrder());
    os << std::endl;

    os << "dims=";
    printTo(os, desc.dims());
    os << std::endl;

    os << "]";
}

void printTo(DotLabel& lbl, const DataDesc& desc) {
    DotLabel subLbl(lbl);
    subLbl.appendPair("type", desc.type());
    subLbl.appendPair("dimsOrder", desc.dimsOrder());
    subLbl.appendPair("dims", desc.dims());
}

//
// StridesRequirement
//

StridesRequirement StridesRequirement::compact() {
    StridesRequirement reqs;
    for (int i = 0; i < MAX_DIMS_64; ++i) {
        reqs.add(i, DimStride::Compact);
    }
    return reqs;
}

void printTo(std::ostream& os, const StridesRequirement& reqs) {
    os << "[" << std::endl;

    for (int i = 0; i < MAX_DIMS_64; ++i) {
        auto req = reqs.get(i);
        if (req != DimStride::Any) {
            printTo(os, i);
            os << "=";
            printTo(os, req);
            os << std::endl;
        }
    }

    os << "]";
}

void printTo(DotLabel& lbl, const StridesRequirement& reqs) {
    DotLabel subLbl(lbl);
    for (int i = 0; i < MAX_DIMS_64; ++i) {
        auto req = reqs.get(i);
        if (req != DimStride::Any) {
            subLbl.appendPair(i, req);
        }
    }
}

namespace {

int applyStrideRequirement(int origStride, int index, const StridesRequirement& reqs) {
    auto req = reqs.get(index);

    if (req == DimStride::Any || req == DimStride::Compact) {
        return origStride;
    } else if (req == DimStride::Aligned) {
        return alignVal(origStride, STRIDE_ALIGNMENT);
    } else {
        VPU_THROW_EXCEPTION << "Unknown stride requirement : " << req;
    }
}

}  // namespace

DimValues calcStrides(const DataDesc& desc, const StridesRequirement& reqs) {
    DimValues strides;

    auto perm = desc.dimsOrder().toPermutation();
    IE_ASSERT(!perm.empty());

    strides.set(perm[0], desc.elemSize());
    strides.set(perm[0], applyStrideRequirement(strides[perm[0]], 0, reqs));

    for (int i = 1; i < perm.size(); i++) {
        strides.set(perm[i], strides[perm[i - 1]] * desc.dim(perm[i - 1]));
        strides.set(perm[i], applyStrideRequirement(strides[perm[i]], i, reqs));
    }

    return strides;
}

bool checkStride(
        const DimValues& strides,
        const DataDesc& desc,
        int ind,
        DimStride req) {
    if (req == DimStride::Any) {
        return true;
    }

    auto perm = desc.dimsOrder().toPermutation();
    IE_ASSERT(!perm.empty());

    auto strideVal = strides[perm[ind]];

    if (req == DimStride::Compact) {
        if (ind == 0) {
            if (strideVal != desc.elemSize()) {
                return false;
            }
        } else {
            if (strides[perm[ind]] != strides[perm[ind - 1]] * desc.dim(perm[ind - 1])) {
                return false;
            }
        }
    } else if (req == DimStride::Aligned) {
        if (strideVal % STRIDE_ALIGNMENT != 0) {
            return false;
        }
    } else {
        VPU_THROW_EXCEPTION << "Unsupported stride requirement : " << req;
    }

    return true;
}

bool checkStrides(
        const DataDesc& desc,
        const DimValues& strides,
        const StridesRequirement& reqs) {
    auto perm = desc.dimsOrder().toPermutation();
    IE_ASSERT(!perm.empty());

    for (int i = 0; i < perm.size(); i++) {
        if (!checkStride(strides, desc, i, reqs.get(i))) {
            return false;
        }
    }

    return true;
}

int calcTotalByteSize(const DataDesc& desc, const DimValues& strides) {
    auto perm = desc.dimsOrder().toPermutation();
    return strides[perm.back()] * desc.dim(perm.back());
}

}  // namespace vpu
