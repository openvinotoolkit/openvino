// Copyright (C) 2018-2020 Intel Corporation
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

const StorageOrder64 ORDER_MASK = std::numeric_limits<StorageOrder64>::max() >> (std::numeric_limits<StorageOrder64>::digits / 4 - MAX_DIMS_64);

}  // namespace

StorageOrder64 maskOrder(StorageOrder64 fullOrder, int size) {
    StorageOrder64 mask = ~ORDER_MASK | ~(std::numeric_limits<StorageOrder64>::max() << (size * 4));
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
DimsOrder DimsOrder::NCDHW = DimsOrder::fromCode(0x43521);
DimsOrder DimsOrder::NDHWC = DimsOrder::fromCode(0x45213);

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

    if (numDims == 0 || numDims == 1) {
        return DimsOrder::C;
    } else if (numDims == 2) {
        return DimsOrder::NC;
    } else if (numDims == 3) {
        return DimsOrder::CHW;
    } else if (numDims == 4) {
        return DimsOrder::NCHW;
    } else if (numDims == 5) {
        return DimsOrder::NCDHW;
    } else {
        return DimsOrder::fromCode(maskOrder(FULL_ORDER_DEFAULT, numDims));
    }
}

DimsOrder DimsOrder::fromPermutation(const DimVector& perm) {
    StorageOrder64 code = 0;

    for (int sh = 0, i = 0; i < perm.size(); i++, sh += 4) {
        code += (((static_cast<StorageOrder64>(perm[i]) + 1ull) & 0xFull) << sh);
    }

    return DimsOrder::fromCode(code);
}

DimsOrder DimsOrder::fromLayout(ie::Layout const& layout) {
    switch (layout) {
    case ie::Layout::C : case ie::Layout::SCALAR : return DimsOrder::C;
    case ie::Layout::NC                          : return DimsOrder::NC;
    case ie::Layout::CHW                         : return DimsOrder::CHW;
    case ie::Layout::NCHW                        : return DimsOrder::NCHW;
    case ie::Layout::NHWC                        : return DimsOrder::NHWC;
    case ie::Layout::NCDHW                       : return DimsOrder::NCDHW;
    case ie::Layout::NDHWC                       : return DimsOrder::NDHWC;
    default:
        VPU_THROW_EXCEPTION << "Unsupported layout " << layout;
    }
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

DimVector DimsOrder::toPermutation() const {
    DimVector out;

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
        {4, 'N'},
        {5, 'D'}
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
std::ostream& operator<<(std::ostream& stream, const DimsOrder& object) {
    printTo(stream, object);
    return stream;
}

//
// Dim
//

int dimToIeInd(vpu::Dim const& dim, int numDims) {
    IE_ASSERT(1 <= numDims && numDims <= 8);
    auto dimsOrder =  DimsOrder::fromNumDims(numDims);
    int dimInd = dimsOrder.dimInd(dim);
    return (numDims - 1) - dimInd;
}

//
// DataDesc
//

DataDesc::DataDesc(const ie::TensorDesc& ieDesc) {
    _type = fromIEPrecision(ieDesc.getPrecision());

    const auto& ieDims = ieDesc.getDims().empty() ? ie::SizeVector{1} : ieDesc.getDims();

    const auto layout = ieDesc.getLayout();
    _dimsOrder = ieDims.size() > 5 ?
        DimsOrder::fromNumDims(ieDesc.getDims().size()) :
        DimsOrder::fromLayout(layout);

    // IE dims are always in ChannelMajor Layout, so we need to use fromNumDims() layout to perform permutation.
    const auto perm = DimsOrder::fromNumDims(ieDims.size()).toPermutation();
    for (size_t i = 0; i < perm.size(); ++i) {
        _dims.set(perm[i], ieDims[ieDims.size() - 1 - i]);
    }
}

DataDesc::DataDesc(DataType type, DimsOrder dimsOrder, const DimValues& dims) :
        _type(type), _dimsOrder(dimsOrder), _dims(dims.empty() ? DimValues{{Dim::C, 1}} : dims) {
    IE_ASSERT(_dimsOrder.numDims() == _dims.size());
    for (const auto& p : _dims) {
        IE_ASSERT(_dimsOrder.hasDim(p.first));
    }
}

int DataDesc::elemSize() const {
    switch (_type) {
    case DataType::U8:
        return sizeof(uint8_t);
    case DataType::I8:
        return sizeof(int8_t);
    case DataType::FP16:
        return sizeof(fp16_t);
    case DataType::FP32:
        return sizeof(float);
    case DataType::S32:
        return sizeof(int32_t);
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

ie::TensorDesc DataDesc::toTensorDesc() const {
    ie::TensorDesc desc;

    switch (this->type()) {
        case DataType::FP16:
            desc.setPrecision(ie::Precision::FP16);
            break;
        case DataType::FP32:
            desc.setPrecision(ie::Precision::FP32);
            break;
        case DataType::I8:
            desc.setPrecision(ie::Precision::I8);
            break;
        case DataType::U8:
            desc.setPrecision(ie::Precision::U8);
            break;
        case DataType::S32:
            desc.setPrecision(ie::Precision::I32);
            break;
        default:
            desc.setPrecision(ie::Precision::UNSPECIFIED);
    }

    ie::SizeVector dims{};

    DataDesc descCopy = *this;
    descCopy.reorder(DimsOrder::fromNumDims(this->numDims()));
    auto perm = descCopy.dimsOrder().toPermutation();
    std::reverse(perm.begin(), perm.end());
    for (auto &p : perm) {
        dims.push_back(descCopy.dim(p));
    }

    desc.setDims(dims);

    if (DimsOrder::C == this->dimsOrder()) {
        desc.setLayout(ie::Layout::C);
    } else if (DimsOrder::NC == this->dimsOrder()) {
        desc.setLayout(ie::Layout::NC);
    } else if (DimsOrder::CHW == this->dimsOrder()) {
        desc.setLayout(ie::Layout::CHW);
    } else if (DimsOrder::NCHW == this->dimsOrder()) {
        desc.setLayout(ie::Layout::NCHW);
    } else if (DimsOrder::NHWC == this->dimsOrder()) {
        desc.setLayout(ie::Layout::NHWC);
    } else if (DimsOrder::NCDHW == this->dimsOrder()) {
        desc.setLayout(ie::Layout::NCDHW);
    } else if (DimsOrder::NDHWC == this->dimsOrder()) {
        desc.setLayout(ie::Layout::NDHWC);
    } else {
        desc.setLayout(ie::Layout::BLOCKED);
    }

    return desc;
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

std::ostream& operator<<(std::ostream& stream, const DataDesc& object) {
    stream << "[" << object.type() << " " << object.dimsOrder() << " {" << object.dims() << "}]";
    return stream;
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

StridesRequirement StridesRequirement::fixed(const std::vector<int>& strides, const DataDesc& desc) {
    StridesRequirement reqs;

    const auto dims = desc.dims();
    const auto dimsOrder = desc.dimsOrder();
    const auto dimOrderVec = dimsOrder.toPermutation();
    auto setStride = [&] (Dim d, int val) {
        IE_ASSERT(dimsOrder.hasDim(d));

        auto perm = dimsOrder.toPermutation();
        auto idx = dimsOrder.dimInd(d);

        auto minStrideVal = idx == 0 ? desc.elemSize() : reqs._fixedStrides[perm[idx - 1]] * dims[perm[idx - 1]];
        IE_ASSERT(val >= minStrideVal);

        reqs._fixedStrides.set(d, val);
    };

    for (const auto& dim : dimOrderVec) {
        const auto idx = dimToIeInd(dim, dims.size());
        setStride(dim, strides[idx]);
    }

    for (int i = 0; i < MAX_DIMS_64; ++i) {
        reqs.add(i, DimStride::Fixed);
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
        return alignVal(origStride, HW_STRIDE_ALIGNMENT);
    } else {
        VPU_THROW_EXCEPTION << "Unknown stride requirement : " << req;
    }
}

}  // namespace

DimValues calcStrides(const DataDesc& desc, const StridesRequirement& reqs) {
    DimValues strides;

    auto perm = desc.dimsOrder().toPermutation();
    IE_ASSERT(!perm.empty());

    strides = reqs.fixedStrides();

    if (strides.empty()) {
        strides.set(perm[0], desc.elemSize());
        strides.set(perm[0], applyStrideRequirement(strides[perm[0]], 0, reqs));

        for (std::size_t i = 1; i < perm.size(); i++) {
            strides.set(perm[i], strides[perm[i - 1]] * desc.dim(perm[i - 1]));
            strides.set(perm[i], applyStrideRequirement(strides[perm[i]], i, reqs));
        }
    }

    return strides;
}

bool checkStride(
        const DimValues& strides,
        const DataDesc& desc,
        int ind,
        const StridesRequirement& reqs) {
    const auto req = reqs.get(ind);
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
        if (strideVal % HW_STRIDE_ALIGNMENT != 0) {
            return false;
        }
    } else if (req == DimStride::Fixed) {
        if (strideVal != reqs.getFixedStride(perm[ind])) {
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
        if (!checkStride(strides, desc, i, reqs)) {
            return false;
        }
    }

    return true;
}

int calcTotalByteSize(const DataDesc& desc, const DimValues& strides) {
    auto perm = desc.dimsOrder().toPermutation();
    return strides[perm.back()] * desc.dim(perm.back());
}

DataType fromIEPrecision(const InferenceEngine::Precision& precision) {
    switch (precision) {
        case InferenceEngine::Precision::U8:   return DataType::U8;
        case InferenceEngine::Precision::I8:   return DataType::I8;
        case InferenceEngine::Precision::I32:  return DataType::S32;
        case InferenceEngine::Precision::FP16: return DataType::FP16;
        case InferenceEngine::Precision::FP32: return DataType::FP32;
        default: VPU_THROW_EXCEPTION << precision << " isn't supported";
    }
}

PermutationIndexVector permuteMapToVector(const PermutationDimsMap& permutation, DimsOrder inputOrder, DimsOrder outputOrder) {
    PermutationIndexVector result;
    for (const auto dstDim : outputOrder.toPermutation()) {
        const auto srcDim = permutation[dstDim];
        const auto srcDimInd = inputOrder.dimInd(srcDim);
        result.push_back(srcDimInd);
    }
    return result;
}

PermutationDimsMap permuteVectorToMap(const PermutationIndexVector& permutation, DimsOrder inputOrder, DimsOrder outputOrder) {
    PermutationDimsMap result;
    const auto inputPermuteDims  = inputOrder.toPermutation();
    const auto outputPermuteDims = outputOrder.toPermutation();
    for (size_t dstIndex = 0; dstIndex < permutation.size(); ++dstIndex) {
        const int srcIndex = permutation[dstIndex];
        const auto dstDim = outputPermuteDims[dstIndex];
        const auto srcDim = inputPermuteDims[srcIndex];
        result.set(dstDim, srcDim);
    }
    return result;
}

PermutationIndexVector combinePermutationVectors(const PermutationIndexVector& first, const PermutationIndexVector& second) {
    PermutationIndexVector result;
    for (const int secondIndexSrc : second) {
        const int firstIndexSrc = first[secondIndexSrc];
        result.push_back(firstIndexSrc);
    }
    return result;
}

PermutationIndexVector calculatePermuteForReorder(DimsOrder oldLayout, DimsOrder newLayout) {
    auto newPermutation = newLayout.toPermutation();
    auto oldIndices     = oldLayout.toIndices();
    PermutationIndexVector result;
    for (const Dim newDim : newPermutation) {
        result.push_back(oldIndices[newDim]);
    }
    return result;
}

}  // namespace vpu
