// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_desc/dnnl_blocked_memory_desc.h"

#include <algorithm>
#include <common/memory_desc_wrapper.hpp>
#include <cstdint>
#include <oneapi/dnnl/dnnl.hpp>

#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

DnnlBlockedMemoryDesc::DnnlBlockedMemoryDesc(ov::element::Type prc, const Shape& shape, const VectorDims& strides)
    : MemoryDesc(shape, DnnlBlocked) {
    const auto ndims = shape.getRank();
    const auto& dims = shape.getDims();

    if (!strides.empty()) {  // custom strides
        if (shape.hasZeroDims() && std::any_of(strides.begin(), strides.end(), [](size_t stride) {
                return stride != 0;
            })) {
            OPENVINO_THROW("Can't create DnnlBlockedMemoryDesc with zero dim, but with non zero strides");
        }
        desc = {DnnlExtensionUtils::convertToDnnlDims(dims),
                DnnlExtensionUtils::ElementTypeToDataType(prc),
                DnnlExtensionUtils::convertToDnnlDims(strides)};
    } else {
        dnnl::memory::dims plain_strides;
        if (shape.hasZeroDims()) {
            plain_strides.resize(ndims, 0);
        } else if (std::any_of(dims.begin(), dims.end(), [](size_t val) {
                       return val == Shape::UNDEFINED_DIM;
                   })) {
            plain_strides.resize(ndims, DNNL_RUNTIME_DIM_VAL);
        } else {
            plain_strides.resize(ndims, 1);
            for (size_t i = 1; i < ndims; i++) {
                plain_strides[ndims - i - 1] = plain_strides[ndims - i] * dims[ndims - i];
            }
        }

        desc = {DnnlExtensionUtils::convertToDnnlDims(dims),
                DnnlExtensionUtils::ElementTypeToDataType(prc),
                plain_strides};
    }

    order.resize(ndims);
    std::iota(order.begin(), order.end(), 0);

    initBlockedParams();
}

/**
 * Construct from blocked parameters
 *
 * OV  IOhw_4i16o4i   dims(N) = {32, 64, 128, 128}
 *   blockedDims  {4, 2, 128, 128, 4, 16, 4}                      // total dims(inner, outermost, auto blocked/padded).
 * Generally sorted by strides. strides      {8388608, 4194304,  32768, 256, 64,  4, 1}      // strides for blockedDims,
 * growing sequence order        {1, 0,   2,   3, 1,  0, 1}                      // matching to original dims
 *
 *   All vectors blockedDims/strides/order have same size equals total num of internal blocked dims(inner_dims +
 * outer_dims)
 *
 *   Tensor descriptor filing is not deterministic. It allows any permutation of index which keeps order of
 *   real dims spliting.
 *      for {1, 0, 2, 3, 1, 0, 1} we can swap elements [1] <=> [4]
 *      but not [0]<=>[4] because it break splitting original dims into internal blocked dims
 *   Normalization of representation: Make strides growing but keep layout same as original. Not all
 *   layout allow us to meet normalize form of tensor desc.
 *
 *   Limitation of conversion first N elements of order should be permutation of [0,1,2 ... N]
 */
DnnlBlockedMemoryDesc::DnnlBlockedMemoryDesc(ov::element::Type prc,
                                             const Shape& shape,
                                             const VectorDims& blockedDims,
                                             const VectorDims& order,
                                             size_t offsetPadding,
                                             const VectorDims& offsetPaddingToData,
                                             const VectorDims& strides)
    : MemoryDesc(shape, DnnlBlocked) {
    using namespace dnnl;
    // scalar case
    if (shape.getRank() == 0) {
        desc.get()->format_kind = dnnl_blocked;
        desc.get()->data_type = memory::convert_to_c(DnnlExtensionUtils::ElementTypeToDataType(prc));
        desc.get()->ndims = 1;
        desc.get()->dims[0] = 1;
        desc.get()->padded_dims[0] = 1;
        desc.get()->format_desc.blocking.strides[0] = 1;
        desc.get()->padded_offsets[0] = 0;
        desc.get()->offset0 = DnnlExtensionUtils::convertToDnnlDim(offsetPadding);

        return;
    }

    if (order.size() != blockedDims.size()) {
        OPENVINO_THROW("Can not construct DnnlBlockedMemoryDesc, order and blocked dims must have equals size");
    }

    if (!offsetPaddingToData.empty() && offsetPaddingToData.size() != order.size()) {
        OPENVINO_THROW("Can not construct DnnlBlockedMemoryDesc, offsetPaddingToData must have equal size with order "
                       "and blocked dims");
    }

    if (!strides.empty() && strides.size() != order.size()) {
        OPENVINO_THROW(
            "Can not construct DnnlBlockedMemoryDesc, strides must have equal size with order and blocked dims");
    }

    if (std::any_of(order.begin(), order.end(), [](size_t val) {
            return val == Shape::UNDEFINED_DIM;
        })) {
        OPENVINO_THROW("DnnlBlockedMemoryDesc doesn't support undefined order.");
    }

    if (std::any_of(blockedDims.begin() + shape.getRank(), blockedDims.end(), [](size_t val) {
            return val == Shape::UNDEFINED_DIM || val == 0;
        })) {
        OPENVINO_THROW("DnnlBlockedMemoryDesc doesn't support undefined or zero blockedDims.");
    }

    auto dims = DnnlExtensionUtils::convertToDnnlDims(shape.getDims());

    size_t outer_ndims = dims.size();

    auto lastIter = order.begin() + outer_ndims;
    for (size_t dim = 0; dim < outer_ndims; dim++) {
        if (std::find(order.begin(), lastIter, dim) == lastIter) {
            OPENVINO_THROW("Can not construct DnnlBlockedMemoryDesc because of incorrect order: ", vec2str(order));
        }
    }

    size_t inner_ndims = order.size() - dims.size();

    const bool emptyDesc = shape.hasZeroDims();
    if (!strides.empty()) {
        if (emptyDesc && std::any_of(strides.begin(), strides.end(), [](size_t dim) {
                return dim != 0;
            })) {
            OPENVINO_THROW("Can't create DnnlBlockedMemoryDesc with zero dim, but with non zero strides");
        }

        bool is_descending_strides = true;
        for (size_t i = 1; i < strides.size(); i++) {
            is_descending_strides &= (strides[i - 1] >= strides[i]);
        }

        // TODO: That's strong constrains and can be mitigated. IE::TensorDesc allow to transpose blocked dims
        //       and may be we can achieve correct "descending strides" form which allow conversion.
        if (!is_descending_strides) {
            OPENVINO_THROW("Can not construct DnnlBlockedMemoryDesc from strides: ", vec2str(strides));
        }
    }

    if (!strides.empty() && !emptyDesc && std::none_of(strides.begin(), strides.end(), [](size_t x) {
            return Shape::UNDEFINED_DIM == x;
        })) {
        bool inner_block_are_dense = one_of(strides.back(), 0u, 1u);  // stride 1 - is dense case, 0 - broad casted
        for (size_t i = outer_ndims; i < strides.size() - 1; i++) {
            inner_block_are_dense &= (strides[i] == strides[i + 1] * blockedDims[i + 1]);
        }

        if (!inner_block_are_dense) {
            OPENVINO_THROW("Can not construct DnnlBlockedMemoryDesc from strides: ",
                           vec2str(strides),
                           " inner blocks are not dense.");
        }
    }

    // Fill general memory desc fields
    desc.get()->format_kind = dnnl_blocked;
    desc.get()->extra.flags = 0;
    desc.get()->data_type = memory::convert_to_c(DnnlExtensionUtils::ElementTypeToDataType(prc));
    desc.get()->ndims = dims.size();
    desc.get()->offset0 = DnnlExtensionUtils::convertToDnnlDim(offsetPadding);
    std::copy(dims.begin(), dims.end(), desc.get()->dims);

    if (!offsetPaddingToData.empty()) {
        bool inner_pad_offsets_is_zero =
            std::all_of(offsetPaddingToData.begin() + outer_ndims, offsetPaddingToData.end(), [](size_t pad) {
                return pad == 0;
            });

        if (!inner_pad_offsets_is_zero) {
            OPENVINO_THROW("Can not construct DnnlBlockedMemoryDesc, inner pad offsets is not zero: ",
                           vec2str(offsetPaddingToData));
        }
        auto dnnlPaddedOffsets = DnnlExtensionUtils::convertToDnnlDims(offsetPaddingToData);
        std::copy(dnnlPaddedOffsets.begin(), dnnlPaddedOffsets.begin() + outer_ndims, desc.get()->padded_offsets);
    } else {
        std::fill(std::begin(desc.get()->padded_offsets), std::begin(desc.get()->padded_offsets) + outer_ndims, 0);
    }

    std::fill(desc.get()->padded_dims, desc.get()->padded_dims + outer_ndims, 1);
    auto dnnlBlkDims = DnnlExtensionUtils::convertToDnnlDims(blockedDims);

    for (size_t i = 0; i < order.size(); i++) {
        auto idx = order[i];
        if (desc.get()->padded_dims[idx] != DNNL_RUNTIME_DIM_VAL && dnnlBlkDims[i] != DNNL_RUNTIME_DIM_VAL) {
            desc.get()->padded_dims[idx] *= dnnlBlkDims[i];
        } else {
            desc.get()->padded_dims[idx] = DNNL_RUNTIME_DIM_VAL;
        }
    }

    // Fill blocking desc
    auto& dnn_blk_desc = desc.get()->format_desc.blocking;
    dnn_blk_desc.inner_nblks = inner_ndims;
    std::copy(dnnlBlkDims.end() - inner_ndims, dnnlBlkDims.end(), dnn_blk_desc.inner_blks);
    std::copy(order.end() - inner_ndims, order.end(), dnn_blk_desc.inner_idxs);

    this->order = order;
    this->blockedDims = blockedDims;
    initOffsetPadding();

    if (strides.empty()) {
        this->recomputeDefaultStrides();
    } else {
        for (size_t i = 0; i < outer_ndims; i++) {
            auto dnnlStrides = DnnlExtensionUtils::convertToDnnlDims(strides);
            dnn_blk_desc.strides[order[i]] = dnnlStrides[i];
        }
        this->strides = strides;
    }
}

DnnlBlockedMemoryDesc::DnnlBlockedMemoryDesc(const Shape& shape,
                                             dnnl::memory::data_type dataType,
                                             dnnl::memory::format_tag format)
    : MemoryDesc(shape, DnnlBlocked) {
    using namespace dnnl;
    if (format == memory::format_tag::any || format == memory::format_tag::undef) {
        OPENVINO_THROW("Unexpected: Can't create dnnl::desc with any or undef format");
    }

    const auto& dims = shape.getDims();
    if (format == memory::format_tag::x && shape.getRank() == 0) {
        desc = dnnl::memory::desc(dnnl::memory::dims(1, 1), dataType, format);
    } else {
        desc = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(dims), dataType, format);
    }

    VectorDims perm;
    VectorDims inner_blks;
    VectorDims inner_idxs;

    dnnl::impl::memory_desc_wrapper::compute_blocking(dnnl::memory::convert_to_c(format), perm, inner_blks, inner_idxs);

    order.swap(perm);
    order.insert(order.end(), inner_idxs.begin(), inner_idxs.end());

    if (shape.hasZeroDims()) {
        auto& blk = desc.get()->format_desc.blocking;
        std::fill(std::begin(blk.strides), std::begin(blk.strides) + desc.get()->ndims, 0);
    }

    initBlockedParams();
}

bool DnnlBlockedMemoryDesc::isCompatible(const MemoryDesc& rhs) const {
    if (auto desc = dynamic_cast<const DnnlBlockedMemoryDesc*>(&rhs)) {
        return isCompatible(*desc);
    }
    if (auto desc = dynamic_cast<const CpuBlockedMemoryDesc*>(&rhs)) {
        return isCompatible(*desc);
    }
    return false;
}

bool DnnlBlockedMemoryDesc::isCompatible(const BlockedMemoryDesc& rhs, CmpMask cmpMask) const {
    if (auto desc = dynamic_cast<const DnnlBlockedMemoryDesc*>(&rhs)) {
        return isCompatible(*desc, cmpMask);
    }
    if (auto desc = dynamic_cast<const CpuBlockedMemoryDesc*>(&rhs)) {
        return isCompatible(*desc, cmpMask);
    }
    return false;
}

bool DnnlBlockedMemoryDesc::isCompatible(const CpuBlockedMemoryDesc& rhs, CmpMask cmpMask) const {
    dnnl::impl::memory_desc_wrapper wrapped(desc.get());
    return wrapped.extra().flags == dnnl_memory_extra_flag_none &&
           BlockedMemoryDesc::isCompatibleInternal(rhs, cmpMask);
}

bool DnnlBlockedMemoryDesc::isCompatible(const DnnlBlockedMemoryDesc& rhs, CmpMask cmpMask) const {
    using namespace dnnl;
    using namespace impl;
    using namespace impl::utils;
    if (this->getShape() != rhs.getShape() || this->getPrecision() != rhs.getPrecision()) {
        return false;
    }

    memory_desc_wrapper wrappedThis(this->desc.get());
    memory_desc_wrapper wrappedRhs(rhs.desc.get());

    // TODO: do we really need this check, seems the code below does the same thing
    if (wrappedThis == wrappedRhs) {
        return true;
    }

    if (one_of(wrappedThis.format_kind(), format_kind::undef, format_kind::any)) {
        return false;
    }

    const uint64_t stride_mask = (0xffffffffffffffff << cmpMask.size()) | cmpMask.to_ullong();
    const bool checkOffset = cmpMask.test(OFFSET_MASK_POS);

    const auto thisExtra = wrappedThis.extra();
    const auto rhsExtra = wrappedRhs.extra();
    return this->getOrder() == rhs.getOrder() &&
           (thisExtra.flags == rhsExtra.flags && thisExtra.compensation_mask == rhsExtra.compensation_mask &&
            thisExtra.scale_adjust == rhsExtra.scale_adjust) &&
           wrappedThis.similar_to(wrappedRhs, true, true, 0, true, checkOffset, stride_mask);
}

static VectorDims extractOrder(const dnnl::memory::desc& desc) {
    const auto dims = desc.get_dims();
    dnnl::impl::memory_desc_wrapper descWrapped(desc.get());

    if (descWrapped.has_runtime_dims_or_strides()) {
        OPENVINO_THROW("Unexpected: Cannot calculate order from undefined dims or strides");
    }

    const auto& blk_desc = descWrapped.blocking_desc();

    const size_t outer_ndims = dims.size();
    const size_t inner_ndims = blk_desc.inner_nblks;
    const size_t total_ndims = outer_ndims + inner_ndims;

    // total inner block size. in case of 4i16o4i will be {16, 16, 1, 1}
    VectorDims total_block_per_dim(outer_ndims, 1);
    for (size_t i = 0; i < inner_ndims; i++) {
        total_block_per_dim[blk_desc.inner_idxs[i]] *= blk_desc.inner_blks[i];
    }
    VectorDims outer_block_dims(std::begin(dims), std::begin(dims) + outer_ndims);
    for (size_t i = 0; i < outer_block_dims.size(); i++) {
        outer_block_dims[i] = div_up(outer_block_dims[i], total_block_per_dim[i]);
    }

    // order of outer dims. In case of IOhw_ will be {1, 0, 2, 3}
    VectorDims outer_order(outer_ndims);
    std::iota(outer_order.begin(), outer_order.end(), 0);
    std::sort(outer_order.begin(), outer_order.end(), [&blk_desc, &outer_block_dims](size_t ind_l, size_t ind_r) {
        return (blk_desc.strides[ind_l] > blk_desc.strides[ind_r]) ||
               (blk_desc.strides[ind_l] == blk_desc.strides[ind_r] &&
                outer_block_dims[ind_l] > outer_block_dims[ind_r]);
    });

    // blocked order
    // [new_outer_order] U [inner_idxs]
    VectorDims blk_order(total_ndims, 0);
    std::copy(outer_order.begin(), outer_order.end(), blk_order.begin());
    std::copy(blk_desc.inner_idxs, blk_desc.inner_idxs + blk_desc.inner_nblks, blk_order.begin() + dims.size());
    return blk_order;
}

DnnlBlockedMemoryDesc::DnnlBlockedMemoryDesc(const_dnnl_memory_desc_t cdesc)
    : MemoryDesc(DnnlExtensionUtils::convertToVectorDims(cdesc->dims, cdesc->ndims), DnnlBlocked) {
    desc = dnnl::memory::desc(DnnlExtensionUtils::clone_desc(cdesc));

    if (desc.get_format_kind() == dnnl::memory::format_kind::any) {
        OPENVINO_THROW("Unexpected: Memory format any is prohibited!");
    }

    dnnl::impl::memory_desc_wrapper descWrapped(desc.get());
    if (!descWrapped.is_blocking_desc()) {
        OPENVINO_THROW("Unexpected: Can't create DnnlBlockedMemoryDesc from not blocking desc");
    }

    order = extractOrder(desc);

    if (getShape().hasZeroDims()) {
        auto& blk = desc.get()->format_desc.blocking;
        std::fill(std::begin(blk.strides), std::begin(blk.strides) + desc.get()->ndims, 0);
    }

    initBlockedParams();
}

bool DnnlBlockedMemoryDesc::hasLayoutType(LayoutType layoutType) const {
    switch (layoutType) {
    case LayoutType::ncsp:
        return isPlainFormat();
    case LayoutType::nspc:
        return isTailCFormat();
    case LayoutType::nCsp8c:
        return isBlockedCFormat(8);
    case LayoutType::nCsp16c:
        return isBlockedCFormat(16);
    default:
        return false;
    }
}

bool DnnlBlockedMemoryDesc::isPlainFormat() const {
    if (shape.getRank() != order.size()) {
        return false;
    }
    for (size_t i = 0; i < order.size(); ++i) {
        if (order[i] != i) {
            return false;
        }
    }
    return true;
}

bool DnnlBlockedMemoryDesc::isBlockedCFormat(size_t blk_size) const {
    if (desc.get_format_kind() != dnnl::memory::format_kind::blocked || desc.get_inner_nblks() != 1 ||
        desc.get_inner_idxs()[0] != 1) {
        return false;
    }

    if ((order.size() - shape.getRank()) != 1) {
        return false;
    }
    for (size_t i = 0; i < order.size() - 1; ++i) {
        if (order[i] != i) {
            return false;
        }
    }
    if (blk_size != UNREACHABLE_DIM && static_cast<int64_t>(blk_size) != desc.get_inner_blks()[0]) {
        return false;
    }

    return true;
}

bool DnnlBlockedMemoryDesc::isTailCFormat() const {
    if (shape.getRank() < 3) {
        return false;
    }
    if (shape.getRank() != order.size()) {
        return false;
    }
    if (!std::is_sorted(order.begin(), --order.end())) {
        return false;
    }
    if (order.back() != 1) {
        return false;
    }
    return true;
}

template <class Dest, class Src>
std::vector<Dest> convert_to_vector(const Src* source, size_t size) {
    std::vector<Dest> result(size);
    for (size_t i = 0; i < size; i++) {
        result[i] = static_cast<Dest>(source[i]);
    }
    return result;
}

static dnnl::memory::desc cloneDescWithNewDims(const dnnl::memory::desc& desc,
                                               const VectorDims& dims,
                                               const VectorDims& order) {
    using namespace dnnl::impl::utils;
    auto mklDims = DnnlExtensionUtils::convertToDnnlDims(dims);
    const auto offsetPadding = desc.get()->offset0;

    dnnl::memory::desc clonedDesc(DnnlExtensionUtils::clone_desc(desc.get()));

    array_copy(clonedDesc.get()->dims, mklDims.data(), mklDims.size());
    dnnl::memory::dims perm(convert_to_vector<dnnl::memory::dim, size_t>(order.data(), mklDims.size()));
    auto innerBlks = clonedDesc.get_inner_blks();
    auto innerIdxs = clonedDesc.get_inner_idxs();

    auto retCode = dnnl::impl::fill_blocked(*clonedDesc.get(), perm, innerBlks, innerIdxs);
    if (retCode != dnnl::impl::status::success) {
        OPENVINO_THROW("Can not clone DnnlBlockedMemoryDesc with dims: ", dims2str(dims));
    }
    // dnnl::impl::fill_blocked always set offset0 to 0
    // so we need to restore actual value
    clonedDesc.get()->offset0 = offsetPadding;

    return clonedDesc;
}

MemoryDescPtr DnnlBlockedMemoryDesc::cloneWithNewDimsImp(const VectorDims& dims) const {
    if (std::any_of(dims.begin(), dims.end(), [](size_t x) {
            return Shape::UNDEFINED_DIM == x;
        })) {
        OPENVINO_THROW("Can't clone desc if new dims are undefined");
    }

    // TODO [DS]: add stride recalculation for strided blobs
    for (int i = strides.size() - 2; i >= 0; i--) {
        if (strides[i] == Shape::UNDEFINED_DIM) {
            break;
        }

        if (strides[i] != strides[i + 1] * blockedDims[i + 1]) {
            OPENVINO_THROW_NOT_IMPLEMENTED("Can't clone desc with new dims for not dense tensor");
        }
    }

    return DnnlBlockedMemoryDescPtr(new DnnlBlockedMemoryDesc(cloneDescWithNewDims(desc, dims, order).get()));
}

bool DnnlBlockedMemoryDesc::isSame(dnnl::memory::format_tag fmt) const {
    dnnl::memory::desc refDesc(desc.get_dims(), desc.get_data_type(), fmt);

    if (desc.get_ndims() != refDesc.get_ndims()) {
        return false;
    }

    if (desc.get_format_kind() != dnnl::memory::format_kind::blocked ||
        refDesc.get_format_kind() != dnnl::memory::format_kind::blocked) {
        OPENVINO_THROW("DnnlMemoryDesc::isSame is not implemented for non blocked memory format");
    }

    auto actualBlkDesc = desc.get()->format_desc.blocking;
    auto refBlkDesc = refDesc.get()->format_desc.blocking;
    if (desc.get_inner_nblks() != refBlkDesc.inner_nblks) {
        return false;
    }

    for (int i = 0; i < actualBlkDesc.inner_nblks; ++i) {
        if (actualBlkDesc.inner_blks[i] != refBlkDesc.inner_blks[i]) {
            return false;
        }
    }

    for (int i = 0; i < actualBlkDesc.inner_nblks; ++i) {
        if (actualBlkDesc.inner_idxs[i] != refBlkDesc.inner_idxs[i]) {
            return false;
        }
    }

    auto actualStrides = desc.get()->format_desc.blocking.strides;
    auto refStrides = refDesc.get()->format_desc.blocking.strides;

    VectorDims actualOrder(desc.get()->ndims);
    {
        const auto dims = desc.get_dims();
        VectorDims total_block_per_dim(dims.size(), 1);
        const auto& blk_desc = desc.get()->format_desc.blocking;
        for (int i = 0; i < blk_desc.inner_nblks; i++) {
            total_block_per_dim[blk_desc.inner_idxs[i]] *= blk_desc.inner_blks[i];
        }
        VectorDims outer_block_dims(std::begin(dims), std::begin(dims) + dims.size());
        for (size_t i = 0; i < outer_block_dims.size(); i++) {
            outer_block_dims[i] = div_up(outer_block_dims[i], total_block_per_dim[i]);
        }

        std::iota(actualOrder.begin(), actualOrder.end(), 0);
        std::sort(actualOrder.begin(),
                  actualOrder.end(),
                  [&actualStrides, &outer_block_dims](size_t ind_l, size_t ind_r) {
                      return (actualStrides[ind_l] > actualStrides[ind_r]) ||
                             (actualStrides[ind_l] == actualStrides[ind_r] &&
                              outer_block_dims[ind_l] > outer_block_dims[ind_r]);
                  });
    }

    VectorDims refOrder(refDesc.get()->ndims);
    {
        const auto dims = refDesc.get_dims();
        VectorDims total_block_per_dim(dims.size(), 1);
        const auto& blk_desc = refDesc.get()->format_desc.blocking;
        for (int i = 0; i < blk_desc.inner_nblks; i++) {
            total_block_per_dim[blk_desc.inner_idxs[i]] *= blk_desc.inner_blks[i];
        }
        VectorDims outer_block_dims(std::begin(dims), std::begin(dims) + dims.size());
        for (size_t i = 0; i < outer_block_dims.size(); i++) {
            outer_block_dims[i] = div_up(outer_block_dims[i], total_block_per_dim[i]);
        }

        std::iota(refOrder.begin(), refOrder.end(), 0);
        std::sort(refOrder.begin(), refOrder.end(), [&refStrides, &outer_block_dims](size_t ind_l, size_t ind_r) {
            return (refStrides[ind_l] > refStrides[ind_r]) ||
                   (refStrides[ind_l] == refStrides[ind_r] && outer_block_dims[ind_l] > outer_block_dims[ind_r]);
        });
    }

    if (actualOrder != refOrder) {
        return false;
    }
    return true;
}

size_t DnnlBlockedMemoryDesc::getMaxMemSize() const {
    if (shape.isStatic() || shape.hasZeroDims()) {
        return getCurrentMemSize();
    }

    const auto& maxDims = shape.getMaxDims();
    if (std::any_of(maxDims.begin(), maxDims.end(), [](size_t x) {
            return Shape::UNDEFINED_DIM == x;
        })) {
        return UNDEFINED_SIZE;
    }

    auto maxDimsDesc = cloneWithNewDims(maxDims);
    return maxDimsDesc->getCurrentMemSize();
}

size_t DnnlBlockedMemoryDesc::getPaddedElementsCount() const {
    if (getShape().hasZeroDims()) {
        return 0;
    }

    auto padded_dims = desc.get_padded_dims();
    if (std::any_of(std::begin(padded_dims), std::begin(padded_dims) + desc.get_ndims(), [](dnnl_dim_t dim) {
            return dim == DNNL_RUNTIME_DIM_VAL;
        })) {
        OPENVINO_THROW("Can't compute padded elements count for non undefined blocked dims");
    }
    return std::accumulate(std::begin(padded_dims),
                           std::begin(padded_dims) + desc.get_ndims(),
                           size_t{1},
                           std::multiplies<>());
}

bool DnnlBlockedMemoryDesc::blocksExtended() const {
    const auto padded_dims = desc.get_padded_dims();
    const auto dims = desc.get_dims();
    for (int i = 0; i < desc.get_ndims(); i++) {
        if (dims[i] != padded_dims[i]) {
            return true;
        }
    }
    return false;
}

void DnnlBlockedMemoryDesc::initBlockDims() {
    const auto dims = desc.get_dims();

    const size_t outer_ndims = dims.size();
    const auto inner_ndims = desc.get_inner_nblks();
    const size_t total_ndims = outer_ndims + inner_ndims;

    // total inner block size. in case of 4i16o4i will be {16, 16, 1, 1}
    VectorDims total_block_per_dim(outer_ndims, 1);
    const auto inner_idxs = desc.get_inner_idxs();
    const auto inner_blks = desc.get_inner_blks();
    const auto inner_nblks = desc.get_inner_nblks();

    for (int i = 0; i < inner_ndims; i++) {
        total_block_per_dim[inner_idxs[i]] *= inner_blks[i];
    }
    // blocked dims
    // [dims via new_outer_order with auto pad] U [inner_blk_dims]
    VectorDims outer_block_dims = DnnlExtensionUtils::convertToVectorDims(dims);
    for (size_t i = 0; i < outer_block_dims.size(); i++) {
        if (outer_block_dims[i] != Shape::UNDEFINED_DIM) {
            outer_block_dims[i] = div_up(outer_block_dims[i], total_block_per_dim[i]);
        }
    }

    // order of outer dims. In case of IOhw_ will be {1, 0, 2, 3}
    VectorDims outer_order(outer_ndims);
    std::copy(order.begin(), order.begin() + outer_ndims, outer_order.begin());

    blockedDims.resize(total_ndims, 0);
    std::copy(inner_blks.begin(), inner_blks.begin() + inner_nblks, blockedDims.end() - inner_nblks);
    std::transform(outer_order.begin(), outer_order.end(), blockedDims.begin(), [&](size_t i) {
        return outer_block_dims[i];
    });
}

void DnnlBlockedMemoryDesc::initStrides() {
    const auto dims = desc.get_dims();

    const size_t outer_ndims = dims.size();
    const size_t inner_nblks = desc.get_inner_nblks();
    const auto inner_blks = desc.get_inner_blks();
    const size_t total_ndims = outer_ndims + inner_nblks;

    // strides of inner dims. In case of 4i16o4i will be {64, 4, 1}
    VectorDims inner_strides(inner_nblks, getShape().hasZeroDims() ? 0 : 1);
    for (size_t i = 1; i < inner_nblks; i++) {
        inner_strides[inner_nblks - 1 - i] = inner_strides[inner_nblks - i] * inner_blks[inner_nblks - i];
    }

    // order of outer dims. In case of IOhw_ will be {1, 0, 2, 3}
    VectorDims outer_order(outer_ndims);
    std::copy(order.begin(), order.begin() + outer_ndims, outer_order.begin());

    // blocked strides
    // [outer_strides via new_outer_order] U [inner_strides]
    strides.resize(total_ndims, 0);
    std::copy(inner_strides.rbegin(), inner_strides.rend(), strides.rbegin());

    const auto desc_strides = desc.get_strides();
    std::transform(outer_order.begin(), outer_order.end(), strides.begin(), [&](size_t i) {
        return desc_strides[i] == DNNL_RUNTIME_DIM_VAL ? Shape::UNDEFINED_DIM : desc_strides[i];
    });
}

void DnnlBlockedMemoryDesc::initOffsetPadding() {
    const auto& padded_offset = desc.get()->padded_offsets;
    offsetPaddingToData = VectorDims(std::begin(padded_offset), std::begin(padded_offset) + getOrder().size());
}

MemoryDescPtr DnnlBlockedMemoryDesc::cloneWithNewPrecision(const ov::element::Type prec) const {
    auto newDesc = std::make_shared<DnnlBlockedMemoryDesc>(*this);
    newDesc->setPrecision(prec);

    return newDesc;
}

void DnnlBlockedMemoryDesc::recomputeDefaultStrides() {
    const auto& rank = getShape().getRank();

    if (order.size() != blockedDims.size()) {
        OPENVINO_THROW("Can't recompute stride: order size != blocked dims size");
    }

    auto& oneDnnStrides = desc.get()->format_desc.blocking.strides;
    if (getShape().hasZeroDims()) {
        std::fill(std::begin(oneDnnStrides), std::begin(oneDnnStrides) + getShape().getRank(), 0);
    } else if (std::any_of(blockedDims.begin(), blockedDims.end(), [](Dim val) {
                   return val == Shape::UNDEFINED_DIM;
               })) {
        std::fill(std::begin(oneDnnStrides), std::begin(oneDnnStrides) + rank, DNNL_RUNTIME_DIM_VAL);
        initStrides();
    } else {
        strides.resize(order.size());
        strides[order.size() - 1] = 1;
        for (size_t i = 2; i <= order.size(); i++) {
            strides[order.size() - i] = strides[order.size() - (i - 1)] * blockedDims[blockedDims.size() - (i - 1)];
        }
        for (size_t i = 0; i < rank; i++) {
            oneDnnStrides[order[i]] = strides[i];
        }
    }
}

DnnlBlockedMemoryDesc::DnnlBlockedMemoryDesc(const dnnl::memory::desc& mdesc, const Shape& shape)
    : MemoryDesc(shape, DnnlBlocked) {
    if (mdesc.get_format_kind() == dnnl::memory::format_kind::any) {
        OPENVINO_THROW("Unexpected: Memory format any is prohibited!");
    }

    dnnl::impl::memory_desc_wrapper descWrapped(mdesc.get());
    if (!descWrapped.is_blocking_desc()) {
        OPENVINO_THROW("Unexpected: Can't create DnnlBlockedMemoryDesc from not blocking desc");
    }

    if (!shape.isCompatible(DnnlExtensionUtils::convertToVectorDims(mdesc.get_dims()))) {
        OPENVINO_THROW("ParameterMismatch: Can not create DnnlBlockedMemoryDesc. memory::desc dims: ",
                       vec2str(mdesc.get_dims()),
                       " are incompatible with provided shape: ",
                       shape.toString(),
                       ".");
    }

    order = extractOrder(mdesc);

    desc = cloneDescWithNewDims(mdesc, shape.getDims(), order);

    if (shape.hasZeroDims()) {
        auto& blk = desc.get()->format_desc.blocking;
        std::fill(std::begin(blk.strides), std::begin(blk.strides) + desc.get()->ndims, 0);
    }

    initBlockedParams();
}

std::string DnnlBlockedMemoryDesc::serializeFormat() const {
    return BlockedMemoryDesc::serializeFormat();
}

}  // namespace ov::intel_cpu
