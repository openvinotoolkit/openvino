// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_desc/dnnl_blocked_memory_desc.h"
#include <dnnl_types.h>
#include <common/memory_desc_wrapper.hpp>

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

DnnlBlockedMemoryDesc::DnnlBlockedMemoryDesc(InferenceEngine::Precision prc, const Shape& shape) : MemoryDesc(shape, DnnlBlocked) {
    const auto ndims = shape.getRank();
    const auto &dims = shape.getDims();
    mkldnn::memory::dims plain_strides;
    if (std::any_of(dims.begin(), dims.end(), [](size_t val) { return val == Shape::UNDEFINED_DIM; })) {
        plain_strides.resize(ndims, DNNL_RUNTIME_DIM_VAL);
    } else {
        plain_strides.resize(ndims, 1);
        for (size_t i = 1; i < ndims; i++) {
            plain_strides[ndims - i -1] = plain_strides[ndims - i] * dims[ndims - i];
        }
    }

    desc = {MKLDNNExtensionUtils::convertToDnnlDims(dims), MKLDNNExtensionUtils::IEPrecisionToDataType(prc), plain_strides};

    order.resize(ndims);
    std::iota(order.begin(), order.end(), 0);

    initBlockedParams();
}

/**
 * Construct from blocked parameters
 *
 * IE  IOhw_4i16o4i   dims(N) = {32, 64, 128, 128}
 *   blockedDims  {4, 2, 128, 128, 4, 16, 4}                      // total dims(inner, outermost, auto blocked/padded). Generally sorted by strides.
 *   strides      {8388608, 4194304,  32768, 256, 64,  4, 1}      // strides for blockedDims, growing sequence
 *   order        {1, 0,   2,   3, 1,  0, 1}                      // matching to original dims
 *
 *   All vectors blockedDims/strides/order have same size equals total num of internal blocked dims(inner_dims + outer_dims)
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
DnnlBlockedMemoryDesc::DnnlBlockedMemoryDesc(InferenceEngine::Precision prc, const Shape& shape, const VectorDims& blockedDims,
                                                 const VectorDims& order, size_t offsetPadding, const VectorDims& offsetPaddingToData,
                                                 const VectorDims& strides) : MemoryDesc(shape, DnnlBlocked) {
    using namespace mkldnn;
    // scalar case
    if (shape.getRank() == 0) {
        desc.data.format_kind = dnnl_blocked;
        desc.data.data_type = memory::convert_to_c(MKLDNNExtensionUtils::IEPrecisionToDataType(prc));
        desc.data.ndims = 1;
        desc.data.dims[0] = 1;
        desc.data.padded_dims[0] = 1;
        desc.data.format_desc.blocking.strides[0] = 1;
        desc.data.padded_offsets[0] = 0;
        desc.data.offset0 = MKLDNNExtensionUtils::convertToDnnlDim(offsetPadding);
        return;
    }

    if (order.size() != blockedDims.size()) {
        IE_THROW() << "Can not construct DnnlBlockedMemoryDesc, order and blocked dims must have equals size";
    }

    if (!offsetPaddingToData.empty() && offsetPaddingToData.size() != order.size()) {
        IE_THROW() << "Can not construct DnnlBlockedMemoryDesc, offsetPaddingToData must have equal size with order and blocked dims";
    }

    if (!strides.empty() && strides.size() != order.size()) {
        IE_THROW() << "Can not construct DnnlBlockedMemoryDesc, strides must have equal size with order and blocked dims";
    }

    if (std::any_of(order.begin(), order.end(), [](size_t val) { return val == Shape::UNDEFINED_DIM; })) {
        IE_THROW() << "DnnlBlockedMemoryDesc doesn't support undefined order.";
    }

    if (std::any_of(blockedDims.begin() + shape.getRank(), blockedDims.end(), [](size_t val) { return val == Shape::UNDEFINED_DIM; })) {
        IE_THROW() << "DnnlBlockedMemoryDesc doesn't support undefined blockedDims.";
    }

    auto dims = MKLDNNExtensionUtils::convertToDnnlDims(shape.getDims());

    size_t outer_ndims = dims.size();
    size_t inner_ndims = order.size() - dims.size();

    if (!strides.empty()) {
        bool is_descending_strides = true;
        for (int i = 1; i < strides.size(); i++) {
            is_descending_strides &= (strides[i - 1] >= strides[i]);
        }

        // TODO: That's strong constrains and can be mitigated. IE::TensorDesc allow to transpose blocked dims
        //       and may be we can achieve correct "descending strides" form which allow conversion.
        if (!is_descending_strides)
            IE_THROW() << "Can not construct DnnlBlockedMemoryDesc from strides: " << vec2str(strides);
    }

    if (!strides.empty() && std::none_of(strides.begin(), strides.end(), [](size_t x) { return Shape::UNDEFINED_DIM == x; })) {
        bool inner_block_are_dense = one_of(strides.back(), 0, 1);  // stride 1 - is dense case, 0 - broad casted
        for (int i = outer_ndims; i < strides.size() - 1; i++) {
            inner_block_are_dense &= (strides[i] == strides[i + 1] * blockedDims[i + 1]);
        }

        if (!inner_block_are_dense)
            IE_THROW() << "Can not construct DnnlBlockedMemoryDesc from strides: " << vec2str(strides) << " inner blocks are not dense.";
    }

    // Fill general memory desc fields
    desc.data.format_kind = dnnl_blocked;
    desc.data.extra.flags = 0;
    desc.data.data_type = memory::convert_to_c(MKLDNNExtensionUtils::IEPrecisionToDataType(prc));
    desc.data.ndims = dims.size();
    desc.data.offset0 = MKLDNNExtensionUtils::convertToDnnlDim(offsetPadding);
    std::copy(dims.begin(), dims.end(), desc.data.dims);

    if (!offsetPaddingToData.empty()) {
        bool inner_pad_offsets_is_zero = std::all_of(offsetPaddingToData.begin() + outer_ndims, offsetPaddingToData.end(),
                                                     [](size_t pad) { return pad == 0; });

        if (!inner_pad_offsets_is_zero)
            IE_THROW() << "Can not construct DnnlBlockedMemoryDesc, inner pad offsets is not zero: " << vec2str(offsetPaddingToData);
        auto dnnlPaddedOffsets = MKLDNNExtensionUtils::convertToDnnlDims(offsetPaddingToData);
        std::copy(dnnlPaddedOffsets.begin(), dnnlPaddedOffsets.begin() + outer_ndims, desc.data.padded_offsets);
    } else {
        std::fill(std::begin(desc.data.padded_offsets), std::begin(desc.data.padded_offsets) + outer_ndims, 0);
    }

    std::fill(desc.data.padded_dims, desc.data.padded_dims + outer_ndims, 1);
    auto dnnlBlkDims = MKLDNNExtensionUtils::convertToDnnlDims(blockedDims);

    for (size_t i = 0; i < order.size(); i++) {
        auto idx = order[i];
        if (desc.data.padded_dims[idx] != DNNL_RUNTIME_DIM_VAL && dnnlBlkDims[i] != DNNL_RUNTIME_DIM_VAL) {
            desc.data.padded_dims[idx] *= dnnlBlkDims[i];
        } else {
            desc.data.padded_dims[idx] = DNNL_RUNTIME_DIM_VAL;
        }
    }

    // Fill blocking desc
    auto &dnn_blk_desc = desc.data.format_desc.blocking;
    dnn_blk_desc.inner_nblks = inner_ndims;
    std::copy(dnnlBlkDims.end() - inner_ndims, dnnlBlkDims.end(), dnn_blk_desc.inner_blks);
    std::copy(order.end() - inner_ndims, order.end(), dnn_blk_desc.inner_idxs);

    this->order = order;
    initBlockDims();
    initOffsetPadding();

    if (strides.empty()) {
        this->recomputeStrides();
    } else {
        for (size_t i = 0; i < outer_ndims; i++) {
            auto dnnlStrides = MKLDNNExtensionUtils::convertToDnnlDims(strides);
            dnn_blk_desc.strides[order[i]] = dnnlStrides[i];
        }
        initStrides();
    }
}

DnnlBlockedMemoryDesc::DnnlBlockedMemoryDesc(const Shape& shape, mkldnn::memory::data_type dataType, mkldnn::memory::format_tag format) :
        MemoryDesc(shape, DnnlBlocked) {
    using namespace mkldnn;
    if (format == memory::format_tag::any || format == memory::format_tag::undef)
        IE_THROW(Unexpected) << "Can't create mkldnn::desc with any or undef format";

    const auto dims = shape.getDims();
    if (format == memory::format_tag::x && shape.getRank() == 0) {
        desc = mkldnn::memory::desc(mkldnn::memory::dims(1, 1), dataType, format);
    } else {
        desc = mkldnn::memory::desc(MKLDNNExtensionUtils::convertToDnnlDims(dims), dataType, format);
    }

    VectorDims perm;
    VectorDims inner_blks;
    VectorDims inner_idxs;

    mkldnn::impl::memory_desc_wrapper::compute_blocking(mkldnn::memory::convert_to_c(format), perm, inner_blks, inner_idxs);

    order.swap(perm);
    order.insert(order.end(), inner_idxs.begin(), inner_idxs.end());

    initBlockedParams();
}

bool DnnlBlockedMemoryDesc::isCompatible(const MemoryDesc& rhs) const {
    if (auto desc = dynamic_cast<const DnnlBlockedMemoryDesc*>(&rhs)) {
        return isCompatible(*desc);
    } else if (auto desc = dynamic_cast<const CpuBlockedMemoryDesc*>(&rhs)) {
        return isCompatible(*desc);
    } else {
        return false;
    }
}

bool DnnlBlockedMemoryDesc::isCompatible(const CpuBlockedMemoryDesc& rhs) const {
    return this->desc.data.extra.flags == dnnl_memory_extra_flag_none && BlockedMemoryDesc::isCompatible(rhs);
}

bool DnnlBlockedMemoryDesc::isCompatible(const DnnlBlockedMemoryDesc& rhs) const {
    using namespace dnnl;
    using namespace impl;
    using namespace impl::utils;
    if (this->getShape() != rhs.getShape() || this->getPrecision() != rhs.getPrecision()) {
        return false;
    }

    if (this->desc == rhs.desc) {
        return true;
    }
    memory_desc_wrapper wrappedThis(this->desc.data);
    memory_desc_wrapper wrappedRhs(rhs.desc.data);
    if (one_of(wrappedThis.format_kind(), format_kind::undef, format_kind::any))
        return false;

    int stride_start = wrappedThis.ndims() > 0 && wrappedThis.dims()[0] == 1 ? 1 : 0;  // ignore batch axis stride if batch size == 1

    const auto thisExtra = this->desc.data.extra;
    const auto rhsExtra = rhs.desc.data.extra;
    return this->getOrder() == rhs.getOrder() && (thisExtra.flags == rhsExtra.flags && thisExtra.compensation_mask == rhsExtra.compensation_mask &&
           thisExtra.scale_adjust == rhsExtra.scale_adjust) && wrappedThis.similar_to(wrappedRhs, true, true, 0, stride_start, true, true);
}

DnnlBlockedMemoryDesc::DnnlBlockedMemoryDesc(const mkldnn::memory::desc& mdesc) :
                MemoryDesc(MKLDNNExtensionUtils::convertToVectorDims(mdesc.dims()), DnnlBlocked) {
    desc = mdesc;
    if (desc.data.format_kind == dnnl::impl::format_kind::any)
        IE_THROW(Unexpected) << "Memory format any is prohibited!";

    mkldnn::impl::memory_desc_wrapper descWrapped(desc.data);
    if (!descWrapped.is_blocking_desc())
        IE_THROW(Unexpected) << "Can't create DnnlBlockedMemoryDesc from not blocking desc";

    if (descWrapped.has_runtime_dims_or_strides()) {
        IE_THROW(Unexpected) << "Cannot calculate order from undefined dims or strides";
    }

    const auto dims = desc.dims();

    const auto &blk_desc = descWrapped.blocking_desc();

    const size_t outer_ndims = dims.size();
    const size_t inner_ndims = blk_desc.inner_nblks;
    const size_t total_ndims = outer_ndims + inner_ndims;

    // total inner block size. in case of 4i16o4i will be {16, 16, 1, 1}
    VectorDims total_block_per_dim(outer_ndims, 1);
    for (int i = 0; i < inner_ndims; i++) {
        total_block_per_dim[blk_desc.inner_idxs[i]] *= blk_desc.inner_blks[i];
    }
    VectorDims outer_block_dims(std::begin(dims), std::begin(dims) + outer_ndims);
    for (size_t i = 0; i < outer_block_dims.size(); i++) {
        outer_block_dims[i] = div_up(outer_block_dims[i], total_block_per_dim[i]);
    }

    // order of outer dims. In case of IOhw_ will be {1, 0, 2, 3}
    VectorDims outer_order(outer_ndims);
    std::iota(outer_order.begin(), outer_order.end(), 0);
    std::sort(outer_order.begin(), outer_order.end(),
              [&blk_desc, &outer_block_dims](size_t ind_l, size_t ind_r) {
                  return (blk_desc.strides[ind_l] > blk_desc.strides[ind_r]) ||
                         (blk_desc.strides[ind_l] == blk_desc.strides[ind_r] && outer_block_dims[ind_l] > outer_block_dims[ind_r]);
              });


    // blocked order
    // [new_outer_order] U [inner_idxs]
    SizeVector blk_order(total_ndims, 0);
    std::copy(outer_order.begin(), outer_order.end(), blk_order.begin());
    std::copy(blk_desc.inner_idxs, blk_desc.inner_idxs + blk_desc.inner_nblks, blk_order.begin() + dims.size());
    order.swap(blk_order);

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
    const auto &blocking = desc.data.format_desc.blocking;

    if (desc.data.format_kind !=dnnl_blocked ||
        blocking.inner_nblks != 1 ||
        blocking.inner_idxs[0] != 1)
        return false;

    if ((order.size() - shape.getRank()) != 1) {
        return false;
    }
    for (size_t i = 0; i < order.size() - 1; ++i) {
        if (order[i] != i) {
            return false;
        }
    }
    if (blk_size != UNREACHABLE_DIM && blk_size != blocking.inner_blks[0]) {
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

MemoryDescPtr DnnlBlockedMemoryDesc::cloneWithNewDimsImp(const VectorDims &dims) const {
    if (std::any_of(dims.begin(), dims.end(), [](size_t x){ return Shape::UNDEFINED_DIM == x; })) {
        IE_THROW() << "Can't clone desc if new dims are undefined";
    }

    // TODO [DS]: add stride recalculation for strided blobs
    getStrides();
    getBlockDims();
    for (int i = strides.size() - 2; i >= 0 ; i--) {
        if (strides[i] == Shape::UNDEFINED_DIM)
            break;

        if (strides[i] != strides[i + 1] * blockedDims[i + 1])
            IE_THROW(NotImplemented) << "Can't clone desc with new dims for not dense tensor";
    }

    using namespace dnnl::impl::utils;
    auto mklDims = MKLDNNExtensionUtils::convertToDnnlDims(dims);
    mkldnn::memory::desc newMklDesc = desc;
    array_copy(newMklDesc.data.dims, mklDims.data(), mklDims.size());
    std::vector<int> perm(order.begin(), order.begin() + mklDims.size());
    auto& blockingDesc = newMklDesc.data.format_desc.blocking;
    auto numInnerBlks = blockingDesc.inner_nblks;
    std::vector<int> innerBlks(std::begin(blockingDesc.inner_blks), std::begin(blockingDesc.inner_blks) + numInnerBlks);
    std::vector<int> innerIdxs(std::begin(blockingDesc.inner_idxs), std::begin(blockingDesc.inner_idxs) + numInnerBlks);
    auto retCode = dnnl::impl::fill_blocked(newMklDesc.data, perm, innerBlks, innerIdxs);
    if (retCode != dnnl::impl::status::success) {
        IE_THROW() << "Can not clone DnnlBlockedMemoryDesc with dims: " << MemoryDescUtils::dims2str(dims);
    }
    return DnnlBlockedMemoryDescPtr(new DnnlBlockedMemoryDesc(newMklDesc));
}

static const std::map<int, std::vector<mkldnn::memory::format_tag>> form_tags_by_ndims {
    {0, {
        mkldnn::memory::format_tag::a   // TODO :: really 1d layout for scalar??
     }}, {1, {
        mkldnn::memory::format_tag::a
     }}, {2, {
        mkldnn::memory::format_tag::ab,
        mkldnn::memory::format_tag::ba
     }}, {3, {
        mkldnn::memory::format_tag::abc,
        mkldnn::memory::format_tag::acb,
        mkldnn::memory::format_tag::bac,
        mkldnn::memory::format_tag::bca,
        mkldnn::memory::format_tag::cba,

        mkldnn::memory::format_tag::Abc16a,
        mkldnn::memory::format_tag::ABc16a16b,
        mkldnn::memory::format_tag::ABc4a4b,
        mkldnn::memory::format_tag::aBc16b,
        mkldnn::memory::format_tag::aBc32b,
        mkldnn::memory::format_tag::ABc16b16a,
        mkldnn::memory::format_tag::Abc4a,
        mkldnn::memory::format_tag::aBc4b,
        mkldnn::memory::format_tag::ABc4b16a4b,
        mkldnn::memory::format_tag::ABc2b8a4b,
        mkldnn::memory::format_tag::ABc16b16a4b,
        mkldnn::memory::format_tag::ABc16b16a2b,
        mkldnn::memory::format_tag::ABc4b4a,
        mkldnn::memory::format_tag::ABc8a16b2a,
        mkldnn::memory::format_tag::ABc8a8b,
        mkldnn::memory::format_tag::ABc8a4b,
        mkldnn::memory::format_tag::aBc8b,
        mkldnn::memory::format_tag::ABc8b16a2b,
        mkldnn::memory::format_tag::ABc8b8a,
        mkldnn::memory::format_tag::Acb16a,
        mkldnn::memory::format_tag::Acb4a,
        mkldnn::memory::format_tag::Acb8a,
        mkldnn::memory::format_tag::BAc16a16b,
        mkldnn::memory::format_tag::BAc16b16a,
     }}, {4, {                                 // Popular
        mkldnn::memory::format_tag::abcd,      // plain
        mkldnn::memory::format_tag::acdb,      // tail_c
        mkldnn::memory::format_tag::aBcd8b,    // blocked 8c
        mkldnn::memory::format_tag::aBcd16b,   // blocked 16c

        mkldnn::memory::format_tag::abdc,

        mkldnn::memory::format_tag::bacd,
        mkldnn::memory::format_tag::bcda,
        mkldnn::memory::format_tag::cdba,
        mkldnn::memory::format_tag::dcab,

        mkldnn::memory::format_tag::Abcd8a,
        mkldnn::memory::format_tag::Abcd16a,
        mkldnn::memory::format_tag::Abcd32a,
        mkldnn::memory::format_tag::ABcd16a16b,
        mkldnn::memory::format_tag::aBcd32b,
        mkldnn::memory::format_tag::ABcd16b16a,
        mkldnn::memory::format_tag::aBCd16b16c,
        mkldnn::memory::format_tag::aBCd16c16b,
        mkldnn::memory::format_tag::Abcd4a,
        mkldnn::memory::format_tag::aBcd4b,
        mkldnn::memory::format_tag::ABcd4b16a4b,
        mkldnn::memory::format_tag::ABcd2b8a4b,
        mkldnn::memory::format_tag::ABcd4b4a,
        mkldnn::memory::format_tag::ABcd4a4b,
        mkldnn::memory::format_tag::aBCd4c16b4c,
        mkldnn::memory::format_tag::aBCd2c8b4c,
        mkldnn::memory::format_tag::ABcd16b16a4b,
        mkldnn::memory::format_tag::ABcd16b16a2b,
        mkldnn::memory::format_tag::aBCd16c16b4c,
        mkldnn::memory::format_tag::aBCd16c16b2c,
        mkldnn::memory::format_tag::aBCd4c4b,
        mkldnn::memory::format_tag::aBCd4b4c,
        mkldnn::memory::format_tag::ABcd8a16b2a,
        mkldnn::memory::format_tag::ABcd8a8b,
        mkldnn::memory::format_tag::ABcd8a32b,
        mkldnn::memory::format_tag::ABcd32a32b,
        mkldnn::memory::format_tag::ABcd8a4b,

        mkldnn::memory::format_tag::ABcd8b16a2b,
        mkldnn::memory::format_tag::aBCd8b16c2b,
        mkldnn::memory::format_tag::ABcd8b8a,
        mkldnn::memory::format_tag::aBCd8b8c,
        mkldnn::memory::format_tag::aBCd8b4c,
        mkldnn::memory::format_tag::aBCd8c16b2c,
        mkldnn::memory::format_tag::aBCd8c8b,

        mkldnn::memory::format_tag::ABcd4a8b8a4b,
        mkldnn::memory::format_tag::ABcd2a8b8a2b,

        mkldnn::memory::format_tag::aBdc16b,
        mkldnn::memory::format_tag::aBdc4b,
        mkldnn::memory::format_tag::aBdc8b,
        mkldnn::memory::format_tag::aCBd16b16c,
        mkldnn::memory::format_tag::aCBd16c16b,
        mkldnn::memory::format_tag::Acdb16a,
        mkldnn::memory::format_tag::Acdb4a,
        mkldnn::memory::format_tag::Acdb8a,
        mkldnn::memory::format_tag::BAcd16a16b,
        mkldnn::memory::format_tag::BAcd16b16a,
        mkldnn::memory::format_tag::ABcd32a32b,
        mkldnn::memory::format_tag::Acdb32a,
        mkldnn::memory::format_tag::aBCd2b4c2b,
        mkldnn::memory::format_tag::aBCd2c4b2c,
        mkldnn::memory::format_tag::aBCd4b8c2b,
        mkldnn::memory::format_tag::aBCd4c8b2c,
    }}, {5, {                                   // Popular
        mkldnn::memory::format_tag::abcde,      // plain
        mkldnn::memory::format_tag::acdeb,      // tail_c
        mkldnn::memory::format_tag::aBcde8b,    // blocked 8c
        mkldnn::memory::format_tag::aBcde16b,   // blocked 16c

        mkldnn::memory::format_tag::abdec,
        mkldnn::memory::format_tag::acbde,
        mkldnn::memory::format_tag::bacde,
        mkldnn::memory::format_tag::bcdea,
        mkldnn::memory::format_tag::cdeba,
        mkldnn::memory::format_tag::decab,

        mkldnn::memory::format_tag::Abcde16a,
        mkldnn::memory::format_tag::Abcde32a,
        mkldnn::memory::format_tag::ABcde16a16b,
        mkldnn::memory::format_tag::aBcde32b,
        mkldnn::memory::format_tag::ABcde16b16a,
        mkldnn::memory::format_tag::aBCde16b16c,
        mkldnn::memory::format_tag::aBCde16c16b,
        mkldnn::memory::format_tag::aBCde2c8b4c,
        mkldnn::memory::format_tag::Abcde4a,
        mkldnn::memory::format_tag::aBcde4b,
        mkldnn::memory::format_tag::ABcde4b4a,
        mkldnn::memory::format_tag::ABcde4a4b,
        mkldnn::memory::format_tag::aBCde4b4c,
        mkldnn::memory::format_tag::aBCde4c16b4c,
        mkldnn::memory::format_tag::aBCde16c16b4c,
        mkldnn::memory::format_tag::aBCde16c16b2c,
        mkldnn::memory::format_tag::aBCde4c4b,
        mkldnn::memory::format_tag::Abcde8a,
        mkldnn::memory::format_tag::ABcde8a8b,
        mkldnn::memory::format_tag::ABcde8a4b,
        mkldnn::memory::format_tag::ABcde8b16a2b,
        mkldnn::memory::format_tag::ABcde4b16a4b,
        mkldnn::memory::format_tag::ABcde2b8a4b,
        mkldnn::memory::format_tag::aBCde8b16c2b,
        mkldnn::memory::format_tag::ABcde8b8a,
        mkldnn::memory::format_tag::aBCde8b8c,
        mkldnn::memory::format_tag::aBCde8b4c,
        mkldnn::memory::format_tag::aBCde4b8c8b4c,
        mkldnn::memory::format_tag::aBCde2b8c8b2c,
        mkldnn::memory::format_tag::aBCde8c16b2c,
        mkldnn::memory::format_tag::aBCde8c8b,
        mkldnn::memory::format_tag::aBdec16b,
        mkldnn::memory::format_tag::aBdec4b,
        mkldnn::memory::format_tag::aBdec8b,
        mkldnn::memory::format_tag::aCBde16b16c,
        mkldnn::memory::format_tag::aCBde16c16b,
        mkldnn::memory::format_tag::Acdeb16a,
        mkldnn::memory::format_tag::Acdeb4a,
        mkldnn::memory::format_tag::Acdeb8a,
        mkldnn::memory::format_tag::BAcde16b16a,
        mkldnn::memory::format_tag::BAcde16a16b,
        mkldnn::memory::format_tag::aBdec32b,
        mkldnn::memory::format_tag::aBCde2b4c2b,
        mkldnn::memory::format_tag::aBCde2c4b2c,
        mkldnn::memory::format_tag::aBCde4b8c2b,
        mkldnn::memory::format_tag::aBCde4c8b2c,
    }}, {6, {                                    // Popular
        mkldnn::memory::format_tag::abcdef,      // plain
        mkldnn::memory::format_tag::acbdef,      // permute
        mkldnn::memory::format_tag::defcab,      // permute
        mkldnn::memory::format_tag::aBcdef16b,   // blocked 16c

        mkldnn::memory::format_tag::aBCdef16b16c,
        mkldnn::memory::format_tag::aBCdef16c16b,
        mkldnn::memory::format_tag::aBcdef4b,
        mkldnn::memory::format_tag::aBCdef2c8b4c,
        mkldnn::memory::format_tag::aBCdef4c4b,
        mkldnn::memory::format_tag::aBCdef4b4c,
        mkldnn::memory::format_tag::aBCdef8b8c,
        mkldnn::memory::format_tag::aBCdef8b4c,
        mkldnn::memory::format_tag::aBCdef8c16b2c,
        mkldnn::memory::format_tag::aBCdef4c16b4c,
        mkldnn::memory::format_tag::aBCdef8c8b,

        mkldnn::memory::format_tag::aBdefc16b,
        mkldnn::memory::format_tag::aCBdef16c16b,
        mkldnn::memory::format_tag::aCBdef16b16c,
        mkldnn::memory::format_tag::aBdefc4b,
        mkldnn::memory::format_tag::aBdefc8b,

        mkldnn::memory::format_tag::Abcdef4a,
        mkldnn::memory::format_tag::Abcdef8a,
        mkldnn::memory::format_tag::Abcdef16a,
        mkldnn::memory::format_tag::Abcdef32a,
        mkldnn::memory::format_tag::aBCdef2b4c2b,
        mkldnn::memory::format_tag::aBCdef2c4b2c,
        mkldnn::memory::format_tag::aBCdef4b8c2b,
        mkldnn::memory::format_tag::aBCdef4c8b2c,
        }}
};

bool DnnlBlockedMemoryDesc::isSame(mkldnn::memory::format_tag fmt) const {
    mkldnn::memory::desc refDesc(desc.dims(), desc.data_type(), fmt);

    if (desc.data.ndims != refDesc.data.ndims)
        return false;

    if (desc.data.format_kind != dnnl_blocked || refDesc.data.format_kind != dnnl_blocked)
        IE_THROW() << "DnnlMemoryDesc::isSame is not implemented for non blocked memory format";

    auto actualBlkDesc = desc.data.format_desc.blocking;
    auto refBlkDesc = refDesc.data.format_desc.blocking;
    if (actualBlkDesc.inner_nblks != refBlkDesc.inner_nblks)
        return false;

    for (size_t i = 0; i < actualBlkDesc.inner_nblks; ++i)
        if (actualBlkDesc.inner_blks[i] != refBlkDesc.inner_blks[i])
            return false;

    for (size_t i = 0; i < actualBlkDesc.inner_nblks; ++i)
        if (actualBlkDesc.inner_idxs[i] != refBlkDesc.inner_idxs[i])
            return false;

    auto actualStrides = desc.data.format_desc.blocking.strides;
    auto refStrides = refDesc.data.format_desc.blocking.strides;

    VectorDims actualOrder(desc.data.ndims);
    {
        const auto dims = desc.dims();
        VectorDims total_block_per_dim(dims.size(), 1);
        const auto &blk_desc = desc.data.format_desc.blocking;
        for (int i = 0; i < blk_desc.inner_nblks; i++) {
            total_block_per_dim[blk_desc.inner_idxs[i]] *= blk_desc.inner_blks[i];
        }
        VectorDims outer_block_dims(std::begin(dims), std::begin(dims) + dims.size());
        for (size_t i = 0; i < outer_block_dims.size(); i++) {
            outer_block_dims[i] = div_up(outer_block_dims[i], total_block_per_dim[i]);
        }

        std::iota(actualOrder.begin(), actualOrder.end(), 0);
        std::sort(actualOrder.begin(), actualOrder.end(),
                  [&actualStrides, &outer_block_dims] (size_t ind_l, size_t ind_r) {
                      return (actualStrides[ind_l] > actualStrides[ind_r]) ||
                             (actualStrides[ind_l] == actualStrides[ind_r] && outer_block_dims[ind_l] > outer_block_dims[ind_r]);
                  });
    }

    VectorDims refOrder(refDesc.data.ndims);
    {
        const auto dims = refDesc.dims();
        VectorDims total_block_per_dim(dims.size(), 1);
        const auto &blk_desc = refDesc.data.format_desc.blocking;
        for (int i = 0; i < blk_desc.inner_nblks; i++) {
            total_block_per_dim[blk_desc.inner_idxs[i]] *= blk_desc.inner_blks[i];
        }
        VectorDims outer_block_dims(std::begin(dims), std::begin(dims) + dims.size());
        for (size_t i = 0; i < outer_block_dims.size(); i++) {
            outer_block_dims[i] = div_up(outer_block_dims[i], total_block_per_dim[i]);
        }

        std::iota(refOrder.begin(), refOrder.end(), 0);
        std::sort(refOrder.begin(), refOrder.end(),
                  [&refStrides, &outer_block_dims] (size_t ind_l, size_t ind_r) {
                      return (refStrides[ind_l] > refStrides[ind_r]) ||
                             (refStrides[ind_l] == refStrides[ind_r] && outer_block_dims[ind_l] > outer_block_dims[ind_r]);
                  });
    }

    if (actualOrder != refOrder) {
        return false;
    }

    return true;
}

mkldnn::memory::format_tag DnnlBlockedMemoryDesc::getFormat() const {
    // TODO [OneDNN]: Previously it was a field of tdesc, but now the brute
    //                force search here. Please avoid of using this method.
    const auto ndims = desc.dims().size();

    // There are no suitable format_tag for this
    if (ndims == 0 || ndims > 6)
        return mkldnn::memory::format_tag::undef;

    for (const auto fmt : form_tags_by_ndims.at(ndims)) {
        if (this->isSame(fmt))
            return fmt;
    }

    return mkldnn::memory::format_tag::undef;
}

std::string DnnlBlockedMemoryDesc::serializeFormat() const {
    auto fmt = getFormat();
    return mkldnn::utils::fmt2str(fmt);
}

size_t DnnlBlockedMemoryDesc::getMaxMemSize() const {
    if (shape.isStatic()) {
        return getCurrentMemSize();
    }

    auto& maxDims = shape.getMaxDims();
    if (std::any_of(maxDims.begin(), maxDims.end(), [](size_t x){ return Shape::UNDEFINED_DIM == x; })) {
        return UNDEFINED_SIZE;
    }

    auto maxDimsDesc = cloneWithNewDims(maxDims);
    return maxDimsDesc->getCurrentMemSize();
}

size_t DnnlBlockedMemoryDesc::getPaddedElementsCount() const {
    return std::accumulate(std::begin(desc.data.padded_dims), std::begin(desc.data.padded_dims) + desc.data.ndims, size_t{1},
                           std::multiplies<int64_t>());
}

bool DnnlBlockedMemoryDesc::blocksExtended() const {
    for (int i = 0; i < desc.data.ndims; i++) {
        if (desc.data.dims[i] != desc.data.padded_dims[i])
            return true;
    }
    return false;
}

void DnnlBlockedMemoryDesc::initBlockDims() {
    const auto dims = desc.dims();

    const auto &blk_desc = desc.data.format_desc.blocking;

    const size_t outer_ndims = dims.size();
    const size_t inner_ndims = blk_desc.inner_nblks;
    const size_t total_ndims = outer_ndims + inner_ndims;

    // total inner block size. in case of 4i16o4i will be {16, 16, 1, 1}
    VectorDims total_block_per_dim(outer_ndims, 1);
    for (int i = 0; i < inner_ndims; i++) {
        total_block_per_dim[blk_desc.inner_idxs[i]] *= blk_desc.inner_blks[i];
    }
    // blocked dims
    // [dims via new_outer_order with auto pad] U [inner_blk_dims]
    VectorDims outer_block_dims = MKLDNNExtensionUtils::convertToVectorDims(dims);
    for (size_t i = 0; i < outer_block_dims.size(); i++) {
        if (outer_block_dims[i] != Shape::UNDEFINED_DIM) {
            outer_block_dims[i] = div_up(outer_block_dims[i], total_block_per_dim[i]);
        }
    }

    // order of outer dims. In case of IOhw_ will be {1, 0, 2, 3}
    VectorDims outer_order(outer_ndims);
    std::copy(order.begin(), order.begin() + outer_ndims, outer_order.begin());

    blockedDims.resize(total_ndims, 0);
    std::copy(blk_desc.inner_blks, blk_desc.inner_blks + blk_desc.inner_nblks,
              blockedDims.end() - blk_desc.inner_nblks);
    std::transform(outer_order.begin(), outer_order.end(), blockedDims.begin(),
                   [&] (size_t i) { return outer_block_dims[i]; });
}

void DnnlBlockedMemoryDesc::initStrides() {
    const auto dims = desc.dims();

    const auto &blk_desc = desc.data.format_desc.blocking;

    const size_t outer_ndims = dims.size();
    const size_t inner_ndims = blk_desc.inner_nblks;
    const size_t total_ndims = outer_ndims + inner_ndims;

    // strides of inner dims. In case of 4i16o4i will be {64, 4, 1}
    VectorDims inner_strides(inner_ndims, 1);
    for (size_t i = 1; i < blk_desc.inner_nblks; i++) {
        inner_strides[blk_desc.inner_nblks - 1 - i] = inner_strides[blk_desc.inner_nblks - i] * blk_desc.inner_blks[blk_desc.inner_nblks - i];
    }

    // order of outer dims. In case of IOhw_ will be {1, 0, 2, 3}
    VectorDims outer_order(outer_ndims);
    std::copy(order.begin(), order.begin() + outer_ndims, outer_order.begin());

    // blocked strides
    // [outer_strides via new_outer_order] U [inner_strides]
    strides.resize(total_ndims, 0);
    std::copy(inner_strides.rbegin(), inner_strides.rend(), strides.rbegin());
    std::transform(outer_order.begin(), outer_order.end(), strides.begin(),
                   [&](size_t i) { return blk_desc.strides[i] == DNNL_RUNTIME_DIM_VAL ? Shape::UNDEFINED_DIM : blk_desc.strides[i]; });
}

void DnnlBlockedMemoryDesc::initOffsetPadding() {
    offsetPaddingToData = VectorDims(std::begin(desc.data.padded_offsets), std::begin(desc.data.padded_offsets) + getOrder().size());
}

MemoryDescPtr DnnlBlockedMemoryDesc::cloneWithUndefStridesAndOffset() const {
    DnnlBlockedMemoryDescPtr newDesc = std::make_shared<DnnlBlockedMemoryDesc>(*this);
    auto &dnnlBlkDesc = newDesc->desc.data.format_desc.blocking;
    std::fill(std::begin(dnnlBlkDesc.strides), std::begin(dnnlBlkDesc.strides) + getShape().getRank(), DNNL_RUNTIME_DIM_VAL);
    newDesc->initStrides();
    newDesc->desc.data.offset0 = DNNL_RUNTIME_DIM_VAL;
    newDesc->status = descStatus::Undefined;
    return newDesc;
}

MemoryDescPtr DnnlBlockedMemoryDesc::cloneWithDefaultStridesAndOffset() const {
    DnnlBlockedMemoryDescPtr newDesc = std::make_shared<DnnlBlockedMemoryDesc>(*this);
    newDesc->recomputeStrides();
    newDesc->desc.data.offset0 = 0;
    newDesc->status = descStatus::Unknown;
    return newDesc;
}

MemoryDescPtr DnnlBlockedMemoryDesc::cloneWithNewPrecision(const InferenceEngine::Precision prec) const {
    auto newDesc = std::make_shared<DnnlBlockedMemoryDesc>(*this);
    newDesc->setPrecision(prec);
    return newDesc;
}

void DnnlBlockedMemoryDesc::recomputeStrides() {
    const auto &rank = getShape().getRank();

    if (order.size() != blockedDims.size())
        IE_THROW() << "Can't recompute stride: order size != blocked dims size";

    auto &oneDnnStrides = desc.data.format_desc.blocking.strides;
    if (std::any_of(blockedDims.begin(), blockedDims.end(), [](Dim val) { return val == Shape::UNDEFINED_DIM; })) {
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
