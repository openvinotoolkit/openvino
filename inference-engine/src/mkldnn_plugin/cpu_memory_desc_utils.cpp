// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_memory_desc.h"
#include "cpu_memory_desc_utils.h"
#include "utils/general_utils.h"
#include <limits>
#include <vector>
#include <numeric>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

namespace MKLDNNPlugin {

/**
 * Convert to  BlockedDescriptor
 *
 * mkl:  IOhw_4i16o4i    dims {32, 64, 128, 128}
 *   strides               // the order of outer dims is encoded here
 *   inner_blks   4 16 4
 *   inner_idxs   1  0 1
 *
 * IE tensor desc has more expressive ability. Any oneDNN blocked tensor can be covreted.
 * How to convert into IE representation:
 *    0. Detect a new_outer_order of outer_dims via descending strides.
 *    1. IE strides :  concatenate strides in new_outer_order and inner strides.
 *    2. IE dims    :  concatenate outer dims in new_outer_order with auto padding and inner blocks
 *    3. IE order   :  concatenate new_outer_order and inner_idxs
 */
BlockedMemoryDesc MemoryDescUtils::convertToBlockedDescriptor(const MKLDNNMemoryDesc& inpDesc) {
    mkldnn::memory::desc desc = inpDesc;
    const auto dims = desc.dims();

    if (desc.data.format_kind != dnnl_blocked)
        IE_THROW() << "Conversion is not possible";

    const auto &blk_desc = desc.data.format_desc.blocking;

    const size_t outer_ndims = dims.size();
    const size_t inner_ndims = blk_desc.inner_nblks;
    const size_t total_ndims = outer_ndims + inner_ndims;

    // strides of inner dims. In case of 4i16o4i will be {64, 4, 1}
    std::vector<size_t> inner_strides(inner_ndims, 1);
    for (size_t i = 1; i < blk_desc.inner_nblks; i++) {
        inner_strides[blk_desc.inner_nblks - 1 - i] = inner_strides[blk_desc.inner_nblks - i] * blk_desc.inner_blks[blk_desc.inner_nblks - i];
    }

    // total inner block size. in case of 4i16o4i will be {16, 16, 1, 1}
    std::vector<size_t> total_block_per_dim(outer_ndims, 1);
    for (int i = 0; i < inner_ndims; i++) {
        total_block_per_dim[blk_desc.inner_idxs[i]] *= blk_desc.inner_blks[i];
    }
    std::vector<size_t> outer_block_dims(std::begin(dims), std::begin(dims) + outer_ndims);
    for (size_t i = 0; i < outer_block_dims.size(); i++) {
        outer_block_dims[i] = div_up(outer_block_dims[i], total_block_per_dim[i]);
    }

    // order of outer dims. In case of IOhw_ will be {1, 0, 2, 3}
    std::vector<size_t> outer_order(outer_ndims);
    std::iota(outer_order.begin(), outer_order.end(), 0);
    std::sort(outer_order.begin(), outer_order.end(),
              [&blk_desc, &outer_block_dims] (size_t ind_l, size_t ind_r) {
                  return (blk_desc.strides[ind_l] > blk_desc.strides[ind_r]) ||
                         (blk_desc.strides[ind_l] == blk_desc.strides[ind_r] && outer_block_dims[ind_l] > outer_block_dims[ind_r]);
              });

    // IE blocked order
    // [new_outer_order] U [inner_idxs]
    SizeVector ie_blk_order(total_ndims, 0);
    std::copy(outer_order.begin(), outer_order.end(), ie_blk_order.begin());
    std::copy(blk_desc.inner_idxs, blk_desc.inner_idxs + blk_desc.inner_nblks, ie_blk_order.begin() + dims.size());

    // IE blocked strides
    // [outer_strides via new_outer_order] U [inner_strides]
    SizeVector ie_blk_strides(total_ndims, 0);
    std::copy(inner_strides.rbegin(), inner_strides.rend(), ie_blk_strides.rbegin());
    std::transform(outer_order.begin(), outer_order.end(), ie_blk_strides.begin(),
                   [&] (size_t i) { return blk_desc.strides[i]; });

    // IE blocked dims
    // [dims via new_outer_order with auto pad] U [inner_blk_dims]
    SizeVector ie_blk_dims(total_ndims, 0);
    std::copy(blk_desc.inner_blks, blk_desc.inner_blks + blk_desc.inner_nblks,
              ie_blk_dims.end() - blk_desc.inner_nblks);
    std::transform(outer_order.begin(), outer_order.end(), ie_blk_dims.begin(),
                   [&] (size_t i) { return outer_block_dims[i]; });

    // IE offset padded to data. Same as for oneDNN
    SizeVector ie_blk_offset_to_data {desc.data.padded_offsets, desc.data.padded_offsets + desc.data.ndims};
    size_t ie_blk_offset0 = desc.data.offset0;

    // TODO: The tensor desc implementation allow to specify offset_to_data for inner blocked dims.
    //       Which is not obvious behavior. It required offset_to_data.size == total_ndims, so will
    //       fill it with zero.
    ie_blk_offset_to_data.insert(ie_blk_offset_to_data.end(), inner_ndims, 0);

    BlockedMemoryDesc res(MKLDNNMemory::convertToIePrec(desc.data_type()), SizeVector {begin(dims), end(dims)}, ie_blk_dims,
                          ie_blk_order, ie_blk_offset0, ie_blk_offset_to_data, ie_blk_strides);
    return res;
}


InferenceEngine::TensorDesc MemoryDescUtils::convertToTensorDesc(const MemoryDesc& desc) {
    if (auto blockingDesc = dynamic_cast<const BlockedMemoryDesc*>(&desc)) {
        return InferenceEngine::TensorDesc(blockingDesc->getPrecision(), blockingDesc->getShape().getStaticDims(),
                                           {blockingDesc->getBlockDims(), blockingDesc->getOrder(), blockingDesc->getOffsetPadding(),
                                            blockingDesc->getOffsetPaddingToData(), blockingDesc->getStrides()});
    } else if (auto mkldnnDesc = dynamic_cast<const MKLDNNMemoryDesc*>(&desc)) {
        auto blockingDesc = convertToBlockedDescriptor(*mkldnnDesc);
        return InferenceEngine::TensorDesc(blockingDesc.getPrecision(), blockingDesc.getShape().getStaticDims(),
                                           {blockingDesc.getBlockDims(), blockingDesc.getOrder(), blockingDesc.getOffsetPadding(),
                                            blockingDesc.getOffsetPaddingToData(), blockingDesc.getStrides()});
    }

    IE_THROW() << "Cannot convert MemoryDesc to InferenceEngine::TensorDesc";

    return InferenceEngine::TensorDesc();
}

MKLDNNMemoryDesc MemoryDescUtils::convertToMKLDNNMemoryDesc(const MemoryDesc& desc) {
    if (auto blockingDesc = dynamic_cast<const BlockedMemoryDesc*>(&desc)) {
        return convertToMKLDNNMemoryDesc(*blockingDesc);
    } else if (auto mkldnnDesc = dynamic_cast<const MKLDNNMemoryDesc*>(&desc)) {
        return *mkldnnDesc;
    } else {
        IE_THROW() << "Cannot convert MemoryDesc to MKLDNNMemoryDesc";
    }
}

MKLDNNMemoryDesc MemoryDescUtils::convertToMKLDNNMemoryDesc(const BlockedMemoryDesc& desc) {
    dnnl_memory_desc_t mkldnnDesc;
    auto dims = desc.getShape().getStaticDims();

    auto ie_blkdDims = desc.getBlockDims();
    auto ie_order = desc.getOrder();
    auto ie_offsetsToData = desc.getOffsetPaddingToData();
    auto ie_strides = desc.getStrides();

    size_t outer_ndims = dims.size();
    size_t inner_ndims = ie_order.size() - dims.size();

    bool is_descending_strides = true;
    for (int i = 1; i < ie_strides.size(); i++) {
        is_descending_strides &= (ie_strides[i-1] >= ie_strides[i]);
    }

    // TODO: That's strong constrains and can be mitigated. IE::TensorDesc allow to transpose blocked dims
    //       and may be we can achieve correct "descending strides" form which allow conversion.
    if (!is_descending_strides)
        IE_THROW() << "Unsupported case for conversion";

    std::vector<size_t> outer_order(outer_ndims, outer_ndims + 1); // outer_order[i] is index of stride for i-th dimension
    for (size_t i = 0; i < outer_ndims; i++) {
        outer_order[ie_order[i]] = i;
    }
    bool outer_is_correct_permutation_of_n =
            std::find(outer_order.begin(), outer_order.end(), outer_ndims + 1) == outer_order.end();

    if (!outer_is_correct_permutation_of_n)
        IE_THROW() << "Unsupported case for conversion";

    bool inner_block_are_dense = one_of(ie_strides.back(), 0, 1);  // stride 1 - is dense case, 0 - broad casted
    for (int i = outer_ndims; i < ie_strides.size() - 1; i++) {
        inner_block_are_dense &= (ie_strides[i] == ie_strides[i+1] * ie_blkdDims[i+1]);
    }

    if (!inner_block_are_dense)
        IE_THROW() << "Unsupported case for conversion";

    bool inner_pad_offsets_is_zero = std::all_of(ie_offsetsToData.begin() + outer_ndims, ie_offsetsToData.end(),
                                                 [](size_t pad) { return  pad == 0; });

    if (!inner_pad_offsets_is_zero)
        IE_THROW() << "Unsupported case for conversion";

    // Fill general memory desc fields
    mkldnnDesc.format_kind = dnnl_blocked;
    mkldnnDesc.extra.flags = 0;
    mkldnnDesc.data_type = memory::convert_to_c(MKLDNNMemory::convertToDataType(desc.getPrecision()));
    mkldnnDesc.ndims = dims.size();
    mkldnnDesc.offset0 = desc.getOffsetPadding();
    std::copy(dims.begin(), dims.end(), mkldnnDesc.dims);
    std::copy(ie_offsetsToData.begin(), ie_offsetsToData.begin() + outer_ndims, mkldnnDesc.padded_offsets);
    std::fill(mkldnnDesc.padded_dims, mkldnnDesc.padded_dims + outer_ndims, 1);
    for (size_t i = 0; i < ie_order.size(); i++) {
        auto idx = ie_order[i];
        mkldnnDesc.padded_dims[idx] *= ie_blkdDims[i];
    }

    // Fill blocking desc
    auto &dnn_blk_desc = mkldnnDesc.format_desc.blocking;
    dnn_blk_desc.inner_nblks = inner_ndims;
    std::copy(ie_blkdDims.end() - inner_ndims, ie_blkdDims.end(), dnn_blk_desc.inner_blks);
    std::copy(ie_order.end() - inner_ndims, ie_order.end(), dnn_blk_desc.inner_idxs);
    for (size_t i = 0; i < outer_ndims; i++) {
        dnn_blk_desc.strides[i] = ie_strides[outer_order[i]];
    }

    return MKLDNNMemoryDesc(mkldnnDesc);
}


/**
 * Construct from IE::TensorDesc
 * @param tDesc
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
 *      but not [0]<=>[4] because it breacke spliting original dims into internal blocked dims
 *   Normalization of representation: Make strides growing but keep layout same as original. Not all
 *   layout allow us to meet normalize form of tensor desc.
 *
 *   Limitation of conversion first N elements of order should be permutation of [0,1,2 ... N]
 */
MKLDNNMemoryDesc MemoryDescUtils::convertToMKLDNNMemoryDesc(const InferenceEngine::TensorDesc& tDesc) {
    mkldnn::memory::desc mkldnnDesc({}, mkldnn::memory::data_type::undef, mkldnn::memory::format_tag::undef);
    auto dims = tDesc.getDims();

    // TODO: implicit conversion of dims is no good...
    if (tDesc.getLayout() == Layout::SCALAR) {
        mkldnnDesc.data.format_kind = dnnl_blocked;
        mkldnnDesc.data.data_type = memory::convert_to_c(MKLDNNMemory::convertToDataType(tDesc.getPrecision()));
        mkldnnDesc.data.ndims = 1;
        mkldnnDesc.data.dims[0] = 1;
        mkldnnDesc.data.padded_dims[0] = 1;
        mkldnnDesc.data.format_desc.blocking.strides[0] = 1;
        mkldnnDesc.data.padded_offsets[0] = 0;
        mkldnnDesc.data.offset0 = tDesc.getBlockingDesc().getOffsetPadding();
        return MKLDNNMemoryDesc(mkldnnDesc);
    }

    if (tDesc.getLayout() == Layout::ANY) {
        mkldnnDesc.data.format_kind = dnnl_format_kind_any;
        mkldnnDesc.data.data_type = memory::convert_to_c(MKLDNNMemory::convertToDataType(tDesc.getPrecision()));
        mkldnnDesc.data.ndims = dims.size();
        std::copy(dims.begin(), dims.end(), mkldnnDesc.data.dims);
        std::copy(dims.begin(), dims.end(), mkldnnDesc.data.padded_dims);
        mkldnnDesc.data.offset0 = tDesc.getBlockingDesc().getOffsetPadding();
        std::fill(mkldnnDesc.data.padded_offsets, mkldnnDesc.data.padded_offsets + dims.size(), 0);
        return MKLDNNMemoryDesc(mkldnnDesc);
    }

    auto ie_blkdDims = tDesc.getBlockingDesc().getBlockDims();
    auto ie_order = tDesc.getBlockingDesc().getOrder();
    auto ie_offsetsToData = tDesc.getBlockingDesc().getOffsetPaddingToData();
    auto ie_strides = tDesc.getBlockingDesc().getStrides();

    size_t outer_ndims = dims.size();
    size_t inner_ndims = ie_order.size() - dims.size();

    bool is_descending_strides = true;
    for (int i = 1; i < ie_strides.size(); i++) {
        is_descending_strides &= (ie_strides[i-1] >= ie_strides[i]);
    }

    // TODO: That's strong constrains and can be mitigated. IE::TensorDesc allow to transpose blocked dims
    //       and may be we can achieve correct "descending strides" form which allow conversion.
    if (!is_descending_strides)
        IE_THROW() << "Unsupported case for conversion";

    std::vector<size_t> outer_order(outer_ndims, outer_ndims + 1); // outer_order[i] is index of stride for i-th dimension
    for (size_t i = 0; i < outer_ndims; i++) {
        outer_order[ie_order[i]] = i;
    }
    bool outer_is_correct_permutation_of_n =
            std::find(outer_order.begin(), outer_order.end(), outer_ndims + 1) == outer_order.end();

    if (!outer_is_correct_permutation_of_n)
        IE_THROW() << "Unsupported case for conversion";

    bool inner_block_are_dense = one_of(ie_strides.back(), 0, 1);  // stride 1 - is dense case, 0 - broad casted
    for (int i = outer_ndims; i < ie_strides.size() - 1; i++) {
        inner_block_are_dense &= (ie_strides[i] == ie_strides[i+1] * ie_blkdDims[i+1]);
    }

    if (!inner_block_are_dense)
        IE_THROW() << "Unsupported case for conversion";

    bool inner_pad_offsets_is_zero = std::all_of(ie_offsetsToData.begin() + outer_ndims, ie_offsetsToData.end(),
                                                 [](size_t pad) { return  pad == 0; });

    if (!inner_pad_offsets_is_zero)
        IE_THROW() << "Unsupported case for conversion";

    // Fill general memory desc fields
    mkldnnDesc.data.format_kind = dnnl_blocked;
    mkldnnDesc.data.data_type = memory::convert_to_c(MKLDNNMemory::convertToDataType(tDesc.getPrecision()));
    mkldnnDesc.data.ndims = dims.size();
    mkldnnDesc.data.offset0 = tDesc.getBlockingDesc().getOffsetPadding();
    std::copy(dims.begin(), dims.end(), mkldnnDesc.data.dims);
    std::copy(ie_offsetsToData.begin(), ie_offsetsToData.begin() + outer_ndims, mkldnnDesc.data.padded_offsets);
    std::fill(mkldnnDesc.data.padded_dims, mkldnnDesc.data.padded_dims + outer_ndims, 1);
    for (size_t i = 0; i < ie_order.size(); i++) {
        auto idx = ie_order[i];
        mkldnnDesc.data.padded_dims[idx] *= ie_blkdDims[i];
    }

    // Fill blocking desc
    auto &dnn_blk_desc = mkldnnDesc.data.format_desc.blocking;
    dnn_blk_desc.inner_nblks = inner_ndims;
    std::copy(ie_blkdDims.end() - inner_ndims, ie_blkdDims.end(), dnn_blk_desc.inner_blks);
    std::copy(ie_order.end() - inner_ndims, ie_order.end(), dnn_blk_desc.inner_idxs);
    for (size_t i = 0; i < outer_ndims; i++) {
        dnn_blk_desc.strides[i] = ie_strides[outer_order[i]];
    }

    return MKLDNNMemoryDesc(mkldnnDesc);
}

//MemoryDescPtr MemoryDescUtils::getUndefinedMemoryDesc(const MKLDNNMemoryDesc& desc) {
//    if (desc.getFormatKind() != dnnl_format_kind_t::dnnl_blocked)
//        IE_THROW() << "Cannot get undefined memory descriptor for non blocked format kind";
//
//    BlockedMemoryDesc bd = convertToBlockedDescriptor(desc);
//
//    std::vector<size_t> notInitArr;
//    std::vector<size_t> zeroArr;
//    for (size_t i = 0; i < bd.getBlockDims().size(); i++) {
//        notInitArr.push_back(std::numeric_limits<size_t>::max());
//        zeroArr.push_back(0);
//    }
//    // MKLDNN doesn't support offset_padding_to_data[i] != 0 (assert(src_d_blk.offset_padding_to_data[d] == 0);)
//    auto retDesc = make_unique<BlockedMemoryDesc>(bd.getPrecision(), bd.getShape().getDims(), bd.getBlockDims(),
//                                                  bd.getOrder(), std::numeric_limits<size_t>::max(), zeroArr, notInitArr);
//    return retDesc;
//}

} // namespace MKLDNNPlugin
