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

mkldnn::memory::format_tag MemoryDescUtils::getLayout(const MemoryDesc& desc) {
    // TODO [OneDNN]: Previously it was a field of tdesc, but now the brute
    //                force search here. Please avoid of using this method.
    if (auto dnnlDesc = dynamic_cast<const MKLDNNMemoryDesc*>(&desc)) {
        const auto ndims = dnnlDesc->getDims().size();

        // There are no suitable format_tag for this
        if (ndims == 0 || ndims > 6)
            return mkldnn::memory::format_tag::undef;

        for (const auto fmt : form_tags_by_ndims.at(ndims)) {
            if (dnnlDesc->isSame(fmt))
                return fmt;
        }
    }

    IE_THROW() << "Cannot get layout for given MemoryDesc";

    return mkldnn::memory::format_tag::undef;
}

InferenceEngine::TensorDesc MemoryDescUtils::convertToTensorDesc(const MemoryDesc& desc) {
    if (auto blockingDesc = dynamic_cast<const BlockedMemoryDesc*>(&desc)) {
        return InferenceEngine::TensorDesc(blockingDesc->getPrecision(), blockingDesc->getShape().getStaticDims(),
                                           {blockingDesc->getBlockDims(), blockingDesc->getOrder(), blockingDesc->getOffsetPadding(),
                                            blockingDesc->getOffsetPaddingToData(), blockingDesc->getStrides()});
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
    dnnl_memory_desc_t mkldnnDesc;
    auto dims = tDesc.getDims();

    // TODO: implicit conversion of dims is no good...
    if (tDesc.getLayout() == Layout::SCALAR) {
        mkldnnDesc.format_kind = dnnl_blocked;
        mkldnnDesc.data_type = memory::convert_to_c(MKLDNNMemory::convertToDataType(tDesc.getPrecision()));
        mkldnnDesc.ndims = 1;
        mkldnnDesc.dims[0] = 1;
        mkldnnDesc.padded_dims[0] = 1;
        mkldnnDesc.format_desc.blocking.strides[0] = 1;
        mkldnnDesc.padded_offsets[0] = 0;
        mkldnnDesc.offset0 = tDesc.getBlockingDesc().getOffsetPadding();
        return MKLDNNMemoryDesc(mkldnnDesc);
    }

    if (tDesc.getLayout() == Layout::ANY) {
        mkldnnDesc.format_kind = dnnl_format_kind_any;
        mkldnnDesc.data_type = memory::convert_to_c(MKLDNNMemory::convertToDataType(tDesc.getPrecision()));
        mkldnnDesc.ndims = dims.size();
        std::copy(dims.begin(), dims.end(), mkldnnDesc.dims);
        std::copy(dims.begin(), dims.end(), mkldnnDesc.padded_dims);
        mkldnnDesc.offset0 = tDesc.getBlockingDesc().getOffsetPadding();
        std::fill(mkldnnDesc.padded_offsets, mkldnnDesc.padded_offsets + dims.size(), 0);
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
    mkldnnDesc.format_kind = dnnl_blocked;
    mkldnnDesc.data_type = memory::convert_to_c(MKLDNNMemory::convertToDataType(tDesc.getPrecision()));
    mkldnnDesc.ndims = dims.size();
    mkldnnDesc.offset0 = tDesc.getBlockingDesc().getOffsetPadding();
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

MKLDNNMemoryDesc MemoryDescUtils::getUndefinedMemoryDesc(const MKLDNNMemoryDesc& desc) {
    if (desc.getFormatKind() != dnnl_format_kind_t::dnnl_blocked)
        IE_THROW() << "Cannot get undefined memory descriptor for non blocked format kind";

    InferenceEngine::TensorDesc td = desc;

    std::vector<size_t> notInitArr;
    std::vector<size_t> zeroArr;
    for (size_t i = 0; i < td.getBlockingDesc().getBlockDims().size(); i++) {
        notInitArr.push_back(std::numeric_limits<size_t>::max());
        zeroArr.push_back(0);
    }
    // MKLDNN doesn't support offset_padding_to_data[i] != 0 (assert(src_d_blk.offset_padding_to_data[d] == 0);)
    return td.getLayout() == InferenceEngine::Layout::ANY
           ? convertToMKLDNNMemoryDesc(td)
           : convertToMKLDNNMemoryDesc(InferenceEngine::TensorDesc(td.getPrecision(), td.getDims(),
                                       {td.getBlockingDesc().getBlockDims(), td.getBlockingDesc().getOrder(),
                                        std::numeric_limits<size_t>::max(), zeroArr, notInitArr}));
}

} // namespace MKLDNNPlugin
