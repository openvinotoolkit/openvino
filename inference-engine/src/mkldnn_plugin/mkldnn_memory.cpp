// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <utility>

#include "utils/general_utils.h"

#include <mkldnn_types.h>
#include "mkldnn_memory.h"
#include "mkldnn_extension_utils.h"
#include "nodes/common/cpu_memcpy.h"
#include "ie_mkldnn.h"

using namespace InferenceEngine;
using namespace mkldnn;

namespace MKLDNNPlugin {

MKLDNNMemory::MKLDNNMemory(const mkldnn::engine& eng) : eng(eng) {}

size_t MKLDNNMemory::GetSize() const {
    uint8_t itemSize = MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(GetDataType()));
    return GetElementsCount() * itemSize;
}

size_t MKLDNNMemory::GetElementsCount() const {
    auto desc = GetDescriptor();
    std::vector<int> dims(desc.data.padded_dims,
                          desc.data.padded_dims + desc.data.ndims);
    return std::accumulate(std::begin(dims), std::end(dims), (size_t) 1, std::multiplies<size_t>());
}

void MKLDNNMemory::Create(memory::dims dims, memory::data_type data_type, memory::format_tag format, const void* data) {
    if (format == memory::format_tag::undef) {
        format = memory::format_tag::any;
    }

    memory::desc desc = MKLDNNMemoryDesc({dims}, data_type, format);

    if (format == memory::format_tag::any) {
        CreateBlockingDesc(desc);
    }

    Create(desc, data);
}

void MKLDNNMemory::Create(const mkldnn::memory::desc& desc, const void *data, bool pads_zeroing) {
    if (data == nullptr) {
        prim.reset(new memory(desc, eng));

        size_t real_size = 0;
        if (desc.data.format_kind == dnnl_format_kind_wino)
            return;
        auto desc_loc = prim->get_desc().data;
        if (desc_loc.ndims > 0) {
            real_size = static_cast<size_t>(desc_loc.padded_dims[0]);
            for (int i = 1; i < desc_loc.ndims; i++) {
                real_size *= desc_loc.padded_dims[i];
            }
        }
    } else {
        // MKLDNN accepts not a const data, probably need to remove some level of consteness in a call stack

        // ========================
        // Equivalent of constructor memory(const primitive_desc &desc, void *hdl)
        // but with ability to skipp pads zeroing.
        prim.reset(new memory(desc, eng, DNNL_MEMORY_NONE));
        if (pads_zeroing)
            prim->set_data_handle(const_cast<void*>(data));
        else
            prim->set_data_handle_no_pads_proc(const_cast<void*>(data));
        //
        // ========================
    }
}

// TODO: It should be done via wrap into Memory;
void MKLDNNMemory::SetData(memory::data_type dataType, memory::format_tag format, const void* data, size_t size, bool ftz) const {
    uint8_t itemSize = MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(dataType));

    IE_ASSERT(!one_of(format, memory::format_tag::undef, memory::format_tag::any));

    auto dst_desc = GetDescriptor();
    memory::desc src_desc{dst_desc.dims(), dataType, format};

    IE_ASSERT(size == dst_desc.get_size());

    if (dst_desc != src_desc) {
        auto memData = GetDescriptor().data;
        memory::dims dims{memData.dims, memData.dims + memData.ndims};

        MKLDNNMemory src(eng);
        src.Create(dims, dataType, format, data);

        std::shared_ptr<mkldnn::reorder> pReorder =
                std::shared_ptr<mkldnn::reorder>(new mkldnn::reorder(src.GetPrimitive(), GetPrimitive()));

        mkldnn::stream loc_stream(eng, stream::flags::default_flags);
        pReorder->execute(loc_stream, *src.prim, *this->prim);
    } else {
        uint8_t* dataPtr = static_cast<uint8_t*>(GetData());
        // We cannot support strides for i/o blobs because it affects performance.
        dataPtr += itemSize * prim->get_desc().data.offset0;
        cpu_memcpy(dataPtr, data, size);
    }

    if (ftz && dataType == memory::data_type::f32) {
        // Internal blobs haven't strides yet.
        auto *memData = static_cast<float *>(GetData());
        memData += prim->get_desc().data.offset0;
        size_t realSize = GetSize() / sizeof(float);
        for (size_t i = 0; i < realSize; i++) {
            if (memData[i] != 0 && (fabsf(memData[i]) < std::numeric_limits<float>::min())) {
                memData[i] = 0.0f;
            }
        }
    }
}

void MKLDNNMemory::SetData(const MKLDNNMemory& src, bool ftz) const {
    // TODO: Optimization. Reorder perfect is not good enough, so in triviale cases we
    //       prefer use simple copy.
    if (src.GetDesc() == this->GetDesc()) {
        auto srcPtr = static_cast<uint8_t*>(src.GetPtr());
        auto dstPtr = static_cast<uint8_t*>(this->GetPtr());
        auto size = this->GetSize();
        cpu_memcpy(dstPtr, srcPtr, size);
    } else {
        mkldnn::reorder reorderPrim(src.GetPrimitive(), GetPrimitive());
        mkldnn::stream loc_stream(eng, stream::flags::default_order);
        reorderPrim.execute(loc_stream, *src.prim, *this->prim);
    }

    if (ftz && src.GetDataType() == mkldnn::memory::data_type::f32 &&
        prim->get_desc().data.format_kind != dnnl_format_kind_wino &&
        GetDataType() != mkldnn::memory::data_type::bf16) {
        // Internal blobs haven't strides yet.
        auto *memData = static_cast<float *>(GetData());
        memData += prim->get_desc().data.offset0;
        size_t realSize = GetSize() / sizeof(float);
        for (size_t i = 0; i < realSize; i++) {
            if (memData[i] != 0 && (fabsf(memData[i]) < std::numeric_limits<float>::min())) {
                memData[i] = 0.0f;
            }
        }
    }
}

void MKLDNNMemory::FillZero() {
    void* dataPtr = GetData();
    memset(dataPtr, 0, GetSize());
}

memory::format_tag MKLDNNMemory::GetPlainFormat(memory::dims dims) {
    switch (dims.size()) {
        case 0:
        case 1:
            return memory::format_tag::a;
        case 2:
            return memory::format_tag::ab;
        case 3:
            return memory::format_tag::abc;
        case 4:
            return memory::format_tag::abcd;
        case 5:
            return memory::format_tag::abcde;
        case 6:
            return memory::format_tag::abcdef;
        default:
            return memory::format_tag::undef;
    }
}

InferenceEngine::Layout MKLDNNMemory::GetPlainLayout(memory::dims dims) {
    switch (dims.size()) {
        case 0: return Layout::SCALAR;
        case 1: return Layout::C;
        case 2: return Layout::NC;
        case 3: return Layout::CHW;
        case 4: return Layout::NCHW;
        case 5: return Layout::NCDHW;
        default:
            return Layout::BLOCKED;
    }
}

bool MKLDNNMemory::isConsistant(mkldnn::memory::dims dims, mkldnn::memory::format_tag format) {
    memory::desc attempt(dims, memory::data_type::f32, format, true);
    return static_cast<bool>(attempt);
}

void MKLDNNMemory::CreateBlockingDesc(memory::desc &desc) {
    auto dims = desc.data.dims;
    int ndims = desc.data.ndims;

    desc.data.format_kind = static_cast<dnnl_format_kind_t>(memory::format_kind::blocked);

    auto& blk = desc.data.format_desc.blocking;

    desc.data.offset0 = 0;

    // TODO: finish implementation or delete
//    for (int i = 0; i < ndims; i++) {
//        blk.block_dims[i] = 1;
//        blk.strides[1][i] = 1;
//        blk.padding_dims[i] = dims[i];
//        blk.offset_padding_to_data[i] = 0;
//    }
//
//    int perm[TENSOR_MAX_DIMS] = {0};
//
//    for (int i = 0; i < ndims; ++i) {
//        perm[i] = i;
//    }
//
//    blk.strides[0][perm[ndims - 1]] = 1;
//
//    for (int d = 1; d < ndims; ++d) {
//        const int prev_idx = perm[ndims - d];
//        const int curr_idx = perm[ndims - 1 - d];
//
//        blk.strides[0][curr_idx] = dims[curr_idx] == 0 ? 1 : blk.strides[0][prev_idx] * (std::max)((ptrdiff_t)1, dims[prev_idx]);
//    }
}

Precision MKLDNNMemory::convertToIePrec(memory::data_type dataType) {
    switch (dataType) {
        case memory::data_type::f32:
            return Precision::FP32;
        case memory::data_type::u8:
            return Precision::U8;
        case memory::data_type::s8:
            return Precision::I8;
        case memory::data_type::s32:
            return Precision::I32;
// TODO: bin is not ported yet
//
//        case memory::data_type::bin:
//            return Precision::BIN;
        case memory::data_type::bf16:
            return Precision::BF16;
        default:
           THROW_IE_EXCEPTION << "Unknown mkldnn data type";
    }
}

memory::data_type MKLDNNMemory::convertToDataType(const InferenceEngine::Precision &precision) {
    switch (precision) {
        case Precision::FP32:
            return memory::data_type::f32;
        case Precision::FP16:
            return memory::data_type::f16;
        case Precision::BF16:
            return memory::data_type::bf16;
        case Precision::I32:
            return memory::data_type::s32;
        case Precision::U8:
        case Precision::BOOL:
            return memory::data_type::u8;
        case Precision::I8:
            return memory::data_type::s8;
        default:
            THROW_IE_EXCEPTION << "Unknown mkldnn data type";
    }
}


memory::format_tag MKLDNNMemory::Convert(const InferenceEngine::Layout layout) {
    switch (layout) {
        case NCHW:
            return memory::format_tag::nchw;
        case NHWC:
            return memory::format_tag::nhwc;
        case NCDHW:
            return memory::format_tag::ncdhw;
        case NDHWC:
            return memory::format_tag::ndhwc;
        case CHW:
            return memory::format_tag::tnc;
        case NC:
            return memory::format_tag::nc;
        case C:
            return memory::format_tag::x;
        case SCALAR:
            return memory::format_tag::x;
        default:
            return memory::format_tag::undef;
    }
}

std::string MKLDNNMemory::formatToString(memory::format_tag fmt) {
    return mkldnn::utils::fmt2str(fmt);
}

bool MKLDNNMemoryDesc::operator==(const MKLDNNMemoryDesc &rhs) const {
    return this->desc == rhs.desc;
}

bool MKLDNNMemoryDesc::operator!=(const MKLDNNMemoryDesc &rhs) const {
    return !(*this == rhs);
}

MKLDNNMemoryDesc::operator mkldnn::memory::desc() const {
    return desc;
}

MKLDNNMemoryDesc::MKLDNNMemoryDesc(mkldnn::memory::dims dims, mkldnn::memory::data_type dataType,
                                   mkldnn::memory::format_tag format): desc(dims, dataType, mkldnn::memory::format_tag::any) {
    if (format != memory::format_tag::undef) {
        if (format == memory::format_tag::x && dims.size() == 0) {
            desc = mkldnn::memory::desc(mkldnn::memory::dims(1, 1), dataType, format);
            MKLDNNMemory::CreateBlockingDesc(desc);
        } else {
            desc = mkldnn::memory::desc(dims, dataType, format);
        }
        return;
    }
    MKLDNNMemory::CreateBlockingDesc(desc);
}

MKLDNNMemoryDesc::MKLDNNMemoryDesc(mkldnn::memory::dims dims, mkldnn::memory::data_type dataType) : desc() {
    const auto ndims = dims.size();
    mkldnn::memory::dims plain_strides(ndims, 1);
    for (size_t i = 1; i < ndims; i++) {
        plain_strides[ndims - i -1] = plain_strides[ndims - i] * dims[ndims - i];
    }
    desc = {dims, dataType, plain_strides};
}

size_t MKLDNNMemoryDesc::GetElementSize() const {
    const auto type = desc.data_type();
    switch (type) {
        case memory::data_type::f16 :
        case memory::data_type::bf16 :
            return 2;
        case memory::data_type::f32 :
        case memory::data_type::s32 :
            return 4;
        case memory::data_type::s8 :
        case memory::data_type::u8 :
            return 1;
        default:
            THROW_IE_EXCEPTION << "Unknown data type";
    }
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
        mkldnn::memory::format_tag::acbdef,      // permuted
        mkldnn::memory::format_tag::defcab,      // permuted
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

        mkldnn::memory::format_tag::Abcdef16a,
        mkldnn::memory::format_tag::Abcdef32a,
        mkldnn::memory::format_tag::aBCdef2b4c2b,
        mkldnn::memory::format_tag::aBCdef2c4b2c,
        mkldnn::memory::format_tag::aBCdef4b8c2b,
        mkldnn::memory::format_tag::aBCdef4c8b2c,
        }}
};

mkldnn::memory::format_tag MKLDNNMemoryDesc::getFormat() const {
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

bool MKLDNNMemoryDesc::isSame(mkldnn::memory::format_tag fmt) const {
    memory::desc tmp_desc(desc.dims(), desc.data_type(), fmt);

    // Format doesn't depend on offset, so exclude it from check
    tmp_desc.data.offset0 = desc.data.offset0;

    return (tmp_desc == desc);
}

bool MKLDNNMemoryDesc::isPlainFormat() const {
    if (desc.data.format_kind != dnnl_blocked ||
        desc.data.format_desc.blocking.inner_nblks != 0)
        return false;

    const auto ndims = desc.data.ndims;
    const auto dims = desc.data.dims;
    const auto &strides = desc.data.format_desc.blocking.strides;
    bool is_plain_strides = (strides[ndims-1] == 1);
    for (int i = 0; i < ndims - 1; i++) {
        is_plain_strides &= (strides[i] == strides[i+1] * dims[i+1]);
    }

    return is_plain_strides;
}

bool MKLDNNMemoryDesc::isBlockedCFormat(size_t blk_size) const {
    const auto &blocking = desc.data.format_desc.blocking;

    if (desc.data.format_kind != dnnl_blocked ||
        blocking.inner_nblks != 1 ||
        blocking.inner_idxs[0] != 1)
        return false;

    const auto &ndims = desc.data.ndims;
    const auto &strides = desc.data.format_desc.blocking.strides;
    const auto &dims = desc.data.padded_dims;

    if (blk_size == UNREACHABLE_DIM) {
        blk_size = blocking.inner_blks[0];
    } else {
        if (blk_size != blocking.inner_blks[0])
            return false;
    }

    bool is_direct_order = (strides[ndims-1] == blocking.inner_blks[0]);
    for (int i = 0; i < ndims - 1; i++) {
        auto dim = (i == 0) ? div_up(dims[i+1], blk_size) : dims[i+1];
        is_direct_order &= (strides[i] >= strides[i+1] * dim);
    }

    return is_direct_order;
}

bool MKLDNNMemoryDesc::isTailCFormat() const {
    const auto &blocking = desc.data.format_desc.blocking;

    if (desc.data.format_kind != dnnl_blocked ||
        blocking.inner_nblks != 0)
        return false;

    const auto &ndims = desc.data.ndims;
    const auto &strides = desc.data.format_desc.blocking.strides;
    const auto &dims = desc.data.padded_dims;
    const auto blk_size = blocking.inner_blks[0];

    // dense permutation of acd..b
    bool is_tailc_strides = (strides[1] == 1 && strides[ndims-1] == dims[1] && strides[0] == dims[2] * strides[2]);
    for (int i = 2; i < ndims - 1; i++) {
        is_tailc_strides &= (strides[i] == strides[i+1] * dims[i+1]);
    }

    return is_tailc_strides;
}

/**
 * Convert to  IE::TensorDesc
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
MKLDNNMemoryDesc::operator InferenceEngine::TensorDesc() const {
    const auto dims = desc.dims();

    if (desc.data.format_kind == dnnl_format_kind_any)
        return TensorDesc {
                MKLDNNMemory::convertToIePrec(desc.data_type()),
                SizeVector {begin(dims), end(dims)},
                Layout::ANY};

    if (desc.data.format_kind != dnnl_blocked)
        THROW_IE_EXCEPTION << "Conversion is not possible";

    const auto &blk_desc = desc.data.format_desc.blocking;

    const size_t outer_ndims = dims.size();
    const size_t inner_ndims = blk_desc.inner_nblks;
    const size_t total_ndims = outer_ndims + inner_ndims;

    // order of outer dims. In case of IOhw_ will be {1, 0, 2, 3}
    std::vector<size_t> outer_order(outer_ndims);
    std::iota(outer_order.begin(), outer_order.end(), 0);
    std::sort(outer_order.begin(), outer_order.end(),
              [&blk_desc] (size_t ind_l, size_t ind_r) {
        return blk_desc.strides[ind_l] > blk_desc.strides[ind_r];
    });

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
                   [&] (size_t i) { return div_up(dims[i], total_block_per_dim[i]); });

    // IE offset padded to data. Same as for oneDNN
    SizeVector ie_blk_offset_to_data {desc.data.padded_offsets, desc.data.padded_offsets + desc.data.ndims};
    size_t ie_blk_offset0 = desc.data.offset0;

    // TODO: The tensor desc implementation allow to specify offset_to_data for inner blocked dims.
    //       Which is not obvious behavior. It required offset_to_data.size == total_ndims, so will
    //       fill it with zero.
    ie_blk_offset_to_data.insert(ie_blk_offset_to_data.end(), inner_ndims, 0);


    BlockingDesc ie_blk_desc { ie_blk_dims,
                               ie_blk_order,
                               ie_blk_offset0,
                               ie_blk_offset_to_data,
                               ie_blk_strides };
    TensorDesc res {
        MKLDNNMemory::convertToIePrec(desc.data_type()),
        SizeVector {begin(dims), end(dims)},
        ie_blk_desc };
    // TODO: BLOCKED is the most common layout which covers all other permuted layout like NHWC.
    //       But for some cases we have to specify it more correctly.. may be.. or just keep
    //       auto detected layout in constructor of TensorDesc.
    return res;
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
MKLDNNMemoryDesc::MKLDNNMemoryDesc(const TensorDesc& tDesc):
        desc({}, mkldnn::memory::data_type::undef, mkldnn::memory::format_tag::undef) {
    auto dims = tDesc.getDims();

    // TODO: implicit conversion of dims is no good...
    if (tDesc.getLayout() == Layout::SCALAR) {
        desc.data.format_kind = dnnl_blocked;
        desc.data.data_type = memory::convert_to_c(MKLDNNMemory::convertToDataType(tDesc.getPrecision()));
        desc.data.ndims = 1;
        desc.data.dims[0] = 1;
        desc.data.padded_dims[0] = 1;
        desc.data.padded_offsets[0] = 0;
        desc.data.offset0 = tDesc.getBlockingDesc().getOffsetPadding();
        return;
    }

    if (tDesc.getLayout() == Layout::ANY) {
        desc.data.format_kind = dnnl_format_kind_any;
        desc.data.data_type = memory::convert_to_c(MKLDNNMemory::convertToDataType(tDesc.getPrecision()));
        desc.data.ndims = dims.size();
        std::copy(dims.begin(), dims.end(), desc.data.dims);
        std::copy(dims.begin(), dims.end(), desc.data.padded_dims);
        desc.data.offset0 = tDesc.getBlockingDesc().getOffsetPadding();
        std::fill(desc.data.padded_offsets, desc.data.padded_offsets + dims.size(), 0);
        return;
    }

    SizeVector ie_blkdDims = tDesc.getBlockingDesc().getBlockDims();
    SizeVector ie_order = tDesc.getBlockingDesc().getOrder();
    SizeVector ie_offsetsToData = tDesc.getBlockingDesc().getOffsetPaddingToData();
    SizeVector ie_strides = tDesc.getBlockingDesc().getStrides();

    size_t outer_ndims = dims.size();
    size_t inner_ndims = ie_order.size() - dims.size();

    bool is_descending_strides = true;
    for (int i = 1; i < ie_strides.size(); i++) {
        is_descending_strides &= (ie_strides[i-1] >= ie_strides[i]);
    }

    // TODO: That's strong constrains and can be mitigated. IE::TensorDesc allow to permute blocked dims
    //       and may be we can achieve correct "descending strides" form which allow conversion.
    if (!is_descending_strides)
        THROW_IE_EXCEPTION << "Unsupported case for conversion";

    std::vector<size_t> outer_order(outer_ndims, outer_ndims + 1); // outer_order[i] is index of stride for i-th dimension
    for (size_t i = 0; i < outer_ndims; i++) {
        outer_order[ie_order[i]] = i;
    }
    bool outer_is_correct_permutation_of_n =
            std::find(outer_order.begin(), outer_order.end(), outer_ndims + 1) == outer_order.end();

    if (!outer_is_correct_permutation_of_n)
        THROW_IE_EXCEPTION << "Unsupported case for conversion";

    bool inner_block_are_dense = one_of(ie_strides.back(), 0, 1);  // stride 1 - is dense case, 0 - broad casted
    for (int i = outer_ndims; i < ie_strides.size() - 1; i++) {
        inner_block_are_dense &= (ie_strides[i] == ie_strides[i+1] * ie_blkdDims[i+1]);
    }

    if (!inner_block_are_dense)
        THROW_IE_EXCEPTION << "Unsupported case for conversion";

    bool inner_pad_offsets_is_zero = std::all_of(ie_offsetsToData.begin() + outer_ndims, ie_offsetsToData.end(),
                                                 [](size_t pad) { return  pad == 0; });

    if (!inner_pad_offsets_is_zero)
        THROW_IE_EXCEPTION << "Unsupported case for conversion";

    // Fill general memory desc fields
    desc.data.format_kind = dnnl_blocked;
    desc.data.data_type = memory::convert_to_c(MKLDNNMemory::convertToDataType(tDesc.getPrecision()));
    desc.data.ndims = dims.size();
    desc.data.offset0 = tDesc.getBlockingDesc().getOffsetPadding();
    std::copy(dims.begin(), dims.end(), desc.data.dims);
    std::copy(ie_offsetsToData.begin(), ie_offsetsToData.begin() + outer_ndims, desc.data.padded_offsets);
    std::fill(desc.data.padded_dims, desc.data.padded_dims + outer_ndims, 1);
    for (size_t i = 0; i < ie_order.size(); i++) {
        auto idx = ie_order[i];
        desc.data.padded_dims[idx] *= ie_blkdDims[i];
    }

    // Fill blocking desc
    auto &dnn_blk_desc = desc.data.format_desc.blocking;
    dnn_blk_desc.inner_nblks = inner_ndims;
    std::copy(ie_blkdDims.end() - inner_ndims, ie_blkdDims.end(), dnn_blk_desc.inner_blks);
    std::copy(ie_order.end() - inner_ndims, ie_order.end(), dnn_blk_desc.inner_idxs);
    for (size_t i = 0; i < outer_ndims; i++) {
        dnn_blk_desc.strides[i] = ie_strides[outer_order[i]];
    }
}

bool MKLDNNMemoryDesc::blocksExtended() const {
    for (int i = 0; i < desc.data.ndims; i++) {
        if (desc.data.dims[i] != desc.data.padded_dims[i])
            return true;
    }
    return false;
}

}  // namespace MKLDNNPlugin
