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

        mkldnn::stream loc_stream(eng, stream::flags::in_order);
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
    mkldnn::reorder reorderPrim(src.GetPrimitive(), GetPrimitive());
    mkldnn::stream loc_stream(eng, stream::flags::in_order);
    reorderPrim.execute(loc_stream, *src.prim, *this->prim);

    if (ftz && src.GetDataType() == mkldnn::memory::data_type::f32 && prim->get_desc().data.format_kind == dnnl_format_kind_wino &&
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
            return memory::format_tag::x;
        case 1:
            return memory::format_tag::x;
        case 2:
            return memory::format_tag::nc;
        case 3:
            return memory::format_tag::tnc;
        case 4:
            return memory::format_tag::nchw;
        case 5:
            return memory::format_tag::ncdhw;
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
        case memory::f32:
            return Precision::FP32;
        case memory::u8:
            return Precision::U8;
        case memory::s8:
            return Precision::I8;
        case memory::s16:
            return Precision::I16;
        case memory::s32:
            return Precision::I32;
        case memory::bin:
            return Precision::BIN;
        case memory::bf16:
            return Precision::BF16;
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

MKLDNNMemoryDesc::operator InferenceEngine::TensorDesc() const {
//    Precision precision;
//    switch (desc.data.data_type) {
//        case mkldnn_f32:
//            precision = Precision::FP32;
//            break;
//        case mkldnn_u8:
//            precision = Precision::U8;
//            break;
//        case mkldnn_s8:
//            precision = Precision::I8;
//            break;
//        case mkldnn_s16:
//            precision = Precision::I16;
//            break;
//        case mkldnn_s32:
//            precision = Precision::I32;
//            break;
//        case mkldnn_bin:
//            precision = Precision::BIN;
//            break;
//        case mkldnn_bf16:
//            precision = Precision::BF16;
//            break;
//        default:
//            THROW_IE_EXCEPTION << "Cannot cast to TensorDesc. Unsupported precision!";
//    }
//    Layout layout;
//    SizeVector order;
//    SizeVector blkDims;
//    auto blkInfo = desc.data.layout_desc.blocking;
//    auto offset = static_cast<size_t>(blkInfo.offset_padding);
//    SizeVector offsetsForDims;
//    SizeVector dims = getDims().ToSizeVector();
//    switch (getFormat()) {
//        case memory::format_tag_undef:
//            THROW_IE_EXCEPTION << "Cannot cast to tensor desc. Format is undefined!";
//        case memory::any:
//            layout = Layout::ANY;
//            return TensorDesc(precision, dims, layout);
//        case memory::x:
//            layout = Layout::C;
//            order = {0};
//            blkDims = dims;
//            break;
//        case memory::oi:
//        case memory::nc:
//            layout = Layout::NC;
//            order = {0, 1};
//            blkDims = dims;
//            break;
//        case memory::tnc:
//            layout = Layout::CHW;
//            order = {0, 1, 2};
//            blkDims = dims;
//            break;
//        case memory::ntc:
//            layout = Layout::CHW;
//            order = {1, 0, 2};
//            blkDims = {static_cast<size_t>(dims[1]),
//                       static_cast<size_t>(dims[0]),
//                       static_cast<size_t>(dims[2])};
//            break;
//        case memory::oihw:
//        case memory::nchw:
//            layout = Layout::NCHW;
//            order = {0, 1, 2, 3};
//            blkDims = dims;
//            break;
//        case memory::hwio:
//            layout = Layout::BLOCKED;
//            order = {2, 3, 1, 0};
//            blkDims = dims;
//            break;
//        case memory::dhwio:
//            layout = Layout::BLOCKED;
//            order = {2, 3, 4, 1, 0};
//            blkDims = dims;
//            break;
//        case memory::hwigo:
//            layout = Layout::BLOCKED;
//            order = {3, 4, 2, 0, 1};
//            blkDims = dims;
//            break;
//        case memory::dhwigo:
//            order = {3, 4, 5, 2, 0, 1};
//            blkDims = dims;
//            layout = Layout::BLOCKED;
//            break;
//        case memory::oidhw:
//        case memory::ncdhw:
//            layout = Layout::NCDHW;
//            order = {0, 1, 2, 3, 4};
//            blkDims = dims;
//            break;
//        case memory::ohwi:
//        case memory::nhwc:
//            layout = Layout::NHWC;
//            order = {0, 2, 3, 1};
//            if (precision == Precision::BIN) {
//                blkDims = {static_cast<size_t>(dims[0]),
//                           static_cast<size_t>(dims[2]),
//                           static_cast<size_t>(dims[3]),
//                           static_cast<size_t>(rnd_up(dims[1], 8))};
//            } else {
//                blkDims = {static_cast<size_t>(dims[0]),
//                           static_cast<size_t>(dims[2]),
//                           static_cast<size_t>(dims[3]),
//                           static_cast<size_t>(dims[1])};
//            }
//            break;
//        case memory::odhwi:
//        case memory::ndhwc:
//            layout = Layout::NDHWC;
//            order = {0, 2, 3, 4, 1};
//            blkDims = {static_cast<size_t>(dims[0]),
//                       static_cast<size_t>(dims[2]),
//                       static_cast<size_t>(dims[3]),
//                       static_cast<size_t>(dims[4]),
//                       static_cast<size_t>(dims[1])};
//            break;
//        case memory::oIhw8i:
//        case memory::nChw8c:
//            order = {0, 1, 2, 3, 1};
//            blkDims = dims;
//            blkDims[1] = blkDims[1] / 8 + (blkDims[1] % 8 ? 1 : 0);
//            blkDims.push_back(8);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::gOhwi8o:
//        case memory::nCdhw8c:
//            order = {0, 1, 2, 3, 4, 1};
//            blkDims = dims;
//            blkDims[1] = blkDims[1] / 8 + (blkDims[1] % 8 ? 1 : 0);
//            blkDims.push_back(8);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::gOdhwi8o:
//            order = {0, 1, 2, 3, 4, 5, 1};
//            blkDims = dims;
//            blkDims[1] = blkDims[1] / 8 + (blkDims[1] % 8 ? 1 : 0);
//            blkDims.push_back(8);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::nChw16c:
//            order = {0, 1, 2, 3, 1};
//            blkDims = dims;
//            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
//            blkDims.push_back(16);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::gOhwi16o:
//        case memory::nCdhw16c:
//            order = {0, 1, 2, 3, 4, 1};
//            blkDims = dims;
//            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
//            blkDims.push_back(16);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::gOdhwi16o:
//            order = {0, 1, 2, 3, 4, 5, 1};
//            blkDims = dims;
//            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
//            blkDims.push_back(16);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::Ohwi8o:
//            order = {0, 1, 2, 3, 0};
//            blkDims = dims;
//            blkDims[0] = blkDims[0] / 8 + (blkDims[0] % 8 ? 1 : 0);
//            blkDims.push_back(8);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::Ohwi16o:
//            order = {0, 1, 2, 3, 0};
//            blkDims = dims;
//            blkDims[0] = blkDims[0] / 16 + (blkDims[0] % 16 ? 1 : 0);
//            blkDims.push_back(16);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::Odhwi8o:
//            order = {0, 2, 3, 4, 1, 0};
//            blkDims = dims;
//            blkDims[0] = blkDims[0] / 8 + (blkDims[0] % 8 ? 1 : 0);
//            blkDims.push_back(8);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::Odhwi16o:
//            order = {0, 2, 3, 4, 1, 0};
//            blkDims = dims;
//            blkDims[0] = blkDims[0] / 16 + (blkDims[0] % 16 ? 1 : 0);
//            blkDims.push_back(16);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::OIhw8i8o:
//            order = {0, 1, 2, 3, 1, 0};
//            blkDims = dims;
//            blkDims[0] = blkDims[0] / 8 + (blkDims[0] % 8 ? 1 : 0);
//            blkDims[1] = blkDims[1] / 8 + (blkDims[1] % 8 ? 1 : 0);
//            blkDims.push_back(8);
//            blkDims.push_back(8);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::OIhw16i16o:
//            order = {0, 1, 2, 3, 1, 0};
//            blkDims = dims;
//            blkDims[0] = blkDims[0] / 16 + (blkDims[0] % 16 ? 1 : 0);
//            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
//            blkDims.push_back(16);
//            blkDims.push_back(16);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::OIhw8o8i:
//            order = {0, 1, 2, 3, 0, 1};
//            blkDims = dims;
//            blkDims[0] = blkDims[0] / 8 + (blkDims[0] % 8 ? 1 : 0);
//            blkDims[1] = blkDims[1] / 8 + (blkDims[1] % 8 ? 1 : 0);
//            blkDims.push_back(8);
//            blkDims.push_back(8);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::OIhw16o16i:
//            order = {0, 1, 2, 3, 0, 1};
//            blkDims = dims;
//            blkDims[0] = blkDims[0] / 16 + (blkDims[0] % 16 ? 1 : 0);
//            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
//            blkDims.push_back(16);
//            blkDims.push_back(16);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::IOhw16o16i:
//            order = {1, 0, 2, 3, 0, 1};
//            blkDims = dims;
//            blkDims[0] = blkDims[0] / 16 + (blkDims[0] % 16 ? 1 : 0);
//            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
//            blkDims.push_back(16);
//            blkDims.push_back(16);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::OIdhw8i8o:
//            order = {0, 1, 2, 3, 4, 1, 0};
//            blkDims = dims;
//            blkDims[0] = blkDims[0] / 8 + (blkDims[0] % 8 ? 1 : 0);
//            blkDims[1] = blkDims[1] / 8 + (blkDims[1] % 8 ? 1 : 0);
//            blkDims.push_back(8);
//            blkDims.push_back(8);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::OIdhw16i16o:
//            order = {0, 1, 2, 3, 4, 1, 0};
//            blkDims = dims;
//            blkDims[0] = blkDims[0] / 16 + (blkDims[0] % 16 ? 1 : 0);
//            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
//            blkDims.push_back(16);
//            blkDims.push_back(16);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::OIdhw8o8i:
//            order = {0, 1, 2, 3, 4, 1, 0};
//            blkDims = dims;
//            blkDims[0] = blkDims[0] / 8 + (blkDims[0] % 8 ? 1 : 0);
//            blkDims[1] = blkDims[1] / 8 + (blkDims[1] % 8 ? 1 : 0);
//            blkDims.push_back(8);
//            blkDims.push_back(8);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::OIdhw16o16i:
//            order = {0, 1, 2, 3, 4, 0, 1};
//            blkDims = dims;
//            blkDims[0] = blkDims[0] / 16 + (blkDims[0] % 16 ? 1 : 0);
//            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
//            blkDims.push_back(16);
//            blkDims.push_back(16);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::gOIhw4o4i:
//            order = {0, 1, 2, 3, 4, 1, 2};
//            blkDims = dims;
//            blkDims[1] = blkDims[1] / 4 + (blkDims[1] % 4 ? 1 : 0);
//            blkDims[2] = blkDims[2] / 4 + (blkDims[2] % 4 ? 1 : 0);
//            blkDims.push_back(4);
//            blkDims.push_back(4);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::gOIhw8i8o:
//            order = {0, 1, 2, 3, 4, 2, 1};
//            blkDims = dims;
//            blkDims[1] = blkDims[1] / 8 + (blkDims[1] % 8 ? 1 : 0);
//            blkDims[2] = blkDims[2] / 8 + (blkDims[2] % 8 ? 1 : 0);
//            blkDims.push_back(8);
//            blkDims.push_back(8);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::gOIhw8o8i:
//            order = {0, 1, 2, 3, 4, 1, 2};
//            blkDims = dims;
//            blkDims[1] = blkDims[1] / 8 + (blkDims[1] % 8 ? 1 : 0);
//            blkDims[2] = blkDims[2] / 8 + (blkDims[2] % 8 ? 1 : 0);
//            blkDims.push_back(8);
//            blkDims.push_back(8);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::gOIhw16i16o:
//            order = {0, 1, 2, 3, 4, 2, 1};
//            blkDims = dims;
//            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
//            blkDims[2] = blkDims[2] / 16 + (blkDims[2] % 16 ? 1 : 0);
//            blkDims.push_back(16);
//            blkDims.push_back(16);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::gOIhw16o16i:
//            order = {0, 1, 2, 3, 4, 1, 2};
//            blkDims = dims;
//            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
//            blkDims[2] = blkDims[2] / 16 + (blkDims[2] % 16 ? 1 : 0);
//            blkDims.push_back(16);
//            blkDims.push_back(16);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::OhIw8o4i:
//            order = {0, 2, 1, 3, 0, 1};
//            blkDims = dims;
//            blkDims[0] = blkDims[0] / 8 + (blkDims[0] % 8 ? 1 : 0);
//            blkDims[1] = blkDims[1] / 4 + (blkDims[1] % 4 ? 1 : 0);
//            blkDims.push_back(8);
//            blkDims.push_back(4);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::OIhw4i16o4i:
//            order = {0, 1, 2, 3, 1, 0, 1};
//            blkDims = dims;
//            blkDims[0] = blkDims[0] / 16 + (blkDims[0] % 16 ? 1 : 0);
//            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
//            blkDims.push_back(4);
//            blkDims.push_back(16);
//            blkDims.push_back(4);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::gOIhw2i8o4i:
//            order = {0, 1, 2, 3, 4, 2, 1, 2};
//            blkDims = dims;
//            blkDims[1] = blkDims[1] / 8 + (blkDims[1] % 8 ? 1 : 0);
//            blkDims[2] = blkDims[2] / 8 + (blkDims[2] % 8 ? 1 : 0);
//            blkDims.push_back(2);
//            blkDims.push_back(8);
//            blkDims.push_back(4);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::gOIhw4i16o4i:
//            order = {0, 1, 2, 3, 4, 2, 1, 2};
//            blkDims = dims;
//            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
//            blkDims[2] = blkDims[2] / 16 + (blkDims[2] % 16 ? 1 : 0);
//            blkDims.push_back(4);
//            blkDims.push_back(16);
//            blkDims.push_back(4);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::OdhIw8o4i:
//            order = {0, 2, 3, 1, 4, 0, 1};
//            blkDims = dims;
//            blkDims[0] = blkDims[0] / 8 + (blkDims[0] % 8 ? 1 : 0);
//            blkDims[1] = blkDims[1] / 4 + (blkDims[1] % 4 ? 1 : 0);
//            blkDims.push_back(8);
//            blkDims.push_back(4);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::OIdhw4i16o4i:
//            order = {0, 1, 2, 3, 4, 1, 0, 1};
//            blkDims = dims;
//            blkDims[0] = blkDims[0] / 16 + (blkDims[0] % 16 ? 1 : 0);
//            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
//            blkDims.push_back(4);
//            blkDims.push_back(16);
//            blkDims.push_back(4);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::gOhIw8o4i:
//            order = {0, 1, 3, 2, 4, 1, 2};
//            blkDims = dims;
//            blkDims[1] = blkDims[1] / 8 + (blkDims[1] % 8 ? 1 : 0);
//            blkDims[2] = blkDims[2] / 4 + (blkDims[2] % 4 ? 1 : 0);
//            blkDims.push_back(8);
//            blkDims.push_back(4);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::gOdhIw8o4i:
//            order = {0, 1, 3, 4, 2, 5, 1, 2};
//            blkDims = dims;
//            blkDims[1] = blkDims[1] / 8 + (blkDims[1] % 8 ? 1 : 0);
//            blkDims[2] = blkDims[2] / 4 + (blkDims[2] % 4 ? 1 : 0);
//            blkDims.push_back(8);
//            blkDims.push_back(4);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::gOIdhw4i16o4i:
//            order = {0, 1, 2, 3, 4, 5, 2, 1, 2};
//            blkDims = dims;
//            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
//            blkDims[2] = blkDims[2] / 16 + (blkDims[2] % 16 ? 1 : 0);
//            blkDims.push_back(4);
//            blkDims.push_back(16);
//            blkDims.push_back(4);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::gOIdhw4i4o:
//            order = {0, 1, 2, 3, 4, 5, 2, 1};
//            blkDims = dims;
//            blkDims[1] = blkDims[1] / 4 + (blkDims[1] % 4 ? 1 : 0);
//            blkDims[2] = blkDims[2] / 4 + (blkDims[2] % 4 ? 1 : 0);
//            blkDims.push_back(4);
//            blkDims.push_back(4);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::gOIdhw8i8o:
//            order = {0, 1, 2, 3, 4, 5, 2, 1};
//            blkDims = dims;
//            blkDims[1] = blkDims[1] / 8 + (blkDims[1] % 8 ? 1 : 0);
//            blkDims[2] = blkDims[2] / 8 + (blkDims[2] % 8 ? 1 : 0);
//            blkDims.push_back(8);
//            blkDims.push_back(8);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::gOIdhw16i16o:
//            order = {0, 1, 2, 3, 4, 5, 2, 1};
//            blkDims = dims;
//            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
//            blkDims[2] = blkDims[2] / 16 + (blkDims[2] % 16 ? 1 : 0);
//            blkDims.push_back(16);
//            blkDims.push_back(16);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::goihw:
//            order = {0, 1, 2, 3, 4};
//            blkDims = dims;
//            layout = Layout::GOIHW;
//            break;
//        case memory::goidhw:
//            order = {0, 1, 2, 3, 4, 5};
//            blkDims = dims;
//            layout = Layout::GOIDHW;
//            break;
//        case memory::Goihw8g:
//            order = {0, 1, 2, 3, 4, 0};
//            blkDims = dims;
//            blkDims[0] = blkDims[0] / 8 + (blkDims[0] % 8 ? 1 : 0);
//            blkDims.push_back(8);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::Goihw16g:
//            order = {0, 1, 2, 3, 4, 0};
//            blkDims = dims;
//            blkDims[0] = blkDims[0] / 16 + (blkDims[0] % 16 ? 1 : 0);
//            blkDims.push_back(16);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::Goidhw8g:
//            order = {0, 1, 2, 3, 4, 5, 0};
//            blkDims = dims;
//            blkDims[0] = blkDims[0] / 8 + (blkDims[0] % 8 ? 1 : 0);
//            blkDims.push_back(8);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::Goidhw16g:
//            order = {0, 1, 2, 3, 4, 5, 0};
//            blkDims = dims;
//            blkDims[0] = blkDims[0] / 16 + (blkDims[0] % 16 ? 1 : 0);
//            blkDims.push_back(16);
//            layout = Layout::BLOCKED;
//            break;
//        case memory::blocked:
//            order.clear();
//            blkDims = dims;
//            for (size_t i = 0; i < blkDims.size(); i++) {
//                order.push_back(i);
//                if ((i && blkInfo.strides[0][i - 1] < blkInfo.strides[0][i]) || blkInfo.block_dims[i] != 1) {
//                    THROW_IE_EXCEPTION << "Cannot cast to tensor desc."
//                                       << " Unsupported blocked format.";
//                }
//            }
//            if (order.size() == 3 && order[0] == 0 && order[1] == 1 && order[2] == 2)
//                layout = Layout::CHW;
//            else
//                layout = Layout::BLOCKED;
//            break;
//        default:
//            THROW_IE_EXCEPTION << "Cannot cast to tensor desc. Format is unsupported!";
//    }
//
//    SizeVector strides(blkDims.size());
//
//    if (layout == Layout::NHWC || layout == Layout::NDHWC || layout == Layout::CHW) {
//        for (size_t i = 0; i < order.size(); i++) {
//            strides[i] = static_cast<size_t>(blkInfo.strides[0][order[i]]);
//        }
//    } else {
//        strides[blkDims.size() - 1] = 1;
//        for (size_t i = 2; i <= order.size(); i++) {
//            if (blkDims.size() - i < dims.size()) {
//                strides[blkDims.size() - i] = static_cast<size_t>(blkInfo.strides[0][order[blkDims.size() - i]]);
//            } else {
//                strides[blkDims.size() - i] = strides[blkDims.size() - i + 1] * blkDims[blkDims.size() - i + 1];
//            }
//        }
//    }
//
//    for (size_t i = 0; i < blkDims.size() && i < TENSOR_MAX_DIMS; i++) {
//        if (i < dims.size())
//            offsetsForDims.push_back(blkInfo.offset_padding_to_data[i]);
//        else
//            offsetsForDims.push_back(0);
//    }
//
//    TensorDesc tensorDesc(precision, dims, {blkDims, order, offset, offsetsForDims, strides});
//
//    tensorDesc.setLayout(layout);
//    return tensorDesc;
    THROW_IE_EXCEPTION << "Converter is not implemented";
    return {};
}

MKLDNNMemoryDesc::MKLDNNMemoryDesc(const TensorDesc& tDesc):
        desc({}, mkldnn::memory::data_type::f32, mkldnn::memory::format_tag::undef) {
    mkldnn::memory::data_type data_type;
    switch (tDesc.getPrecision()) {
        case Precision::FP32:
            data_type = mkldnn::memory::data_type::f32;
            break;
        case Precision::U8:
            data_type = mkldnn::memory::data_type::u8;
            break;
        case Precision::I8:
            data_type = mkldnn::memory::data_type::s8;
            break;
        case Precision::I16:
            data_type = mkldnn::memory::data_type::s16;
            break;
        case Precision::I32:
            data_type = mkldnn::memory::data_type::s32;
            break;
        case Precision::BIN:
            data_type = mkldnn::memory::data_type::bin;
            break;
        case Precision::BOOL:
            data_type = mkldnn::memory::data_type::u8;
            break;
        case Precision::BF16:
            data_type = mkldnn::memory::data_type::bf16;
            break;
        default:
            THROW_IE_EXCEPTION << "Cannot create MKLDNNMemoryDesc from TensorDesc. Unsupported precision: " << tDesc.getPrecision();
    }

    mkldnn::memory::format mkldnnFormat = memory::format::format_undef;
    SizeVector blkdDims = tDesc.getBlockingDesc().getBlockDims();
    SizeVector order = tDesc.getBlockingDesc().getOrder();
    SizeVector offsetsToData = tDesc.getBlockingDesc().getOffsetPaddingToData();
    SizeVector strides = tDesc.getBlockingDesc().getStrides();
    auto realDims = MKLDNNDims(tDesc.getDims());
    switch (tDesc.getLayout()) {
        case ANY:
            mkldnnFormat = memory::format::any;
            break;
        case NCHW:
            mkldnnFormat = memory::format::nchw;
            break;
        case NCDHW:
            mkldnnFormat = memory::format::ncdhw;
            break;
        case NHWC:
            mkldnnFormat = memory::format::nhwc;
            break;
        case NDHWC:
            mkldnnFormat = memory::format::ndhwc;
            break;
        case OIHW:
            mkldnnFormat = memory::format::oihw;
            break;
        case GOIHW:
            mkldnnFormat = memory::format::goihw;
            break;
        case OIDHW:
            mkldnnFormat = memory::format::oidhw;
            break;
        case GOIDHW:
            mkldnnFormat = memory::format::goidhw;
            break;
        case SCALAR:
        case C:
            mkldnnFormat = memory::format::x;
            break;
        case CHW:
            if (order == SizeVector{0, 1, 2})
                mkldnnFormat = memory::format::tnc;
            else if (order == SizeVector{1, 0, 2})
                mkldnnFormat = memory::format::ntc;
            else
                mkldnnFormat = memory::format::blocked;
            break;
        case HW:
        case NC:
            mkldnnFormat = memory::format::nc;
            break;
        case BLOCKED:
            mkldnnFormat = memory::format::blocked;
            if (realDims.ndims() == 1) {
                mkldnnFormat = memory::format::x;
            } else if (realDims.ndims() == 2) {
                mkldnnFormat = memory::format::nc;
            } else if (realDims.ndims() == 3) {
                if (order == SizeVector{0, 1, 2})
                    mkldnnFormat = memory::format::tnc;
                else if (order == SizeVector{1, 0, 2})
                    mkldnnFormat = memory::format::ntc;
            } else if (realDims.ndims() == 4) {
                if (order.size() == 7 &&
                    order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3 && order[4] == 1 && order[5] == 0 && order[6] == 1) {
                    if (blkdDims[4] == 4 && blkdDims[5] == 16 && blkdDims[6] == 4) {
                        mkldnnFormat = memory::format::OIhw4i16o4i;
                    }
                } else if (order.size() == 6 && order[0] == 0 && order[1] == 2 && order[2] == 1 && order[3] == 3 && order[4] == 0 && order[5] == 1) {
                    if (blkdDims[4] == 8 && blkdDims[5] == 4) {
                        mkldnnFormat = memory::format::OhIw8o4i;
                    }
                } else if (order.size() == 6 && order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3 && order[4] == 1 && order[5] == 0) {
                    if (blkdDims[4] == 8 && blkdDims[5] == 8) {
                        mkldnnFormat = memory::format::OIhw8i8o;
                    } else if (blkdDims[4] == 16 && blkdDims[5] == 16) {
                        mkldnnFormat = memory::format::OIhw16i16o;
                    }
                } else if (order.size() == 6 && order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3 && order[4] == 0 && order[5] == 1) {
                    if (blkdDims[4] == 8 && blkdDims[5] == 8) {
                        mkldnnFormat = memory::format::OIhw8o8i;
                    } else if (blkdDims[4] == 16 && blkdDims[5] == 16) {
                        mkldnnFormat = memory::format::OIhw16o16i;
                    }
                } else if (order.size() == 6 && order[0] == 1 && order[1] == 0 && order[2] == 2 && order[3] == 3 && order[4] == 0 && order[5] == 1) {
                    if (blkdDims[4] == 16 && blkdDims[5] == 16) {
                        mkldnnFormat = memory::format::IOhw16o16i;
                    }
                } else if (order.size() == 5 && order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3 && order[4] == 0) {
                    if (blkdDims[4] == 8) {
                        mkldnnFormat = memory::format::Ohwi8o;
                    } else if (blkdDims[4] == 16) {
                        mkldnnFormat = memory::format::Ohwi16o;
                    }
                } else if (order.size() == 5 && order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3 && order[4] == 1) {
                    if (blkdDims[4] == 8) {
                        mkldnnFormat = memory::format::nChw8c;
                    } else if (blkdDims[4] == 16) {
                        mkldnnFormat = memory::format::nChw16c;
                    }
                } else if (order.size() == 4) {
                    if (order[0] == 2 && order[1] == 3 && order[2] == 1 && order[3] == 0) {
                        mkldnnFormat = memory::format::hwio;
                    } else if (order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3) {
                        mkldnnFormat = memory::format::nchw;
                    } else if (order[0] == 0 && order[1] == 2 && order[2] == 3 && order[3] == 1) {
                        mkldnnFormat = memory::format::nhwc;
                    }
                }
            } else if (realDims.ndims() == 5) {
                if (order.size() == 5 && order[0] == 2 && order[1] == 3 && order[2] == 4 && order[3] == 1 && order[4] == 0) {
                    mkldnnFormat = memory::format::dhwio;
                } else if (order.size() == 5 && order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3 && order[4] == 4) {
                    mkldnnFormat = memory::format::goihw;
                } else if (order.size() == 5 && order[0] == 3 && order[1] == 4 && order[2] == 2 && order[3] == 0 && order[4] == 1) {
                    mkldnnFormat = memory::format::hwigo;
                } else if (order.size() == 6 && order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3 && order[4] == 4 && order[5] == 0) {
                    if (blkdDims[5] == 8) {
                        mkldnnFormat = memory::format::Goihw8g;
                    } else if (blkdDims[5] == 16) {
                        mkldnnFormat = memory::format::Goihw16g;
                    }
                } else if (order.size() == 6 &&
                        order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3 && order[4] == 4 && order[5] == 1) {
                    if (blkdDims[5] == 8) {
                        mkldnnFormat = memory::format::nCdhw8c;
                    } else if (blkdDims[5] == 16) {
                        mkldnnFormat = memory::format::nCdhw16c;
                    }
                } else if (order.size() == 6 &&
                           order[0] == 0 && order[1] == 2 && order[2] == 3 && order[3] == 4 && order[4] == 1 && order[5] == 0) {
                    if (blkdDims[5] == 8) {
                        mkldnnFormat = memory::format::Odhwi8o;
                    } else if (blkdDims[5] == 16) {
                        mkldnnFormat = memory::format::Odhwi16o;
                    }
                } else if (order.size() == 7 &&
                           order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3 && order[4] == 4 && order[5] == 1 && order[6] == 0) {
                    if (blkdDims[6] == 8) {
                        mkldnnFormat = memory::format::OIdhw8i8o;
                    } else if (blkdDims[6] == 16) {
                        mkldnnFormat = memory::format::OIdhw16i16o;
                    }
                } else if (order.size() == 7 &&
                           order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3 && order[4] == 4 && order[5] == 0 && order[6] == 1) {
                    if (blkdDims[6] == 8) {
                        mkldnnFormat = memory::format::OIdhw8o8i;
                    } else if (blkdDims[6] == 16) {
                        mkldnnFormat = memory::format::OIdhw16o16i;
                    }
                } else if (order.size() == 7 &&
                           order[0] == 0 && order[1] == 2 && order[2] == 3 && order[3] == 1 && order[4] == 4 && order[5] == 0 && order[6] == 1) {
                    if (blkdDims[5] == 8) {
                        mkldnnFormat = memory::format::OdhIw8o4i;
                    }
                } else if (order.size() == 8 &&
                           order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3 && order[4] == 4 && order[5] == 1 && order[6] == 0 &&
                           order[7] == 1) {
                    if (blkdDims[7] == 4) {
                        mkldnnFormat = memory::format::OIdhw4i16o4i;
                    }
                } else if (order.size() == 7 &&
                           order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3 && order[4] == 4 && order[5] == 2 && order[6] == 1) {
                    if (blkdDims[6] == 4) {
                        mkldnnFormat = memory::format::gOIhw4i4o;
                    } else if (blkdDims[6] == 8) {
                        mkldnnFormat = memory::format::gOIhw8i8o;
                    } else if (blkdDims[6] == 16) {
                        mkldnnFormat = memory::format::gOIhw16i16o;
                    }
                } else if (order.size() == 7 &&
                           order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3 && order[4] == 4 && order[5] == 1 && order[6] == 2) {
                    if (blkdDims[6] == 4) {
                        mkldnnFormat = memory::format::gOIhw4o4i;
                    } else if (blkdDims[6] == 8) {
                        mkldnnFormat = memory::format::gOIhw8o8i;
                    } else if (blkdDims[6] == 16) {
                        mkldnnFormat = memory::format::gOIhw16o16i;
                    }
                } else if (order.size() == 7 &&
                           order[0] == 0 && order[1] == 1 && order[2] == 3 && order[3] == 2 && order[4] == 4 && order[5] == 1 && order[6] == 2) {
                    if (blkdDims[5] == 8 && blkdDims[6] == 4) {
                        mkldnnFormat = memory::format::gOhIw8o4i;
                    }
                } else if (order.size() == 8 &&
                           order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3 && order[4] == 4 &&
                           order[5] == 2 && order[6] == 1 && order[7] == 2) {
                    if (blkdDims[5] == 2 && blkdDims[6] == 8 && blkdDims[7] == 4) {
                        mkldnnFormat = memory::format::gOIhw2i8o4i;
                    } else if (blkdDims[5] == 4 && blkdDims[6] == 16 && blkdDims[7] == 4) {
                        mkldnnFormat = memory::format::gOIhw4i16o4i;
                    }
                } else if (order.size() == 5) {
                    if (order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3 && order[4] == 4) {
                        mkldnnFormat = memory::format::ncdhw;
                    } else if (order[0] == 0 && order[1] == 2 && order[2] == 3 && order[3] == 4 && order[4] == 1) {
                        mkldnnFormat = memory::format::ndhwc;
                    }
                }
            } else if (realDims.ndims() == 6) {
                if (order.size() == 6 && order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3 && order[4] == 4 && order[5] == 5) {
                    mkldnnFormat = memory::format::goidhw;
                } else if (order.size() == 6 && order[0] == 3 && order[1] == 4 && order[2] == 5 && order[3] == 2 && order[4] == 0 && order[5] == 1) {
                    mkldnnFormat = memory::format::dhwigo;
                } else if (order.size() == 7 &&
                           order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3 && order[4] == 4 && order[5] == 5 && order[6] == 0) {
                    if (blkdDims[6] == 8) {
                        mkldnnFormat = memory::format::Goidhw8g;
                    } else if (blkdDims[6] == 16) {
                        mkldnnFormat = memory::format::Goidhw16g;
                    }
                } else if (order.size() == 7 &&
                           order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3 && order[4] == 4 && order[5] == 5 && order[6] == 1) {
                    if (blkdDims[6] == 8) {
                        mkldnnFormat = memory::format::gOdhwi8o;
                    } else if (blkdDims[6] == 16) {
                        mkldnnFormat = memory::format::gOdhwi16o;
                    }
                } else if (order.size() == 8 &&
                           order[0] == 0 && order[1] == 1 && order[2] == 3 && order[3] == 4 && order[4] == 2 && order[5] == 5 &&
                           order[6] == 1 && order[7] == 2) {
                    if (blkdDims[6] == 8 && blkdDims[7] == 4) {
                        mkldnnFormat = memory::format::gOdhIw8o4i;
                    }
                } else if (order.size() == 8 &&
                           order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3 && order[4] == 4 && order[5] == 5 &&
                           order[6] == 2 && order[7] == 1) {
                    if (blkdDims[6] == 4 && blkdDims[7] == 4) {
                        mkldnnFormat = memory::format::gOIdhw4i4o;
                    } else if (blkdDims[6] == 8 && blkdDims[7] == 8) {
                        mkldnnFormat = memory::format::gOIdhw8i8o;
                    } else if (blkdDims[6] == 16 && blkdDims[7] == 16) {
                        mkldnnFormat = memory::format::gOIdhw16i16o;
                    }
                } else if (order.size() == 9 &&
                           order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3 && order[4] == 4 && order[5] == 5 &&
                           order[6] == 2 && order[7] == 1 && order[8] == 2) {
                    if (blkdDims[6] == 4 && blkdDims[7] == 16 && blkdDims[8] == 4) {
                        mkldnnFormat = memory::format::gOIdhw4i16o4i;
                    }
                }
            }
            break;
        case CN:
            mkldnnFormat = memory::format::blocked;
            break;
    }
    if (mkldnnFormat == memory::format_undef)
        THROW_IE_EXCEPTION << "Cannot detect the right memory format!";

    bool notDefault = false;
    size_t currentStride = 1;
    for (size_t i = 0; i < order.size(); i++) {
        if (offsetsToData[i] != 0) {
            notDefault = true;
            break;
        }
        if (strides[strides.size() - (1 +i)] != currentStride) {
            notDefault = true;
            break;
        }
        currentStride *= blkdDims[blkdDims.size() - (1 + i)];
    }

    bool blocked = false;
    std::unordered_set<size_t> exist_order;
    for (auto& ord : order) {
        if (exist_order.find(ord) != exist_order.end()) {
            blocked = true;
            break;
        }
        exist_order.insert(ord);
    }

    if (notDefault && mkldnnFormat == memory::blocked && blocked)
        THROW_IE_EXCEPTION << "Currently MKLDNNPlugin supports only packaged memory for unknown blocked format";

    if (mkldnnFormat == memory::blocked) {
        desc = MKLDNNMemoryDesc(realDims, data_type, memory::any);
        desc.data.format = mkldnn_blocked;

        auto& blk = desc.data.layout_desc.blocking;

        blk.offset_padding = tDesc.getBlockingDesc().getOffsetPadding();

        for (size_t i = 0; i < realDims.ndims(); i++) {
            blk.block_dims[i] = 1;
            blk.strides[1][i] = 1;
            blk.padding_dims[i] = realDims[i];
            blk.offset_padding_to_data[i] = offsetsToData[i];
        }

        int perm[TENSOR_MAX_DIMS] = {0};

        for (size_t i = 0; i < realDims.ndims(); ++i) {
            perm[i] = i;
        }

        blk.strides[0][perm[realDims.ndims() - 1]] = 1;

        for (int d = 1; d < realDims.ndims(); ++d) {
            const int prev_idx = perm[realDims.ndims() - d];
            const int curr_idx = perm[realDims.ndims() - 1 - d];

            blk.strides[0][curr_idx] = realDims[curr_idx] == 0 ? 1 : blk.strides[0][prev_idx] * (std::max)((ptrdiff_t)1, realDims[prev_idx]);
        }
    } else {
        desc = MKLDNNMemoryDesc(realDims, data_type, mkldnnFormat);
    }

    desc.data.layout_desc.blocking.offset_padding = tDesc.getBlockingDesc().getOffsetPadding();
    for (size_t i = 0; i < tDesc.getBlockingDesc().getOffsetPaddingToData().size() && i < TENSOR_MAX_DIMS; i++) {
        desc.data.layout_desc.blocking.offset_padding_to_data[i] = static_cast<ptrdiff_t>(offsetsToData[i]);
    }

    if (notDefault) {
        for (size_t i = 0; i < strides.size() && i < desc.data.ndims; i++) {
            desc.data.layout_desc.blocking.strides[0][i] = static_cast<ptrdiff_t>(strides[order[i]]);
        }
    }
}

bool MKLDNNMemoryDesc::blocksExtended() const {
    return desc.data.format_desc.blocking.inner_nblks != 0;
}

}  // namespace MKLDNNPlugin
