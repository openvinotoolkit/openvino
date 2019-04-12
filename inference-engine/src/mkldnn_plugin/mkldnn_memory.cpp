// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <utility>

#include <mkldnn_types.h>
#include "mkldnn_memory.h"
#include "mkldnn_node.h"
#include "mkldnn_extension_utils.h"

using namespace InferenceEngine;
using namespace mkldnn;

namespace MKLDNNPlugin {

MKLDNNMemory::MKLDNNMemory(const engine& eng) : eng(eng) {}

size_t MKLDNNMemory::GetSize() const {
    uint8_t itemSize = MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(GetDataType()));

    auto desc = GetDescriptor();
    std::vector<int> dims(desc.data.layout_desc.blocking.padding_dims,
                          desc.data.layout_desc.blocking.padding_dims + desc.data.ndims);
    return std::accumulate(std::begin(dims), std::end(dims), (size_t) 1, std::multiplies<size_t>()) * itemSize;
}

void MKLDNNMemory::Create(memory::dims dims, memory::data_type data_type, memory::format format, const void* data) {
    if (!isConsistant(dims, format)) {
        THROW_IE_EXCEPTION << "dims and format are inconsistent.";
    }

    if (format == memory::blocked) {
        format = memory::any;
    }

    memory::desc desc = MKLDNNMemoryDesc({dims}, data_type, format);

    if (format == memory::any) {
        CreateBlockingDesc(desc);
    }

    Create(desc, data);
}

void MKLDNNMemory::Create(const mkldnn::memory::desc& desc, const void *data) {
    auto primitive_desc = memory::primitive_desc(desc, eng);
    uint8_t itemSize = MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(desc.data.data_type));

    if (data == nullptr) {
        prim.reset(new memory(primitive_desc));

        size_t real_size = 0;
        if (desc.data.format == mkldnn_wino_fmt)
            return;
        if (prim->get_primitive_desc().desc().data.ndims > 0) {
            real_size = static_cast<size_t>(prim->get_primitive_desc().desc().data.layout_desc.blocking.padding_dims[0]);
            for (int i = 1; i < prim->get_primitive_desc().desc().data.ndims; i++) {
                real_size *= prim->get_primitive_desc().desc().data.layout_desc.blocking.padding_dims[i];
            }
        }
        uint8_t* dataPtr = static_cast<uint8_t*>(GetData());
        dataPtr += itemSize * prim->get_primitive_desc().desc().data.layout_desc.blocking.offset_padding;

        memset(dataPtr, 0, real_size * itemSize);
    } else {
        // MKLDNN accepts not a const data, probably need to remove some level of consteness in a call stack
        prim.reset(new memory(primitive_desc, const_cast<void*>(data)));
    }
}

void MKLDNNMemory::SetData(memory::data_type dataType, memory::format format, const void* data, size_t size, bool ftz) const {
    uint8_t itemSize = MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(dataType));

    if (static_cast<mkldnn_memory_format_t>(format) != GetDescriptor().data.format ||
            GetDataType() != dataType) {
        auto memData = GetDescriptor().data;

        std::vector<ptrdiff_t> dims(memData.dims, memData.dims + memData.ndims);

        auto dataType = GetDataType();

        MKLDNNMemory src(eng);
        src.Create(dims, dataType, format, data);

        std::shared_ptr<mkldnn::reorder> pReorder =
                std::shared_ptr<mkldnn::reorder>(new mkldnn::reorder(src.GetPrimitive(), GetPrimitive()));

        mkldnn::stream(stream::kind::eager).submit({*pReorder});
    } else {
        uint8_t* dataPtr = static_cast<uint8_t*>(GetData());
        // We cannot support strides for i/o blobs because it affects performance.
        dataPtr += itemSize * prim->get_primitive_desc().desc().data.layout_desc.blocking.offset_padding;
        memcpy(dataPtr, data, size);
    }

    if (ftz && dataType == mkldnn_f32) {
        // Internal blobs haven't strides yet.
        auto *memData = static_cast<float *>(GetData());
        memData += prim->get_primitive_desc().desc().data.layout_desc.blocking.offset_padding;
        size_t realSize = GetSize() / sizeof(float);
        for (size_t i = 0; i < realSize; i++) {
            if (memData[i] != 0 && (fabsf(memData[i]) < std::numeric_limits<float>::min())) {
                memData[i] = 0.0f;
            }
        }
    }
}

void MKLDNNMemory::SetData(const MKLDNNMemory& memory, bool ftz) const {
    mkldnn::reorder reorderPrim(memory.GetPrimitive(), GetPrimitive());
    mkldnn::stream(stream::kind::eager).submit({reorderPrim});

    if (ftz && memory.GetDataType() == mkldnn::memory::f32 && GetFormat() != mkldnn::memory::wino_fmt) {
        // Internal blobs haven't strides yet.
        auto *memData = static_cast<float *>(GetData());
        memData += prim->get_primitive_desc().desc().data.layout_desc.blocking.offset_padding;
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

bool MKLDNNMemory::isConsistant(memory::dims dims, memory::format format) {
    using f = mkldnn::memory::format;

    size_t ndims = 0;

    switch (format) {
        case f::x:
            ndims = 1; break;
        case f::nc:
        case f::oi:
        case f::io:
            ndims = 2; break;
        case f::ntc:
        case f::tnc:
            ndims = 3; break;
        case f::nchw:
        case f::nhwc:
        case f::chwn:
        case f::nChw8c:
        case f::nChw16c:
        case f::oihw:
        case f::ihwo:
        case f::hwio:
        case f::OIhw8i8o:
        case f::OIhw16i16o:
        case f::OIhw8o8i:
        case f::OIhw16o16i:
        case f::OIhw8i16o2i:
        case f::OIhw8o16i2o:
        case f::Ohwi8o:
        case f::Ohwi16o:
        case f::OhIw16o4i:
        case f::OIhw4i16o4i:
            ndims = 4; break;
        // DHW
        case f::ncdhw:
        case f::ndhwc:
        case f::nCdhw8c:
        case f::nCdhw16c:
        case f::oidhw:
        case f::OIdhw8i8o:
        case f::OIdhw16i16o:
        case f::OIdhw8o8i:
        case f::OIdhw16o16i:
        case f::OIdhw8i16o2i:
        case f::Odhwi8o:
        case f::Odhwi16o:
        // Group HW
        case f::hwigo:
        case f::goihw:
        case f::gOIhw8i8o:
        case f::gOIhw16i16o:
        case f::gOIhw8i16o2i:
        case f::gOIhw8o16i2o:
        case f::gOhwi8o:
        case f::gOhwi16o:
        case f::gOIhw8o8i:
        case f::gOIhw16o16i:
        case f::gOhIw16o4i:
        case f::Goihw8g:
        case f::Goihw16g:
            ndims = 5; break;
        case f::goidhw:
        case f::gOIdhw8i8o:
        case f::gOIdhw16i16o:
        case f::gOIdhw8i16o2i:
        case f::gOdhwi8o:
        case f::gOdhwi16o:
        case f::gOIdhw8o8i:
        case f::gOIdhw16o16i:
            ndims = 6; break;
        case f::format_undef:
            ndims = 0; break;
        case f::any:
        case f::wino_fmt:
        case f::blocked:
            return true;
        default:
            return false;
    }

    return (dims.size() == ndims);
}

bool MKLDNNMemory::IsPlainFormat(memory::format format) {
    std::vector<memory::format> plains = {memory::nc, memory::nchw, memory::ncdhw, memory::nhwc, memory::ndhwc, memory::chwn,
        memory::oi, memory::io, memory::oihw, memory::oidhw, memory::ihwo, memory::tnc,
        memory::goihw,
        memory::blocked};

    for (auto it : plains) {
        if (format == it) {
            return true;
        }
    }

    return false;
}

memory::format MKLDNNMemory::GetPlainFormat(memory::dims dims) {
    switch (dims.size()) {
        case 1:
            return memory::x;
        case 2:
            return memory::nc;
        case 3:
            return memory::tnc;
        case 4:
            return memory::nchw;
        case 5:
            return memory::ncdhw;
        default:
            return memory::blocked;
    }
}

InferenceEngine::Layout MKLDNNMemory::GetPlainLayout(memory::dims dims) {
    switch (dims.size()) {
        case 0: return Layout::SCALAR;
        case 1: return Layout::C;
        case 2: return Layout::NC;
        case 3: return Layout::CHW;
        case 4: return Layout::NCHW;
        default:
            return Layout::BLOCKED;
    }
}

void MKLDNNMemory::CreateBlockingDesc(memory::desc &desc) {
    auto dims = desc.data.dims;
    int ndims = desc.data.ndims;

    desc.data.format = mkldnn_blocked;

    auto& blk = desc.data.layout_desc.blocking;

    blk.offset_padding = 0;

    for (int i = 0; i < ndims; i++) {
        blk.block_dims[i] = 1;
        blk.strides[1][i] = 1;
        blk.padding_dims[i] = dims[i];
        blk.offset_padding_to_data[i] = 0;
    }

    int perm[TENSOR_MAX_DIMS] = {0};

    for (int i = 0; i < ndims; ++i) {
        perm[i] = i;
    }

    blk.strides[0][perm[ndims - 1]] = 1;

    for (int d = 1; d < ndims; ++d) {
        const int prev_idx = perm[ndims - d];
        const int curr_idx = perm[ndims - 1 - d];

        blk.strides[0][curr_idx] = dims[curr_idx] == 0 ? 1 : blk.strides[0][prev_idx] * (std::max)((ptrdiff_t)1, dims[prev_idx]);
    }
}
memory::format MKLDNNMemory::Convert(const InferenceEngine::Layout layout) {
    switch (layout) {
        case NCHW:
            return memory::nchw;
        case NHWC:
            return memory::nhwc;
        case NCDHW:
            return memory::ncdhw;
        case NDHWC:
            return memory::ndhwc;
        case CHW:
            return memory::tnc;
        case NC:
            return memory::nc;
        case C:
            return memory::x;
        default:
            return memory::blocked;
    }
}

std::string MKLDNNMemory::formatToString(memory::format fmt) {
    switch (fmt) {
        case memory::format_undef: return "undef";
        case memory::any: return "any";
        case memory::blocked: return "blocked";

        case memory::x: return "x";

        case memory::nc: return "nc";
        case memory::oi: return "oi";
        case memory::io: return "io";

        case memory::ntc: return "ntc";
        case memory::tnc: return "tnc";

        case memory::nchw: return "nchw";
        case memory::nhwc: return "nhwc";
        case memory::chwn: return "chwn";
        case memory::nChw8c: return "nChw8c";
        case memory::nChw16c: return "nChw16c";

        case memory::ncdhw: return "ncdhw";
        case memory::ndhwc: return "ndhwc";
        case memory::nCdhw8c: return "nCdhw8c";
        case memory::nCdhw16c: return "nCdhw16c";

        case memory::oihw: return "oihw";
        case memory::ihwo: return "ihwo";
        case memory::OIhw8i8o: return "OIhw8i8o";
        case memory::OIhw16i16o: return "OIhw16i16o";
        case memory::OIhw8o8i: return "OIhw8o8i";
        case memory::OIhw16o16i: return "OIhw16o16i";
        case memory::OIhw8i16o2i: return "OIhw8i16o2i";
        case memory::OIhw8o16i2o: return "OIhw8o16i2o";
        case memory::Ohwi8o: return "Ohwi8o";
        case memory::Ohwi16o: return "Ohwi16o";
        case memory::OhIw16o4i: return "OhIw16o4i";

        case memory::oidhw: return "oidhw";
        case memory::OIdhw8i8o: return "OIdhw8i8o";
        case memory::OIdhw16i16o: return "OIdhw16i16o";
        case memory::OIdhw8o8i: return "OIdhw8o8i";
        case memory::OIdhw16o16i: return "OIdhw16o16i";
        case memory::OIdhw8i16o2i: return "OIdhw8i16o2i";
        case memory::Odhwi8o: return "Odhwi8o";
        case memory::Odhwi16o: return "Odhwi16o";

        case memory::goihw: return "goihw";
        case memory::hwigo: return "hwigo";
        case memory::hwio: return "hwio";
        case memory::gOIhw8i8o: return "gOIhw8i8o";
        case memory::gOIhw16i16o: return "gOIhw16i16o";
        case memory::gOIhw8i16o2i: return "gOIhw8i16o2i";
        case memory::gOIhw8o16i2o: return "gOIhw8o16i2o";
        case memory::gOhwi8o: return "gOhwi8o";
        case memory::gOhwi16o: return "gOhwi16o";
        case memory::gOIhw8o8i: return "gOIhw8o8i";
        case memory::gOIhw16o16i: return "gOIhw16o16i";
        case memory::gOhIw16o4i: return "gOhIw16o4i";

        case memory::goidhw: return "goidhw";
        case memory::gOIdhw8i8o: return "gOIdhw8i8o";
        case memory::gOIdhw16i16o: return "gOIdhw16i16o";
        case memory::gOIdhw8i16o2i: return "gOIdhw8i16o2i";
        case memory::gOdhwi8o: return "gOdhwi8o";
        case memory::gOdhwi16o: return "gOdhwi16o";
        case memory::gOIdhw8o8i: return "gOIdhw8o8i";
        case memory::gOIdhw16o16i: return "gOIdhw16o16i";

        default: {
            THROW_IE_EXCEPTION << "Unknown data format.";
        }
    }
}

bool MKLDNNMemoryDesc::operator==(const MKLDNNMemoryDesc &rhs) const {
    auto dims_equal = [] (mkldnn_memory_desc_t ldata, mkldnn_memory_desc_t rdata) {
        if (ldata.ndims != rdata.ndims)
            return false;
        for (int i = 0; i < ldata.ndims; i++) {
            if (ldata.dims[i] != rdata.dims[i])
                return false;
        }
        return true;
    };
    auto blocking_equal = [] (mkldnn_memory_desc_t ldata, mkldnn_memory_desc_t rdata) {
        if (ldata.ndims != rdata.ndims)
            return false;
        mkldnn_blocking_desc_t lblock = ldata.layout_desc.blocking;
        mkldnn_blocking_desc_t rblock = rdata.layout_desc.blocking;
        if (lblock.offset_padding != rblock.offset_padding)
            return false;
        for (int i = 0; i < ldata.ndims; i++) {
            if (lblock.block_dims[i] != rblock.block_dims[i] ||
                lblock.offset_padding_to_data[i] != rblock.offset_padding_to_data[i] ||
                lblock.padding_dims[i] != rblock.padding_dims[i] || lblock.strides[0][i] != rblock.strides[0][i] ||
                lblock.strides[1][i] != rblock.strides[1][i])
                return false;
        }
        return true;
    };
    return dims_equal(this->desc.data, rhs.desc.data) &&
           this->desc.data.data_type == rhs.desc.data.data_type &&
           this->desc.data.format == rhs.desc.data.format &&
           this->desc.data.primitive_kind == rhs.desc.data.primitive_kind &&
           blocking_equal(this->desc.data, rhs.desc.data);
}

bool MKLDNNMemoryDesc::operator!=(const MKLDNNMemoryDesc &rhs) const {
    return !(*this == rhs);
}

MKLDNNMemoryDesc::operator mkldnn::memory::desc() const {
    return desc;
}

MKLDNNMemoryDesc::MKLDNNMemoryDesc(mkldnn::memory::dims dims, mkldnn::memory::data_type dataType,
                                   mkldnn::memory::format format): desc(dims, dataType, mkldnn::memory::any) {
    if (format != memory::blocked) {
        desc = mkldnn::memory::desc(dims, dataType, format);
        return;
    }
    MKLDNNMemory::CreateBlockingDesc(desc);
}

MKLDNNMemoryDesc::operator InferenceEngine::TensorDesc() const {
    Precision precision;
    switch (desc.data.data_type) {
        case mkldnn_f32:
            precision = Precision::FP32;
            break;
        case mkldnn_u8:
            precision = Precision::U8;
            break;
        case mkldnn_s8:
            precision = Precision::I8;
            break;
        case mkldnn_s16:
            precision = Precision::I16;
            break;
        case mkldnn_s32:
            precision = Precision::I32;
            break;
        case mkldnn_bin:
            precision = Precision::BIN;
            break;
        default:
            THROW_IE_EXCEPTION << "Cannot cast to TensorDesc. Unsupported precision!";
    }
    Layout layout;
    SizeVector order;
    SizeVector blkDims;
    auto blkInfo = desc.data.layout_desc.blocking;
    auto offset = static_cast<size_t>(blkInfo.offset_padding);
    SizeVector offsetsForDims;
    SizeVector dims = getDims().ToSizeVector();
    switch (getFormat()) {
        case memory::format_undef:
            THROW_IE_EXCEPTION << "Cannot cast to tensor desc. Format is undefined!";
        case memory::any:
            layout = Layout::ANY;
            return TensorDesc(precision, dims, layout);
        case memory::x:
            layout = Layout::C;
            order = {0};
            blkDims = dims;
            break;
        case memory::oi:
        case memory::nc:
            layout = Layout::NC;
            order = {0, 1};
            blkDims = dims;
            break;
        case memory::tnc:
            layout = Layout::CHW;
            order = {0, 1, 2};
            blkDims = dims;
            break;
        case memory::ntc:
            layout = Layout::CHW;
            order = {1, 0, 2};
            blkDims = {static_cast<size_t>(dims[1]),
                       static_cast<size_t>(dims[0]),
                       static_cast<size_t>(dims[2])};
            break;
        case memory::oihw:
        case memory::nchw:
            layout = Layout::NCHW;
            order = {0, 1, 2, 3};
            blkDims = dims;
            break;
        case memory::ncdhw:
            layout = Layout::NCDHW;
            order = {0, 1, 2, 3, 4};
            blkDims = dims;
            break;
        case memory::nhwc:
            layout = Layout::NHWC;
            order = {0, 2, 3, 1};
            if (precision == Precision::BIN) {
                blkDims = {static_cast<size_t>(dims[0]),
                           static_cast<size_t>(dims[2]),
                           static_cast<size_t>(dims[3]),
                           static_cast<size_t>(rnd_up(dims[1], 8))};
            } else {
                blkDims = {static_cast<size_t>(dims[0]),
                           static_cast<size_t>(dims[2]),
                           static_cast<size_t>(dims[3]),
                           static_cast<size_t>(dims[1])};
            }
            break;
        case memory::ndhwc:
            layout = Layout::NDHWC;
            order = {0, 2, 3, 4, 1};
            blkDims = {static_cast<size_t>(dims[0]),
                       static_cast<size_t>(dims[2]),
                       static_cast<size_t>(dims[3]),
                       static_cast<size_t>(dims[4]),
                       static_cast<size_t>(dims[1])};
            break;
        case memory::oIhw8i:
        case memory::nChw8c:
            order = {0, 1, 2, 3, 1};
            blkDims = dims;
            blkDims[1] = blkDims[1] / 8 + (blkDims[1] % 8 ? 1 : 0);
            blkDims.push_back(8);
            layout = Layout::BLOCKED;
            break;
        case memory::nCdhw8c:
            order = {0, 1, 2, 3, 4, 1};
            blkDims = dims;
            blkDims[1] = blkDims[1] / 8 + (blkDims[1] % 8 ? 1 : 0);
            blkDims.push_back(8);
            layout = Layout::BLOCKED;
            break;
        case memory::nChw16c:
            order = {0, 1, 2, 3, 1};
            blkDims = dims;
            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
            blkDims.push_back(16);
            layout = Layout::BLOCKED;
            break;
        case memory::nCdhw16c:
            order = {0, 1, 2, 3, 4, 1};
            blkDims = dims;
            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
            blkDims.push_back(16);
            layout = Layout::BLOCKED;
            break;
        case memory::blocked:
            order.clear();
            blkDims = dims;
            for (size_t i = 0; i < blkDims.size(); i++) {
                order.push_back(i);
                if ((i && blkInfo.strides[0][i - 1] < blkInfo.strides[0][i]) || blkInfo.block_dims[i] != 1) {
                    THROW_IE_EXCEPTION << "Cannot cast to tensor desc."
                                       << " Unsupported blocked format.";
                }
            }
            if (order.size() == 3 && order[0] == 0 && order[1] == 1 && order[2] == 2)
                layout = Layout::CHW;
            else
                layout = Layout::BLOCKED;
            break;
        default:
            THROW_IE_EXCEPTION << "Cannot cast to tensor desc. Format is unsupported!";
    }

    SizeVector strides(blkDims.size());

    if (layout == Layout::NHWC || layout == Layout::NDHWC || layout == Layout::CHW) {
        for (size_t i = 0; i < order.size(); i++) {
            strides[i] = static_cast<size_t>(blkInfo.strides[0][order[i]]);
        }
    } else {
        strides[blkDims.size() - 1] = 1;
        for (size_t i = 2; i <= order.size(); i++) {
            if (blkDims.size() - i < dims.size()) {
                strides[blkDims.size() - i] = static_cast<size_t>(blkInfo.strides[0][order[blkDims.size() - i]]);
            } else {
                strides[blkDims.size() - i] = strides[blkDims.size() - i + 1] * blkDims[blkDims.size() - i + 1];
            }
        }
    }

    for (size_t i = 0; i < blkDims.size() && i < TENSOR_MAX_DIMS; i++) {
        if (i < dims.size())
            offsetsForDims.push_back(blkInfo.offset_padding_to_data[i]);
        else
            offsetsForDims.push_back(0);
    }

    TensorDesc tensorDesc(precision, dims, {blkDims, order, offset, offsetsForDims, strides});

    tensorDesc.setLayout(layout);
    return tensorDesc;
}

MKLDNNMemoryDesc::MKLDNNMemoryDesc(const TensorDesc& tDesc):
        desc({}, mkldnn::memory::data_type::f32, mkldnn::memory::format::format_undef) {
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
        default:
            THROW_IE_EXCEPTION << "Cannot create MKLDNNMemoryDesc from TensorDesc. Unsupported precision!";
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
            } else if (realDims.ndims() == 4) {
                if (order.size() == 5 && order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3 && order[4] == 1) {
                    if (blkdDims[4] == 8) {
                        mkldnnFormat = memory::format::nChw8c;
                    } else if (blkdDims[4] == 16) {
                        mkldnnFormat = memory::format::nChw16c;
                    }
                } else if (order.size() == 4) {
                    if (order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3) {
                        mkldnnFormat = memory::format::nchw;
                    } else if (order[0] == 0 && order[1] == 2 && order[2] == 3 && order[3] == 1) {
                        mkldnnFormat = memory::format::nhwc;
                    }
                }
            } else if (realDims.ndims() == 5) {
                if (order.size() == 6 &&
                        order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3 && order[4] == 4 && order[5] == 1) {
                    if (blkdDims[5] == 8) {
                        mkldnnFormat = memory::format::nCdhw8c;
                    } else if (blkdDims[5] == 16) {
                        mkldnnFormat = memory::format::nCdhw16c;
                    }
                } else if (order.size() == 5) {
                    if (order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3 && order[4] == 4) {
                        mkldnnFormat = memory::format::ncdhw;
                    } else if (order[0] == 0 && order[1] == 2 && order[2] == 3 && order[3] == 4 && order[4] == 1) {
                        mkldnnFormat = memory::format::ndhwc;
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
    for (int i = 0; i < desc.data.ndims; i++) {
        if (desc.data.dims[i] != desc.data.layout_desc.blocking.padding_dims[i])
            return true;
    }
    return false;
}

}  // namespace MKLDNNPlugin
