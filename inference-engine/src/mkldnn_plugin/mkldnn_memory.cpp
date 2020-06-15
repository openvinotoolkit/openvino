// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <utility>

#include <mkldnn_types.h>
#include <mkldnn_debug.h>
#include "mkldnn_memory.h"
#include "mkldnn_node.h"
#include "mkldnn_extension_utils.h"

using namespace InferenceEngine;
using namespace mkldnn;

namespace MKLDNNPlugin {

MKLDNNMemory::MKLDNNMemory(const engine& eng) : eng(eng) {}

size_t MKLDNNMemory::GetSize() const {
    uint8_t itemSize = MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(GetDataType()));
    return GetElementsCount() * itemSize;
}

size_t MKLDNNMemory::GetElementsCount() const {
    auto desc = GetDescriptor();
    std::vector<int> dims(desc.data.layout_desc.blocking.padding_dims,
                          desc.data.layout_desc.blocking.padding_dims + desc.data.ndims);
    return std::accumulate(std::begin(dims), std::end(dims), (size_t) 1, std::multiplies<size_t>());
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

void MKLDNNMemory::Create(const mkldnn::memory::desc& desc, const void *data, bool pads_zeroing) {
    auto primitive_desc = memory::primitive_desc(desc, eng);

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
    } else {
        // MKLDNN accepts not a const data, probably need to remove some level of consteness in a call stack

        // ========================
        // Equivalent of constructor memory(const primitive_desc &desc, void *hdl)
        // but with ability to skipp pads zeroing.
        mkldnn_primitive_t result;
        error::wrap_c_api(mkldnn_primitive_create(&result, primitive_desc.get(), nullptr, nullptr),
                "could not create a memory primitive");
        auto *mem = new memory(nullptr);
        mem->reset(result);
        if (pads_zeroing)
            mem->set_data_handle(const_cast<void*>(data));
        else
            mem->set_data_handle_no_pads_proc(const_cast<void*>(data));
        //
        // ========================

        prim.reset(mem);
    }
}

void MKLDNNMemory::SetData(memory::data_type dataType, memory::format format, const void* data, size_t size, bool ftz) const {
    uint8_t itemSize = MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(dataType));

    if (static_cast<mkldnn_memory_format_t>(format) != GetDescriptor().data.format ||
            GetDataType() != dataType) {
        auto memData = GetDescriptor().data;

        std::vector<ptrdiff_t> dims(memData.dims, memData.dims + memData.ndims);

        auto data_type = GetDataType();

        MKLDNNMemory src(eng);
        src.Create(dims, data_type, format, data);

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

    if (ftz && memory.GetDataType() == mkldnn::memory::f32 && GetFormat() != mkldnn::memory::wino_fmt &&
        GetDataType() != mkldnn::memory::bf16) {
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
        case f::ohwi:
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
        case f::OhIw8o4i:
        case f::IOhw16o16i:
            ndims = 4; break;
        // DHW
        case f::ncdhw:
        case f::ndhwc:
        case f::nCdhw8c:
        case f::nCdhw16c:
        case f::oidhw:
        case f::odhwi:
        case f::OIdhw8i8o:
        case f::OIdhw16i16o:
        case f::OIdhw8o8i:
        case f::OIdhw16o16i:
        case f::OIdhw8i16o2i:
        case f::OIdhw4i16o4i:
        case f::Odhwi8o:
        case f::Odhwi16o:
        case f::OdhIw8o4i:
        // Group HW
        case f::hwigo:
        case f::goihw:
        case f::gOIhw8i8o:
        case f::gOIhw16i16o:
        case f::gOIhw8i16o2i:
        case f::gOIhw8o16i2o:
        case f::gOhwi8o:
        case f::gOhwi16o:
        case f::gOIhw4o4i:
        case f::gOIhw8o8i:
        case f::gOIhw16o16i:
        case f::gOhIw8o4i:
        case f::gOhIw16o4i:
        case f::gOIhw2i8o4i:
        case f::gOIhw4i16o4i:
        case f::Goihw8g:
        case f::Goihw16g:
        case f::dhwio:
            ndims = 5; break;
        case f::goidhw:
        case f::gOIdhw4i4o:
        case f::gOIdhw8i8o:
        case f::gOIdhw16i16o:
        case f::gOIdhw8i16o2i:
        case f::gOdhwi8o:
        case f::gOdhwi16o:
        case f::gOIdhw8o8i:
        case f::gOdhIw8o4i:
        case f::gOIdhw16o16i:
        case f::gOIdhw4i16o4i:
        case f::Goidhw8g:
        case f::Goidhw16g:
        case f::dhwigo:
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
    std::vector<memory::format> plains = {
    /* 1D */  memory::x,
    /* 2D */  memory::nc, memory::oi, memory::io,
    /* 3D */  memory::tnc, memory::ntc, memory::oiw, memory::wio,
    /* 4D */  memory::nchw, memory::nhwc, memory::chwn, memory::ihwo, memory::oihw, memory::hwio,
    /* 5D */  memory::ncdhw, memory::ndhwc, memory::oidhw, memory::goihw, memory::dhwio,
    /* 6D */  memory::goidhw, memory::dhwigo,
              memory::blocked};

    for (auto it : plains) {
        if (format == it) {
            return true;
        }
    }

    return false;
}

bool MKLDNNMemory::IsGroupedFormat(memory::format format) {
    using f = mkldnn::memory::format;

    std::vector<memory::format> groupedFormats = {f::hwigo, f::goihw, f::gOIhw8i8o, f::gOIhw16i16o, f::gOIhw8i16o2i,
            f::gOIhw8o16i2o, f::gOhwi8o, f::gOhwi16o, f::gOIhw8o8i, f::gOIhw16o16i, f::gOhIw8o4i, f::gOhIw16o4i, f::Goihw8g, f::Goihw16g,
            f::goidhw, f::gOIdhw4i4o, f::gOIdhw8i8o, f::gOIdhw16i16o, f::gOIdhw8i16o2i, f::gOdhwi8o, f::gOdhwi16o, f::gOIdhw8o8i, f::gOIdhw16o16i,
            f::gOIhw4i16o4i, f::dhwigo, f::gOIhw2i8o4i, f::gOIhw4o4i, f::Goidhw8g, f::Goidhw16g, f::gOIdhw4i16o4i, f::gOdhIw8o4i};

    for (auto it : groupedFormats) {
        if (format == it) {
            return true;
        }
    }

    return false;
}

memory::format MKLDNNMemory::GetPlainFormat(memory::dims dims) {
    switch (dims.size()) {
        case 0:
            return memory::x;
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
        case 5: return Layout::NCDHW;
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
        case SCALAR:
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
        case memory::hwio: return "hwio";
        case memory::ohwi: return "ohwi";
        case memory::OIhw8i8o: return "OIhw8i8o";
        case memory::OIhw16i16o: return "OIhw16i16o";
        case memory::OIhw8o8i: return "OIhw8o8i";
        case memory::OIhw16o16i: return "OIhw16o16i";
        case memory::OIhw8i16o2i: return "OIhw8i16o2i";
        case memory::OIhw8o16i2o: return "OIhw8o16i2o";
        case memory::Ohwi8o: return "Ohwi8o";
        case memory::Ohwi16o: return "Ohwi16o";
        case memory::OhIw8o4i: return "OhIw8o4i";
        case memory::OhIw16o4i: return "OhIw16o4i";
        case memory::OIhw4i16o4i: return "OIhw4i16o4i";
        case memory::IOhw16o16i: return "IOhw16o16i";

        case memory::oidhw: return "oidhw";
        case memory::dhwio: return "dhwio";
        case memory::odhwi: return "dhwio";
        case memory::OIdhw8i8o: return "OIdhw8i8o";
        case memory::OIdhw16i16o: return "OIdhw16i16o";
        case memory::OIdhw8o8i: return "OIdhw8o8i";
        case memory::OIdhw16o16i: return "OIdhw16o16i";
        case memory::OIdhw8i16o2i: return "OIdhw8i16o2i";
        case memory::Odhwi8o: return "Odhwi8o";
        case memory::Odhwi16o: return "Odhwi16o";
        case memory::OdhIw8o4i: return "OdhIw8o4i";
        case memory::OIdhw4i16o4i: return "OIdhw4i16o4i";

        case memory::goihw: return "goihw";
        case memory::hwigo: return "hwigo";
        case memory::dhwigo: return "dhwigo";
        case memory::gOIhw8i8o: return "gOIhw8i8o";
        case memory::gOIhw16i16o: return "gOIhw16i16o";
        case memory::gOIhw8i16o2i: return "gOIhw8i16o2i";
        case memory::gOIhw8o16i2o: return "gOIhw8o16i2o";
        case memory::gOhwi8o: return "gOhwi8o";
        case memory::gOhwi16o: return "gOhwi16o";
        case memory::gOIhw4o4i: return "gOIhw4o4i";
        case memory::gOIhw8o8i: return "gOIhw8o8i";
        case memory::gOIhw16o16i: return "gOIhw16o16i";
        case memory::gOhIw16o4i: return "gOhIw16o4i";
        case memory::gOhIw8o4i: return "gOhIw8o4i";
        case memory::gOIhw4i16o4i: return "gOIhw4i16o4i";
        case memory::gOIhw2i8o4i: return "gOIhw2i8o4i";

        case memory::goidhw: return "goidhw";
        case memory::gOIdhw4i4o: return "gOIdhw4i4o";
        case memory::gOIdhw8i8o: return "gOIdhw8i8o";
        case memory::gOIdhw16i16o: return "gOIdhw16i16o";
        case memory::gOIdhw8i16o2i: return "gOIdhw8i16o2i";
        case memory::gOdhwi8o: return "gOdhwi8o";
        case memory::gOdhwi16o: return "gOdhwi16o";
        case memory::gOIdhw8o8i: return "gOIdhw8o8i";
        case memory::gOIdhw16o16i: return "gOIdhw16o16i";
        case memory::gOIdhw4i16o4i: return "gOIdhw4i16o4i";
        case memory::gOdhIw8o4i: return "gOdhIw8o4i";

        case memory::Goihw8g: return "Goihw8g";
        case memory::Goihw16g: return "Goihw16g";
        case memory::Goidhw8g: return "Goidhw8g";
        case memory::Goidhw16g: return "Goidhw16g";

        default: {
            THROW_IE_EXCEPTION << "Unknown data format.";
        }
    }
}

bool MKLDNNMemoryDesc::operator==(const MKLDNNMemoryDesc &rhs) const {
    auto dims_equal = [] (const mkldnn_memory_desc_t &ldata, const mkldnn_memory_desc_t &rdata) {
        if (ldata.ndims != rdata.ndims)
            return false;
        for (int i = 0; i < ldata.ndims; i++) {
            if (ldata.dims[i] != rdata.dims[i])
                return false;
        }
        return true;
    };
    auto blocking_equal = [] (const mkldnn_memory_desc_t &ldata, const mkldnn_memory_desc_t &rdata) {
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
        if (format == memory::x && dims.size() == 0) {
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
        case mkldnn_bf16:
            precision = Precision::BF16;
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
        case memory::hwio:
            layout = Layout::BLOCKED;
            order = {2, 3, 1, 0};
            blkDims = dims;
            break;
        case memory::dhwio:
            layout = Layout::BLOCKED;
            order = {2, 3, 4, 1, 0};
            blkDims = dims;
            break;
        case memory::hwigo:
            layout = Layout::BLOCKED;
            order = {3, 4, 2, 0, 1};
            blkDims = dims;
            break;
        case memory::dhwigo:
            order = {3, 4, 5, 2, 0, 1};
            blkDims = dims;
            layout = Layout::BLOCKED;
            break;
        case memory::oidhw:
        case memory::ncdhw:
            layout = Layout::NCDHW;
            order = {0, 1, 2, 3, 4};
            blkDims = dims;
            break;
        case memory::ohwi:
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
        case memory::odhwi:
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
        case memory::gOhwi8o:
        case memory::nCdhw8c:
            order = {0, 1, 2, 3, 4, 1};
            blkDims = dims;
            blkDims[1] = blkDims[1] / 8 + (blkDims[1] % 8 ? 1 : 0);
            blkDims.push_back(8);
            layout = Layout::BLOCKED;
            break;
        case memory::gOdhwi8o:
            order = {0, 1, 2, 3, 4, 5, 1};
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
        case memory::gOhwi16o:
        case memory::nCdhw16c:
            order = {0, 1, 2, 3, 4, 1};
            blkDims = dims;
            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
            blkDims.push_back(16);
            layout = Layout::BLOCKED;
            break;
        case memory::gOdhwi16o:
            order = {0, 1, 2, 3, 4, 5, 1};
            blkDims = dims;
            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
            blkDims.push_back(16);
            layout = Layout::BLOCKED;
            break;
        case memory::Ohwi8o:
            order = {0, 1, 2, 3, 0};
            blkDims = dims;
            blkDims[0] = blkDims[0] / 8 + (blkDims[0] % 8 ? 1 : 0);
            blkDims.push_back(8);
            layout = Layout::BLOCKED;
            break;
        case memory::Ohwi16o:
            order = {0, 1, 2, 3, 0};
            blkDims = dims;
            blkDims[0] = blkDims[0] / 16 + (blkDims[0] % 16 ? 1 : 0);
            blkDims.push_back(16);
            layout = Layout::BLOCKED;
            break;
        case memory::Odhwi8o:
            order = {0, 2, 3, 4, 1, 0};
            blkDims = dims;
            blkDims[0] = blkDims[0] / 8 + (blkDims[0] % 8 ? 1 : 0);
            blkDims.push_back(8);
            layout = Layout::BLOCKED;
            break;
        case memory::Odhwi16o:
            order = {0, 2, 3, 4, 1, 0};
            blkDims = dims;
            blkDims[0] = blkDims[0] / 16 + (blkDims[0] % 16 ? 1 : 0);
            blkDims.push_back(16);
            layout = Layout::BLOCKED;
            break;
        case memory::OIhw8i8o:
            order = {0, 1, 2, 3, 1, 0};
            blkDims = dims;
            blkDims[0] = blkDims[0] / 8 + (blkDims[0] % 8 ? 1 : 0);
            blkDims[1] = blkDims[1] / 8 + (blkDims[1] % 8 ? 1 : 0);
            blkDims.push_back(8);
            blkDims.push_back(8);
            layout = Layout::BLOCKED;
            break;
        case memory::OIhw16i16o:
            order = {0, 1, 2, 3, 1, 0};
            blkDims = dims;
            blkDims[0] = blkDims[0] / 16 + (blkDims[0] % 16 ? 1 : 0);
            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
            blkDims.push_back(16);
            blkDims.push_back(16);
            layout = Layout::BLOCKED;
            break;
        case memory::OIhw8o8i:
            order = {0, 1, 2, 3, 0, 1};
            blkDims = dims;
            blkDims[0] = blkDims[0] / 8 + (blkDims[0] % 8 ? 1 : 0);
            blkDims[1] = blkDims[1] / 8 + (blkDims[1] % 8 ? 1 : 0);
            blkDims.push_back(8);
            blkDims.push_back(8);
            layout = Layout::BLOCKED;
            break;
        case memory::OIhw16o16i:
            order = {0, 1, 2, 3, 0, 1};
            blkDims = dims;
            blkDims[0] = blkDims[0] / 16 + (blkDims[0] % 16 ? 1 : 0);
            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
            blkDims.push_back(16);
            blkDims.push_back(16);
            layout = Layout::BLOCKED;
            break;
        case memory::IOhw16o16i:
            order = {1, 0, 2, 3, 0, 1};
            blkDims = dims;
            blkDims[0] = blkDims[0] / 16 + (blkDims[0] % 16 ? 1 : 0);
            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
            blkDims.push_back(16);
            blkDims.push_back(16);
            layout = Layout::BLOCKED;
            break;
        case memory::OIdhw8i8o:
            order = {0, 1, 2, 3, 4, 1, 0};
            blkDims = dims;
            blkDims[0] = blkDims[0] / 8 + (blkDims[0] % 8 ? 1 : 0);
            blkDims[1] = blkDims[1] / 8 + (blkDims[1] % 8 ? 1 : 0);
            blkDims.push_back(8);
            blkDims.push_back(8);
            layout = Layout::BLOCKED;
            break;
        case memory::OIdhw16i16o:
            order = {0, 1, 2, 3, 4, 1, 0};
            blkDims = dims;
            blkDims[0] = blkDims[0] / 16 + (blkDims[0] % 16 ? 1 : 0);
            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
            blkDims.push_back(16);
            blkDims.push_back(16);
            layout = Layout::BLOCKED;
            break;
        case memory::OIdhw8o8i:
            order = {0, 1, 2, 3, 4, 1, 0};
            blkDims = dims;
            blkDims[0] = blkDims[0] / 8 + (blkDims[0] % 8 ? 1 : 0);
            blkDims[1] = blkDims[1] / 8 + (blkDims[1] % 8 ? 1 : 0);
            blkDims.push_back(8);
            blkDims.push_back(8);
            layout = Layout::BLOCKED;
            break;
        case memory::OIdhw16o16i:
            order = {0, 1, 2, 3, 4, 0, 1};
            blkDims = dims;
            blkDims[0] = blkDims[0] / 16 + (blkDims[0] % 16 ? 1 : 0);
            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
            blkDims.push_back(16);
            blkDims.push_back(16);
            layout = Layout::BLOCKED;
            break;
        case memory::gOIhw4o4i:
            order = {0, 1, 2, 3, 4, 1, 2};
            blkDims = dims;
            blkDims[1] = blkDims[1] / 4 + (blkDims[1] % 4 ? 1 : 0);
            blkDims[2] = blkDims[2] / 4 + (blkDims[2] % 4 ? 1 : 0);
            blkDims.push_back(4);
            blkDims.push_back(4);
            layout = Layout::BLOCKED;
            break;
        case memory::gOIhw8i8o:
            order = {0, 1, 2, 3, 4, 2, 1};
            blkDims = dims;
            blkDims[1] = blkDims[1] / 8 + (blkDims[1] % 8 ? 1 : 0);
            blkDims[2] = blkDims[2] / 8 + (blkDims[2] % 8 ? 1 : 0);
            blkDims.push_back(8);
            blkDims.push_back(8);
            layout = Layout::BLOCKED;
            break;
        case memory::gOIhw8o8i:
            order = {0, 1, 2, 3, 4, 1, 2};
            blkDims = dims;
            blkDims[1] = blkDims[1] / 8 + (blkDims[1] % 8 ? 1 : 0);
            blkDims[2] = blkDims[2] / 8 + (blkDims[2] % 8 ? 1 : 0);
            blkDims.push_back(8);
            blkDims.push_back(8);
            layout = Layout::BLOCKED;
            break;
        case memory::gOIhw16i16o:
            order = {0, 1, 2, 3, 4, 2, 1};
            blkDims = dims;
            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
            blkDims[2] = blkDims[2] / 16 + (blkDims[2] % 16 ? 1 : 0);
            blkDims.push_back(16);
            blkDims.push_back(16);
            layout = Layout::BLOCKED;
            break;
        case memory::gOIhw16o16i:
            order = {0, 1, 2, 3, 4, 1, 2};
            blkDims = dims;
            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
            blkDims[2] = blkDims[2] / 16 + (blkDims[2] % 16 ? 1 : 0);
            blkDims.push_back(16);
            blkDims.push_back(16);
            layout = Layout::BLOCKED;
            break;
        case memory::OhIw8o4i:
            order = {0, 2, 1, 3, 0, 1};
            blkDims = dims;
            blkDims[0] = blkDims[0] / 8 + (blkDims[0] % 8 ? 1 : 0);
            blkDims[1] = blkDims[1] / 4 + (blkDims[1] % 4 ? 1 : 0);
            blkDims.push_back(8);
            blkDims.push_back(4);
            layout = Layout::BLOCKED;
            break;
        case memory::OIhw4i16o4i:
            order = {0, 1, 2, 3, 1, 0, 1};
            blkDims = dims;
            blkDims[0] = blkDims[0] / 16 + (blkDims[0] % 16 ? 1 : 0);
            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
            blkDims.push_back(4);
            blkDims.push_back(16);
            blkDims.push_back(4);
            layout = Layout::BLOCKED;
            break;
        case memory::gOIhw2i8o4i:
            order = {0, 1, 2, 3, 4, 2, 1, 2};
            blkDims = dims;
            blkDims[1] = blkDims[1] / 8 + (blkDims[1] % 8 ? 1 : 0);
            blkDims[2] = blkDims[2] / 8 + (blkDims[2] % 8 ? 1 : 0);
            blkDims.push_back(2);
            blkDims.push_back(8);
            blkDims.push_back(4);
            layout = Layout::BLOCKED;
            break;
        case memory::gOIhw4i16o4i:
            order = {0, 1, 2, 3, 4, 2, 1, 2};
            blkDims = dims;
            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
            blkDims[2] = blkDims[2] / 16 + (blkDims[2] % 16 ? 1 : 0);
            blkDims.push_back(4);
            blkDims.push_back(16);
            blkDims.push_back(4);
            layout = Layout::BLOCKED;
            break;
        case memory::OdhIw8o4i:
            order = {0, 2, 3, 1, 4, 0, 1};
            blkDims = dims;
            blkDims[0] = blkDims[0] / 8 + (blkDims[0] % 8 ? 1 : 0);
            blkDims[1] = blkDims[1] / 4 + (blkDims[1] % 4 ? 1 : 0);
            blkDims.push_back(8);
            blkDims.push_back(4);
            layout = Layout::BLOCKED;
            break;
        case memory::OIdhw4i16o4i:
            order = {0, 1, 2, 3, 4, 1, 0, 1};
            blkDims = dims;
            blkDims[0] = blkDims[0] / 16 + (blkDims[0] % 16 ? 1 : 0);
            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
            blkDims.push_back(4);
            blkDims.push_back(16);
            blkDims.push_back(4);
            layout = Layout::BLOCKED;
            break;
        case memory::gOhIw8o4i:
            order = {0, 1, 3, 2, 4, 1, 2};
            blkDims = dims;
            blkDims[1] = blkDims[1] / 8 + (blkDims[1] % 8 ? 1 : 0);
            blkDims[2] = blkDims[2] / 4 + (blkDims[2] % 4 ? 1 : 0);
            blkDims.push_back(8);
            blkDims.push_back(4);
            layout = Layout::BLOCKED;
            break;
        case memory::gOdhIw8o4i:
            order = {0, 1, 3, 4, 2, 5, 1, 2};
            blkDims = dims;
            blkDims[1] = blkDims[1] / 8 + (blkDims[1] % 8 ? 1 : 0);
            blkDims[2] = blkDims[2] / 4 + (blkDims[2] % 4 ? 1 : 0);
            blkDims.push_back(8);
            blkDims.push_back(4);
            layout = Layout::BLOCKED;
            break;
        case memory::gOIdhw4i16o4i:
            order = {0, 1, 2, 3, 4, 5, 2, 1, 2};
            blkDims = dims;
            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
            blkDims[2] = blkDims[2] / 16 + (blkDims[2] % 16 ? 1 : 0);
            blkDims.push_back(4);
            blkDims.push_back(16);
            blkDims.push_back(4);
            layout = Layout::BLOCKED;
            break;
        case memory::gOIdhw4i4o:
            order = {0, 1, 2, 3, 4, 5, 2, 1};
            blkDims = dims;
            blkDims[1] = blkDims[1] / 4 + (blkDims[1] % 4 ? 1 : 0);
            blkDims[2] = blkDims[2] / 4 + (blkDims[2] % 4 ? 1 : 0);
            blkDims.push_back(4);
            blkDims.push_back(4);
            layout = Layout::BLOCKED;
            break;
        case memory::gOIdhw8i8o:
            order = {0, 1, 2, 3, 4, 5, 2, 1};
            blkDims = dims;
            blkDims[1] = blkDims[1] / 8 + (blkDims[1] % 8 ? 1 : 0);
            blkDims[2] = blkDims[2] / 8 + (blkDims[2] % 8 ? 1 : 0);
            blkDims.push_back(8);
            blkDims.push_back(8);
            layout = Layout::BLOCKED;
            break;
        case memory::gOIdhw16i16o:
            order = {0, 1, 2, 3, 4, 5, 2, 1};
            blkDims = dims;
            blkDims[1] = blkDims[1] / 16 + (blkDims[1] % 16 ? 1 : 0);
            blkDims[2] = blkDims[2] / 16 + (blkDims[2] % 16 ? 1 : 0);
            blkDims.push_back(16);
            blkDims.push_back(16);
            layout = Layout::BLOCKED;
            break;
        case memory::goihw:
            order = {0, 1, 2, 3, 4};
            blkDims = dims;
            layout = Layout::GOIHW;
            break;
        case memory::goidhw:
            order = {0, 1, 2, 3, 4, 5};
            blkDims = dims;
            layout = Layout::GOIDHW;
            break;
        case memory::Goihw8g:
            order = {0, 1, 2, 3, 4, 0};
            blkDims = dims;
            blkDims[0] = blkDims[0] / 8 + (blkDims[0] % 8 ? 1 : 0);
            blkDims.push_back(8);
            layout = Layout::BLOCKED;
            break;
        case memory::Goihw16g:
            order = {0, 1, 2, 3, 4, 0};
            blkDims = dims;
            blkDims[0] = blkDims[0] / 16 + (blkDims[0] % 16 ? 1 : 0);
            blkDims.push_back(16);
            layout = Layout::BLOCKED;
            break;
        case memory::Goidhw8g:
            order = {0, 1, 2, 3, 4, 5, 0};
            blkDims = dims;
            blkDims[0] = blkDims[0] / 8 + (blkDims[0] % 8 ? 1 : 0);
            blkDims.push_back(8);
            layout = Layout::BLOCKED;
            break;
        case memory::Goidhw16g:
            order = {0, 1, 2, 3, 4, 5, 0};
            blkDims = dims;
            blkDims[0] = blkDims[0] / 16 + (blkDims[0] % 16 ? 1 : 0);
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
        case Precision::BOOL:
            data_type = mkldnn::memory::data_type::u8;
            break;
        case Precision::BF16:
            data_type = mkldnn::memory::data_type::bf16;
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
    for (int i = 0; i < desc.data.ndims; i++) {
        if (desc.data.dims[i] != desc.data.layout_desc.blocking.padding_dims[i])
            return true;
    }
    return false;
}

mkldnn::memory::format get_format_tag(const InferenceEngine::TensorDesc &tdesc) {
    MKLDNNMemoryDesc mkldnn_tdesc(tdesc);
    return mkldnn_tdesc.getFormat();
}

std::string format_tag_to_string(mkldnn::memory::format tag) {
    return mkldnn_fmt2str(static_cast<mkldnn_memory_format_t>(tag));
}

MKLDNNMemoryDesc MKLDNNMemoryDesc::create_uninit_version() const {
    mkldnn::memory::desc new_desc(this->desc);
    if (this->isDefined()) {
        for (auto &s : new_desc.data.layout_desc.blocking.strides[0])
            s = std::numeric_limits<ptrdiff_t>::max();

        for (auto &s : new_desc.data.layout_desc.blocking.strides[1])
            s = std::numeric_limits<ptrdiff_t>::max();

        for (auto &d : new_desc.data.layout_desc.blocking.offset_padding_to_data)
            d = 0;

        new_desc.data.layout_desc.blocking.offset_padding = 0;
    }

    return MKLDNNMemoryDesc(new_desc);
}

bool MKLDNNMemoryDesc::isUninit() const {
    if (getFormat() == mkldnn::memory::any)
        return true;

    if (desc.data.layout_desc.blocking.offset_padding == std::numeric_limits<size_t>::max())
        return true;

    for (auto &s : desc.data.layout_desc.blocking.strides[0])
        if (s == std::numeric_limits<ptrdiff_t>::max())
            return true;

    for (auto &s : desc.data.layout_desc.blocking.strides[1])
        if (s == std::numeric_limits<ptrdiff_t>::max())
            return true;

    for (auto &s : desc.data.layout_desc.blocking.offset_padding_to_data)
        // TODO: max or zero? The previous code in create_uninit_version
        //       sets it to zero...
        if (s == std::numeric_limits<ptrdiff_t>::max())
            return true;

    return false;
}

InferenceEngine::Precision MKLDNNMemoryDesc::getPrecision() const {
    return MKLDNNExtensionUtils::DataTypeToIEPrecision(getDataType());
}

bool MKLDNNExtensionUtils::initTensorsAreEqual(const InferenceEngine::TensorDesc &desc1, const InferenceEngine::TensorDesc &desc2) {
    if (desc1.getDims() != desc2.getDims() || desc1.getPrecision() != desc2.getPrecision())
        return false;
    if (desc1.getLayout() == InferenceEngine::Layout::SCALAR && desc2.getLayout() == InferenceEngine::Layout::SCALAR)
        return true;
    if (desc1.getLayout() == InferenceEngine::Layout::ANY || desc2.getLayout() == InferenceEngine::Layout::ANY)
        return true;
    bool batch1 = desc1.getDims()[0] == 1;
    const auto& in1Block = desc1.getBlockingDesc();
    const auto& in2Block = desc2.getBlockingDesc();
    size_t uninitNum = std::numeric_limits<size_t>::max();
    if (in1Block.getBlockDims().size() != in2Block.getBlockDims().size())
        return false;
    for (size_t i = 0; i < in1Block.getBlockDims().size(); i++) {
        if (in1Block.getBlockDims()[i] != in2Block.getBlockDims()[i] &&
            in1Block.getBlockDims()[i] != uninitNum && in2Block.getBlockDims()[i] != uninitNum)
            return false;
        if (in1Block.getOffsetPaddingToData()[i] != in2Block.getOffsetPaddingToData()[i] &&
            in1Block.getOffsetPaddingToData()[i] != uninitNum && in2Block.getOffsetPaddingToData()[i] != uninitNum)
            return false;
        if (i >= batch1 && in1Block.getStrides()[i] != in2Block.getStrides()[i] &&
            in1Block.getStrides()[i] != uninitNum && in2Block.getStrides()[i] != uninitNum)
            return false;
        if (in1Block.getOrder()[i] != in2Block.getOrder()[i] &&
            in1Block.getOrder()[i] != uninitNum && in2Block.getOrder()[i] != uninitNum)
            return false;
    }
    return !(in1Block.getOffsetPadding() != in2Block.getOffsetPadding() &&
             in1Block.getOffsetPadding() != uninitNum && in2Block.getOffsetPadding() != uninitNum);
}


    /*
     * @param blocked_dims blocked dimensions
     * @param order the order of dimensions
     * @param offset offset to the current memory block
     * @param dimOffsets per-dimension offset from the padding to actual data,
     * @param strides strides for each dimension
     */


}  // namespace MKLDNNPlugin
