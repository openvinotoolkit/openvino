// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_dump.h"
#include "blob_factory.hpp"
#include "mkldnn_memory.h"

#include "common/memory_desc_wrapper.hpp"

#include <fstream>
#include <cpu_memory_desc_utils.h>

using namespace InferenceEngine;

namespace MKLDNNPlugin {

// IEB file format routine
static unsigned char IEB_MAGIC[4] = {'I', 'E', 'B', '0'};
static unsigned char NO_SCALES = 0xFF;

struct IEB_HEADER {
    unsigned char magic[4];
    unsigned char ver[2];

    unsigned char precision;  // 0-8
    unsigned char ndims;
    unsigned int  dims[7];  // max is 7-D blob

    unsigned char scaling_axis;  // FF - no scaling
    unsigned char reserved[3];

    unsigned long data_offset;
    unsigned long data_size;
    unsigned long scaling_data_offset;
    unsigned long scaling_data_size;
};

static IEB_HEADER prepare_header(const TensorDesc& desc) {
    IEB_HEADER header = {};

    header.magic[0] = IEB_MAGIC[0];
    header.magic[1] = IEB_MAGIC[1];
    header.magic[2] = IEB_MAGIC[2];
    header.magic[3] = IEB_MAGIC[3];

    // IEB file format version 0.1
    header.ver[0] = 0;
    header.ver[1] = 1;

    header.precision = desc.getPrecision();

    if (desc.getDims().size() > 7)
        IE_THROW() << "Dumper support max 7D blobs";

    header.ndims = desc.getDims().size();
    for (int i = 0; i < header.ndims; i++)
        header.dims[i] = desc.getDims()[i];

    header.scaling_axis = NO_SCALES;

    return header;
}

static TensorDesc parse_header(IEB_HEADER &header) {
    if (header.magic[0] != IEB_MAGIC[0] ||
        header.magic[1] != IEB_MAGIC[1] ||
        header.magic[2] != IEB_MAGIC[2] ||
        header.magic[3] != IEB_MAGIC[3])
        IE_THROW() << "Dumper cannot parse file. Wrong format.";

    if (header.ver[0] != 0 ||
        header.ver[1] != 1)
        IE_THROW() << "Dumper cannot parse file. Unsupported IEB format version.";

    Precision prc = Precision(static_cast<Precision::ePrecision>(header.precision));
    SizeVector dims(header.ndims);
    for (int i = 0; i < header.ndims; i++)
        dims[i] = header.dims[i];

    return TensorDesc {prc, dims, TensorDesc::getLayoutByDims(dims) };
}


bool is_plain(const Blob::Ptr &blob) {
    bool res = true;

    auto orig_strides = blob->getTensorDesc().getBlockingDesc().getStrides();
    auto orig_order = blob->getTensorDesc().getBlockingDesc().getOrder();
    auto dims = blob->getTensorDesc().getDims();

    for (int stride = 1, i = dims.size() - 1; i >= 0; --i) {
        if (stride != orig_strides[i] || i != orig_order[i]) res = false;
        stride *= dims[i];
    }

    return res;
}

static Blob::Ptr prepare_plain_data(Blob::Ptr blob) {
    // check if it already plain
    if (is_plain(blob)) return blob;

    Blob::Ptr pln_blob = make_plain_blob(blob->getTensorDesc().getPrecision(), blob->getTensorDesc().getDims());
    pln_blob->allocate();

    // Copy to plain
    MKLDNNMemoryDesc mdesc = MemoryDescUtils::convertToMKLDNNMemoryDesc(blob->getTensorDesc());
    mkldnn::memory::desc desc = mdesc;
    mkldnn::impl::memory_desc_wrapper blob_wrp(desc.data);

    size_t data_size = blob->size();

    // TODO: make it with blob_copy utility
    switch (blob->getTensorDesc().getPrecision()) {
        case Precision::FP32:
        case Precision::I32: {
            auto *pln_blob_ptr = pln_blob->buffer().as<int32_t*>();
            auto *blob_ptr = blob->buffer().as<int32_t*>();
            for (size_t i = 0; i < data_size; i++)
                pln_blob_ptr[i] = blob_ptr[blob_wrp.off_l(i)];
            break;
        }
        case Precision::I16:
        case Precision::U16:
        case Precision::BF16: {
            auto *pln_blob_ptr = pln_blob->buffer().as<int16_t *>();
            auto *blob_ptr = blob->buffer().as<int16_t *>();
            for (size_t i = 0; i < data_size; i++) pln_blob_ptr[i] = blob_ptr[blob_wrp.off_l(i)];
            break;
        }
        case Precision::I8:
        case Precision::U8: {
            auto *pln_blob_ptr = pln_blob->buffer().as<int8_t*>();
            auto *blob_ptr = blob->buffer().as<int8_t *>();
            for (size_t i = 0; i < data_size; i++)
                pln_blob_ptr[i] = blob_ptr[blob_wrp.off_l(i)];
            break;
        }
        default:
            IE_THROW() << "Dumper. Unsupported precision";
    }

    return pln_blob;
}

void BlobDumper::dump(std::ostream &stream) const {
    if (!_blob)
        IE_THROW() << "Dumper cannot dump empty Blob";

    if (_blob->buffer().as<float*>() == nullptr)
        IE_THROW() << "Dumper cannot dump. Blob is not allocated.";

    IEB_HEADER header = prepare_header(_blob->getTensorDesc());
    Blob::Ptr pln_blob = prepare_plain_data(_blob);

    header.data_offset = sizeof(header);
    header.data_size = pln_blob->byteSize();
    header.scaling_data_offset = 0;
    header.scaling_data_size = 0;

    if (_scales) {
        header.scaling_axis = 1;
        header.scaling_data_offset = header.data_offset + header.data_size;
        header.scaling_data_size = _scales->byteSize();
    }

    stream.write(reinterpret_cast<char*>(&header), sizeof(header));
    stream.write(pln_blob->buffer().as<char*>(), pln_blob->byteSize());

    if (_scales) {
        stream.write(_scales->buffer().as<char*>(), _scales->byteSize());
    }
}

void BlobDumper::dumpAsTxt(std::ostream &stream) const {
    if (!_blob)
        IE_THROW() << "Dumper cannot dump empty Blob";

    if (_blob->buffer().as<float*>() == nullptr)
        IE_THROW() << "Dumper cannot dump. Blob is not allocated.";

    SizeVector dims = _blob->getTensorDesc().getDims();

    // Header like "U8 4D shape: 2 3 224 224 ()
    stream << _blob->getTensorDesc().getPrecision().name() << " "
           << dims.size() << "D "
           << "shape: ";
    for (size_t d : dims) stream << d << " ";
    stream << "(" << _blob->size() << ")" <<
    " by address 0x" << std::hex << _blob->buffer().as<long long>() << std::dec <<std::endl;

    // Dump data
    MKLDNNMemoryDesc mdesc = MemoryDescUtils::convertToMKLDNNMemoryDesc(_blob->getTensorDesc());
    mkldnn::memory::desc desc = mdesc;
    mkldnn::impl::memory_desc_wrapper blob_wrp(desc.data);

    size_t data_size = _blob->size();
    switch (_blob->getTensorDesc().getPrecision()) {
        case Precision::FP32: {
            auto *blob_ptr = _blob->buffer().as<float*>();
            for (size_t i = 0; i < data_size; i++)
                stream << blob_ptr[blob_wrp.off_l(i)] << std::endl;
            break;
        }
        case Precision::BF16:
        {
            auto *blob_ptr = _blob->buffer().as<int16_t *>();
            for (size_t i = 0; i < data_size; i++) {
                int i16n = blob_ptr[blob_wrp.off_l(i)];
                i16n = i16n << 16;
                float fn = *(reinterpret_cast<float *>(&i16n));
                stream << fn << std::endl;
            }
            break;
        }
        case Precision::I32: {
            auto *blob_ptr = _blob->buffer().as<int32_t*>();
            for (size_t i = 0; i < data_size; i++)
                stream << blob_ptr[blob_wrp.off_l(i)] << std::endl;
            break;
        }
        case Precision::I16: {
            auto *blob_ptr = _blob->buffer().as<int16_t*>();
            for (size_t i = 0; i < data_size; i++)
                stream << static_cast<int>(blob_ptr[blob_wrp.off_l(i)]) << std::endl;
            break;
        }
        case Precision::U16: {
            auto *blob_ptr = _blob->buffer().as<uint16_t*>();
            for (size_t i = 0; i < data_size; i++)
                stream << static_cast<int>(blob_ptr[blob_wrp.off_l(i)]) << std::endl;
            break;
        }
        case Precision::I8: {
            auto *blob_ptr = _blob->buffer().as<int8_t*>();
            for (size_t i = 0; i < data_size; i++)
                stream << static_cast<int>(blob_ptr[blob_wrp.off_l(i)]) << std::endl;
            break;
        }
        case Precision::U8: {
            auto *blob_ptr = _blob->buffer().as<uint8_t*>();
            for (size_t i = 0; i < data_size; i++)
                stream << static_cast<int>(blob_ptr[blob_wrp.off_l(i)]) << std::endl;
            break;
        }
        default:
            IE_THROW() << "Dumper. Unsupported precision";
    }
}

BlobDumper BlobDumper::read(std::istream &stream) {
    IEB_HEADER header;
    stream.read(reinterpret_cast<char*>(&header), sizeof(header));

    TensorDesc desc = parse_header(header);
    Blob::Ptr blob = make_blob_with_precision(desc);
    blob->allocate();

    stream.seekg(header.data_offset, stream.beg);
    stream.read(blob->buffer().as<char*>(), header.data_size);

    BlobDumper res(blob);

    // Parse scales fields.
    if (header.scaling_axis != NO_SCALES) {
        if (header.scaling_axis != 1)
            IE_THROW() << "Dumper support scaling only for channel dims.";

        size_t scl_size = header.scaling_data_size / sizeof(float);
        auto scl = make_blob_with_precision({Precision::FP32, {scl_size}, C});
        scl->allocate();

        stream.seekg(header.scaling_data_offset, stream.beg);
        stream.read(scl->buffer().as<char*>(), header.scaling_data_size);

        res._scales = scl;
    }
    return res;
}

BlobDumper BlobDumper::read(const std::string &file_path) {
    std::ifstream file;
    file.open(file_path);
    if (!file.is_open())
        IE_THROW() << "Dumper cannot open file " << file_path;

    auto res = read(file);
    file.close();
    return res;
}

void BlobDumper::dump(const std::string &dump_path) const {
    std::ofstream dump_file;
    dump_file.open(dump_path);
    if (!dump_file.is_open())
        IE_THROW() << "Dumper cannot create dump file";

    dump(dump_file);
    dump_file.close();
}

void BlobDumper::dumpAsTxt(const std::string& dump_path) const {
    std::ofstream dump_file;
    dump_file.open(dump_path);
    if (!dump_file.is_open())
        IE_THROW() << "Dumper cannot create dump file";

    dumpAsTxt(dump_file);
    dump_file.close();
}

Blob::Ptr BlobDumper::get() {
    return _blob;
}

template <typename data_t>
static void plain_copy(const Blob::Ptr &from, const Blob::Ptr &scls, Blob::Ptr &to) {
    auto dims = from->getTensorDesc().getDims();

    size_t data_size = from->size();
    size_t outer_size = dims[0];
    size_t c_size = dims.size() > 1 ? dims[1] : 1;
    size_t inner_size = dims.size() == 4 ? dims[2]*dims[3] :
                        dims.size() == 3 ? dims[2] : 1;

    auto to_data  = to->buffer().as<float*>();
    auto from_data = from->buffer().as<data_t*>();

    if (scls) {
        auto scls_data = scls->buffer().as<float*>();

        for (size_t o=0; o < outer_size; o++)
        for (size_t c=0; c < c_size; c++)
        for (size_t i=0; i < inner_size; i++)
            *to_data++ = static_cast<float>(*from_data++) * scls_data[c];
    } else {
        for (size_t i=0; i < data_size; i++)
            *to_data++ = static_cast<float>(*from_data++);
    }
}

Blob::Ptr BlobDumper::getRealValue() {
    if (_blob->getTensorDesc().getPrecision() == Precision::FP32 && !_scales)
        return _blob;

    auto res = make_plain_blob(Precision::FP32, _blob->getTensorDesc().getDims());
    res->allocate();

    switch (_blob->getTensorDesc().getPrecision()) {
        case Precision::U8: plain_copy<uint8_t>(_blob, _scales, res); break;
        case Precision::FP32: plain_copy<float>(_blob, _scales, res); break;
        case Precision::I8: plain_copy<int8_t >(_blob, _scales, res); break;
        default: IE_THROW() << "Unsupported precesion for getRealValue method.";
    }

    return res;
}


BlobDumper& BlobDumper::withScales(InferenceEngine::Blob::Ptr scales) {
    if ( _blob->getTensorDesc().getDims().size() < 2  ||
        scales->getTensorDesc().getDims().size() != 1 ||
        scales->getTensorDesc().getDims()[0] != _blob->getTensorDesc().getDims()[1] ||
        scales->getTensorDesc().getPrecision() != Precision::FP32)
        IE_THROW() << "Dumper cannot use passed scales. Blob has incompatible shape.";

    _scales = scales;
    return *this;
}

BlobDumper& BlobDumper::withoutScales() {
    _scales.reset();
    return *this;
}


const InferenceEngine::Blob::Ptr& BlobDumper::getScales() const {
    return _scales;
}

}  // namespace MKLDNNPlugin
