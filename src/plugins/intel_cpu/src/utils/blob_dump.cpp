// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_dump.h"

#include <cpu_memory.h>
#include <memory_desc/cpu_memory_desc_utils.h>
#include <nodes/common/cpu_memcpy.h>

#include <fstream>

#include "common/memory_desc_wrapper.hpp"
#include "dnnl_extension_utils.h"
#include "utils/bfloat16.hpp"

namespace ov::intel_cpu {

// IEB file format routine
static const unsigned char IEB_MAGIC[4] = {'I', 'E', 'B', '0'};
static const unsigned char NO_SCALES = 0xFF;

struct IEB_HEADER {
    unsigned char magic[4];
    unsigned char ver[2];

    unsigned char precision;  // 0-8
    unsigned char ndims;
    unsigned int dims[7];  // max is 7-D blob

    unsigned char scaling_axis;  // FF - no scaling
    unsigned char reserved[3];

    uint64_t data_offset;
    uint64_t data_size;
    uint64_t scaling_data_offset;
    uint64_t scaling_data_size;
};

static IEB_HEADER prepare_header(const MemoryDesc& desc) {
    IEB_HEADER header = {};

    header.magic[0] = IEB_MAGIC[0];
    header.magic[1] = IEB_MAGIC[1];
    header.magic[2] = IEB_MAGIC[2];
    header.magic[3] = IEB_MAGIC[3];

    // IEB file format version 0.1
    header.ver[0] = 0;
    header.ver[1] = 1;

    header.precision = static_cast<char>(ov::element::Type_t(desc.getPrecision()));

    if (desc.getShape().getRank() > 7) {
        OPENVINO_THROW("Dumper support max 7D blobs");
    }

    header.ndims = desc.getShape().getRank();
    const auto& dims = desc.getShape().getStaticDims();
    for (int i = 0; i < header.ndims; i++) {
        header.dims[i] = dims[i];
    }

    header.scaling_axis = NO_SCALES;

    return header;
}

static DnnlBlockedMemoryDesc parse_header(IEB_HEADER& header) {
    if (header.magic[0] != IEB_MAGIC[0] || header.magic[1] != IEB_MAGIC[1] || header.magic[2] != IEB_MAGIC[2] ||
        header.magic[3] != IEB_MAGIC[3]) {
        OPENVINO_THROW("Dumper cannot parse file. Wrong format.");
    }

    if (header.ver[0] != 0 || header.ver[1] != 1) {
        OPENVINO_THROW("Dumper cannot parse file. Unsupported IEB format version.");
    }

    const auto prc = static_cast<ov::element::Type_t>(header.precision);
    VectorDims dims(header.ndims);
    for (int i = 0; i < header.ndims; i++) {
        dims[i] = header.dims[i];
    }

    return DnnlBlockedMemoryDesc{prc, Shape(dims)};
}

void BlobDumper::prepare_plain_data(const MemoryPtr& memory, std::vector<uint8_t>& data) const {
    const auto& desc = memory->getDesc();
    size_t data_size = desc.getShape().getElementsCount();
    const auto size = data_size * desc.getPrecision().size();
    data.resize(size);

    // check if it already plain
    if (desc.hasLayoutType(LayoutType::ncsp)) {
        cpu_memcpy(data.data(), memory->getDataAs<const uint8_t>(), size);
        return;
    }

    // Copy to plain
    const void* ptr = memory->getData();

    switch (desc.getPrecision()) {
    case ov::element::f32:
    case ov::element::i32: {
        auto* pln_blob_ptr = reinterpret_cast<int32_t*>(data.data());
        auto* blob_ptr = reinterpret_cast<const int32_t*>(ptr);
        for (size_t i = 0; i < data_size; i++) {
            pln_blob_ptr[i] = blob_ptr[desc.getElementOffset(i)];
        }
        break;
    }
    case ov::element::bf16: {
        auto* pln_blob_ptr = reinterpret_cast<int16_t*>(data.data());
        auto* blob_ptr = reinterpret_cast<const int16_t*>(ptr);
        for (size_t i = 0; i < data_size; i++) {
            pln_blob_ptr[i] = blob_ptr[desc.getElementOffset(i)];
        }
        break;
    }
    case ov::element::f16: {
        auto* pln_blob_ptr = reinterpret_cast<float16*>(data.data());
        auto* blob_ptr = reinterpret_cast<const float16*>(ptr);
        for (size_t i = 0; i < data_size; i++) {
            pln_blob_ptr[i] = blob_ptr[desc.getElementOffset(i)];
        }
        break;
    }
    case ov::element::i8:
    case ov::element::u8: {
        auto* pln_blob_ptr = reinterpret_cast<int8_t*>(data.data());
        auto* blob_ptr = reinterpret_cast<const int8_t*>(ptr);
        for (size_t i = 0; i < data_size; i++) {
            pln_blob_ptr[i] = blob_ptr[desc.getElementOffset(i)];
        }
        break;
    }
    default:
        OPENVINO_THROW("Dumper. Unsupported precision");
    }
}

void BlobDumper::dump(std::ostream& stream) const {
    if (memory == nullptr) {
        OPENVINO_THROW("Dumper cannot dump. Memory is not allocated.");
    }

    IEB_HEADER header = prepare_header(memory->getDesc());
    std::vector<uint8_t> data;
    prepare_plain_data(this->memory, data);

    header.data_offset = sizeof(header);
    header.data_size = data.size();
    header.scaling_data_offset = 0;
    header.scaling_data_size = 0;

    stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
    stream.write(reinterpret_cast<char*>(data.data()), data.size());
}

void BlobDumper::dumpAsTxt(std::ostream& stream) const {
    if (memory == nullptr) {
        OPENVINO_THROW("Dumper cannot dump. Memory is not allocated.");
    }

    const auto& desc = memory->getDesc();
    const auto dims = desc.getShape().getStaticDims();
    size_t data_size = desc.getShape().getElementsCount();

    // Header like "U8 4D shape: 2 3 224 224 ()
    stream << memory->getDesc().getPrecision().get_type_name() << " " << dims.size() << "D "
           << "shape: ";
    for (size_t d : dims) {
        stream << d << " ";
    }
    stream << "(" << data_size << ")"
           << " by address 0x" << std::hex << memory->getDataAs<const int64_t>() << std::dec << '\n';

    const void* ptr = memory->getData();

    switch (desc.getPrecision()) {
    case ov::element::f32: {
        auto* blob_ptr = reinterpret_cast<const float*>(ptr);
        for (size_t i = 0; i < data_size; i++) {
            stream << blob_ptr[desc.getElementOffset(i)] << '\n';
        }
        break;
    }
    case ov::element::i32: {
        auto* blob_ptr = reinterpret_cast<const int32_t*>(ptr);
        for (size_t i = 0; i < data_size; i++) {
            stream << blob_ptr[desc.getElementOffset(i)] << '\n';
        }
        break;
    }
    case ov::element::bf16: {
        auto* blob_ptr = reinterpret_cast<const bfloat16_t*>(ptr);
        for (size_t i = 0; i < data_size; i++) {
            auto fn = static_cast<float>(blob_ptr[desc.getElementOffset(i)]);
            stream << fn << '\n';
        }
        break;
    }
    case ov::element::f16: {
        auto* blob_ptr = reinterpret_cast<const float16*>(ptr);
        for (size_t i = 0; i < data_size; i++) {
            stream << blob_ptr[desc.getElementOffset(i)] << '\n';
        }
        break;
    }
    case ov::element::i8: {
        auto* blob_ptr = reinterpret_cast<const int8_t*>(ptr);
        for (size_t i = 0; i < data_size; i++) {
            stream << static_cast<int>(blob_ptr[desc.getElementOffset(i)]) << '\n';
        }
        break;
    }
    case ov::element::u8: {
        auto* blob_ptr = reinterpret_cast<const uint8_t*>(ptr);
        for (size_t i = 0; i < data_size; i++) {
            stream << static_cast<int>(blob_ptr[desc.getElementOffset(i)]) << '\n';
        }
        break;
    }
    case ov::element::i64: {
        auto* blob_ptr = reinterpret_cast<const int64_t*>(ptr);
        for (size_t i = 0; i < data_size; i++) {
            stream << blob_ptr[desc.getElementOffset(i)] << '\n';
        }
        break;
    }
    case ov::element::u32: {
        auto* blob_ptr = reinterpret_cast<const uint32_t*>(ptr);
        for (size_t i = 0; i < data_size; i++) {
            stream << blob_ptr[desc.getElementOffset(i)] << '\n';
        }
        break;
    }
    case ov::element::u16: {
        auto* blob_ptr = reinterpret_cast<const uint16_t*>(ptr);
        for (size_t i = 0; i < data_size; i++) {
            stream << blob_ptr[desc.getElementOffset(i)] << '\n';
        }
        break;
    }
    case ov::element::i16: {
        auto* blob_ptr = reinterpret_cast<const int16_t*>(ptr);
        for (size_t i = 0; i < data_size; i++) {
            stream << blob_ptr[desc.getElementOffset(i)] << '\n';
        }
        break;
    }
    default:
        break;
        OPENVINO_THROW("Dumper. Unsupported precision");
    }
}

BlobDumper BlobDumper::read(std::istream& stream) {
    IEB_HEADER header;
    stream.read(reinterpret_cast<char*>(&header), sizeof(header));

    const auto desc = parse_header(header);

    BlobDumper res(desc);
    stream.seekg(header.data_offset, stream.beg);
    stream.read(reinterpret_cast<char*>(res.getDataPtr()), header.data_size);

    return res;
}

BlobDumper BlobDumper::read(const std::string& file_path) {
    std::ifstream file;
    file.open(file_path);
    if (!file.is_open()) {
        OPENVINO_THROW("Dumper cannot open file ", file_path);
    }

    auto res = read(file);
    file.close();
    return res;
}

void BlobDumper::dump(const std::string& dump_path) const {
    std::ofstream dump_file;
    dump_file.open(dump_path, std::ios::binary);
    if (!dump_file.is_open()) {
        OPENVINO_THROW("Dumper cannot create dump file ", dump_path);
    }

    dump(dump_file);
    dump_file.close();
}

void BlobDumper::dumpAsTxt(const std::string& dump_path) const {
    std::ofstream dump_file;
    dump_file.open(dump_path);
    if (!dump_file.is_open()) {
        OPENVINO_THROW("Dumper cannot create dump file ", dump_path);
    }

    dumpAsTxt(dump_file);
    dump_file.close();
}

}  // namespace ov::intel_cpu
