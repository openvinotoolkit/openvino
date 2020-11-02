// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/blob_reader.hpp>

#include <sstream>
#include <memory>
#include <vector>
#include <string>

#include <ie_input_info.hpp>

#include <vpu/graph_transformer.hpp>
#include <vpu/backend/blob_format.hpp>
#include <vpu/model/data.hpp>

namespace vpu {

namespace {

template <typename T>
T readFromBlob(const std::vector<char>& blob, uint32_t& offset) {
    IE_ASSERT(offset + sizeof(T) <= blob.size());

    auto srcPtr = blob.data() + offset;
    offset += sizeof(T);

    return *reinterpret_cast<const T*>(srcPtr);
}

}  // namespace

void BlobReader::parse(const std::vector<char>& blob) {
    if (blob.empty() || blob.size() < sizeof(ElfN_Ehdr) + sizeof(mv_blob_header)) {
        VPU_THROW_EXCEPTION << "BlobReader error: Blob is empty";
    }

    _pBlob = blob.data();

    _blobHeader = *reinterpret_cast<const mv_blob_header*>(blob.data() + sizeof(ElfN_Ehdr));
    if (_blobHeader.magic_number != BLOB_MAGIC_NUMBER) {
        VPU_THROW_EXCEPTION << "BlobReader error: The magic number imported blob doesn't match graph_transformer";
    }
    if (_blobHeader.blob_ver_major != BLOB_VERSION_MAJOR || _blobHeader.blob_ver_minor != BLOB_VERSION_MINOR) {
        VPU_THROW_EXCEPTION << "BlobReader error: The version of imported blob doesn't match graph_transformer";
    }

    _inputInfo.totalSize = _blobHeader.inputs_size;
    _outputInfo.totalSize = _blobHeader.outputs_size;

    auto inputInfoSecOffset = _blobHeader.input_info_section_offset;
    for (uint32_t i = 0; i < _blobHeader.inputs_count; i++) {
        auto ioIdx = readFromBlob<uint32_t>(blob, inputInfoSecOffset);
        IE_ASSERT(ioIdx == i);

        auto ioBufferOffset = readFromBlob<int32_t>(blob, inputInfoSecOffset);

        auto nameLength = readFromBlob<uint32_t>(blob, inputInfoSecOffset);
        std::string inputName(nameLength, 0);
        for (auto& c : inputName) {
            c = readFromBlob<char>(blob, inputInfoSecOffset);
        }

        // Truncate zeros
        inputName = inputName.c_str();

        auto dataType = readFromBlob<DataType>(blob, inputInfoSecOffset);
        auto orderCode = readFromBlob<uint32_t>(blob, inputInfoSecOffset);

        auto numDims = readFromBlob<uint32_t>(blob, inputInfoSecOffset);

        auto dimsOrder = DimsOrder::fromCode(orderCode);
        auto perm = dimsOrder.toPermutation();
        IE_ASSERT(perm.size() == numDims);

        auto dimsLocation = readFromBlob<Location>(blob, inputInfoSecOffset);
        VPU_THROW_UNLESS(dimsLocation == Location::Blob,
            "BlobReader error while parsing {} input data: only Blob location for input shape is supported, but {} was given",
            inputName, dimsLocation);
        auto dimsOffset = _blobHeader.const_data_section_offset + readFromBlob<uint32_t>(blob, inputInfoSecOffset);

        // Skip strides' location and offset
        inputInfoSecOffset += 2 * sizeof(uint32_t);

        DimValues vpuDims;

        for (int i = 0; i < perm.size(); ++i) {
            vpuDims.set(perm[i], readFromBlob<uint32_t>(blob, dimsOffset));
        }

        ie::TensorDesc ieDesc = DataDesc(dataType, dimsOrder, vpuDims).toTensorDesc();
        ie::Data inputData(inputName, ieDesc);

        ie::InputInfo input;
        input.setInputData(std::make_shared<ie::Data>(inputData));

        _networkInputs[input.name()]    = std::make_shared<ie::InputInfo>(input);
        _inputInfo.offset[input.name()] = ioBufferOffset;
    }

    auto outputInfoSecOffset = _blobHeader.output_info_section_offset;
    for (size_t i = 0; i < _blobHeader.outputs_count; i++) {
        auto ioIdx = readFromBlob<uint32_t>(blob, outputInfoSecOffset);
        IE_ASSERT(ioIdx == i);

        auto ioBufferOffset = readFromBlob<int32_t>(blob, outputInfoSecOffset);

        auto nameLength = readFromBlob<uint32_t>(blob, outputInfoSecOffset);
        std::string outputName(nameLength, 0);
        for (auto& c : outputName) {
            c = readFromBlob<char>(blob, outputInfoSecOffset);
        }

        // Truncate zeros
        outputName = outputName.c_str();

        auto dataType = readFromBlob<DataType>(blob, outputInfoSecOffset);
        auto orderCode = readFromBlob<uint32_t>(blob, outputInfoSecOffset);

        auto numDims = readFromBlob<uint32_t>(blob, outputInfoSecOffset);

        auto dimsOrder = DimsOrder::fromCode(orderCode);
        auto perm = dimsOrder.toPermutation();
        IE_ASSERT(perm.size() == numDims);

        auto dimsLocation = readFromBlob<Location>(blob, outputInfoSecOffset);
        VPU_THROW_UNLESS(dimsLocation == Location::Blob,
            "BlobReader error while parsing {} output data: only Blob location for output shape is supported, but {} was given",
            outputName, dimsLocation);
        auto dimsOffset = _blobHeader.const_data_section_offset + readFromBlob<uint32_t>(blob, outputInfoSecOffset);

        // Skip strides' location and offset
        outputInfoSecOffset += 2 * sizeof(uint32_t);

        DimValues vpuDims;

        for (int i = 0; i < perm.size(); ++i) {
            vpuDims.set(perm[i], readFromBlob<uint32_t>(blob, dimsOffset));
        }

        ie::TensorDesc ieDesc = DataDesc(dataType, dimsOrder, vpuDims).toTensorDesc();
        ie::Data outputData(outputName, ieDesc);

        _networkOutputs[outputData.getName()]    = std::make_shared<ie::Data>(outputData);
        _outputInfo.offset[outputData.getName()] = ioBufferOffset;
    }
}

}  // namespace vpu
