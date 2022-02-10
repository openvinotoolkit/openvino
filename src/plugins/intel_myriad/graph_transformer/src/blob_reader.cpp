// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/blob_reader.hpp>

#include <sstream>
#include <memory>
#include <vector>
#include <string>

#include <ie_input_info.hpp>
#include <ie_ngraph_utils.hpp>
#include <vpu/graph_transformer.hpp>
#include <vpu/backend/blob_format.hpp>
#include <vpu/model/data.hpp>
#include <vpu/utils/shape_io.hpp>

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

    const auto readIO = [this, &blob](DataInfo& ioInfo, uint32_t& ioSectionOffset, uint32_t idx) {
            auto ioIdx = readFromBlob<uint32_t>(blob, ioSectionOffset);
            VPU_THROW_UNLESS(ioIdx == idx, "BlobReader failed on I/O processing, its' ioIdx parameter (which is {}) is "
                "different from its' processing order (which is {})", ioIdx, idx);

            auto ioBufferOffset = readFromBlob<int32_t>(blob, ioSectionOffset);

            auto nameLength = readFromBlob<uint32_t>(blob, ioSectionOffset);
            std::string ioName(nameLength, 0);
            for (auto& c : ioName) {
                c = readFromBlob<char>(blob, ioSectionOffset);
            }

            // Truncate zeros
            ioName = ioName.c_str();

            auto dataType = readFromBlob<DataType>(blob, ioSectionOffset);
            auto orderCode = readFromBlob<uint32_t>(blob, ioSectionOffset);

            auto numDims = readFromBlob<uint32_t>(blob, ioSectionOffset);

            auto dimsOrder = DimsOrder::fromCode(orderCode);
            auto perm = dimsOrder.toPermutation();
            IE_ASSERT(perm.size() == numDims);

            auto dimsLocation = readFromBlob<Location>(blob, ioSectionOffset);
            VPU_THROW_UNLESS(dimsLocation == Location::Blob,
                             "BlobReader error while parsing data {}: only Blob location for input/output shape is supported, but {} was given",
                             ioName, dimsLocation);
            auto dimsOffset = _blobHeader.const_data_section_offset + readFromBlob<uint32_t>(blob, ioSectionOffset);

            // Skip strides' location and offset
            ioSectionOffset += 2 * sizeof(uint32_t);

            DimValues vpuDims;

            for (const auto& dim : perm) {
                vpuDims.set(dim, readFromBlob<uint32_t>(blob, dimsOffset));
            }

            ie::TensorDesc ieDesc = DataDesc(dataType, dimsOrder, vpuDims).toTensorDesc();
            ie::Data ioData(ioName, ieDesc);

            ioInfo.offset[ioName] = ioBufferOffset;
            ioInfo.descFromPlugin[ioName] = ieDesc;

            return ioData;
    };

    auto inputSectionOffset = _blobHeader.input_info_section_offset;
    for (uint32_t i = 0; i < _blobHeader.inputs_count; i++) {
        const auto processedInput = readIO(_inputInfo, inputSectionOffset, i);
        if (!isIOShapeName(processedInput.getName())) {
            ie::InputInfo input;
            input.setInputData(std::make_shared<ie::Data>(processedInput));
            _networkInputs[processedInput.getName()] = std::make_shared<ie::InputInfo>(input);
        }
    }

    auto outputSectionOffset = _blobHeader.output_info_section_offset;
    for (uint32_t i = 0; i < _blobHeader.outputs_count; i++) {
        const auto processedOutput = readIO(_outputInfo, outputSectionOffset, i);
        if (!isIOShapeName(processedOutput.getName())) {
            _networkOutputs[processedOutput.getName()] = std::make_shared<ie::Data>(processedOutput);
        }
    }
    if (blob.size() != _blobHeader.file_size) {
        auto networkInfoOffset = _blobHeader.file_size;
        const auto nih = readFromBlob<network_info_header>(blob, networkInfoOffset);
        auto extractParameter = [&blob, &networkInfoOffset](bool isResult) {
            const auto nph = readFromBlob<network_params_header>(blob, networkInfoOffset);
            std::string parameterFriendlyName(nph.name_lenght, '0');

            for (auto idx = 0; idx < nph.name_lenght; ++idx) {
                parameterFriendlyName[idx] = readFromBlob<char>(blob, networkInfoOffset);
            }

            ov::Shape parameterShape(nph.shape_size);
            for (auto idx = 0; idx < nph.shape_size; ++idx) {
                parameterShape[idx] = readFromBlob<size_t>(blob, networkInfoOffset);
            }

            ov::element::Type_t parameterType = readFromBlob<ov::element::Type_t>(blob, networkInfoOffset);
            std::shared_ptr<ov::Node> parameter =
                std::make_shared<ov::op::v0::Parameter>(parameterType,
                                                        parameterShape);

            std::unordered_set<std::string> tensorNames;
            for (auto idx = 0; idx < nph.output_tensor_names_size; ++idx) {
                const auto nameLenght = readFromBlob<size_t>(blob, networkInfoOffset);
                std::string tensorName;
                for (auto nameSymbolIdx = 0; nameSymbolIdx < nameLenght; ++nameSymbolIdx) {
                    tensorName += readFromBlob<char>(blob, networkInfoOffset);
                }
                tensorNames.insert(tensorName);
            }
            if (isResult) {
                auto fakeParameter = parameter;
                parameter = std::make_shared<ov::op::v0::Result>(parameter);

                const auto inputNameLenght = readFromBlob<size_t>(blob, networkInfoOffset);

                std::string inputName;
                for (auto nameSymbolIdx = 0; nameSymbolIdx < inputNameLenght; ++nameSymbolIdx) {
                    inputName += readFromBlob<char>(blob, networkInfoOffset);
                }
                fakeParameter->set_friendly_name(inputName);

                parameter = parameter->copy_with_new_inputs({fakeParameter});
            }
            parameter->set_friendly_name(parameterFriendlyName);
            parameter->output(0).get_tensor().set_names(tensorNames);

            return parameter;
        };

        for (auto paramIdx = 0; paramIdx < nih.parameters_size; ++paramIdx) {
            _parameters.emplace_back(extractParameter(false));
        }

        for (auto paramIdx = 0; paramIdx < nih.results_size; ++paramIdx) {
            _results.emplace_back(extractParameter(true));
        }
    }
}

}  // namespace vpu
