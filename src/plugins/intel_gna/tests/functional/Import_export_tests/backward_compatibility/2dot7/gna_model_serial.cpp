// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <array>
#include <ios>
#include <iomanip>
#include <map>
#include <ie_algorithm.hpp>
#include <ie_common.h>
#include <ie_precision.hpp>

#if defined __INTEL_COMPILER || defined _MSC_VER
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif

#include "gna_model_serial.hpp"
#include "common/versioning.hpp"
#include "gna2_model_helper.hpp"
#include "serial/headers/2dot7/gna_model_header.hpp"

#ifdef GNA_DEBUG
#include <ngraph/pass/manager.hpp>
#include "transformations/serialize.hpp"
#endif

using namespace ov::intel_gna::header_2_dot_7;

inline void writeNBytes(const void *ptr, uint32_t size, std::ostream & os) {
    os.write(static_cast<const char*>(ptr), size);
}

template <class T>
inline void writeBits(const T & obj, std::ostream & os) {
    os.write(reinterpret_cast<const char *>(&obj), sizeof(T));
}

template <class T>
inline void readBits(T & obj, std::istream & is) {
    is.read(reinterpret_cast<char *>(&obj), sizeof(T));
}

inline void readNBytes(void * ptr, uint32_t size, std::istream & is) {
    is.read(reinterpret_cast<char *>(ptr), size);
}

template <int nBits, class T>
inline void readNBits(T & obj, std::istream & is) {
    std::array<uint8_t, nBits / 8> tmp;
    is.read(reinterpret_cast<char *>(&tmp), nBits / 8);

    obj = * reinterpret_cast<T*>(&tmp.front());
}

inline void * offsetToPointer(void * const base, uint64_t offset) {
    return reinterpret_cast<uint8_t *>(base) + offset;
}

template <class T>
inline void readOffset(T & ptr, void *base,  std::istream & is) {
    uint64_t offset = 0ull;
    readBits(offset, is);
    ptr = reinterpret_cast<T>(offsetToPointer(base, offset));
}

#define offsetFromBase(field)\
getOffsetFromBase(field, #field)

bool IsEmptyTensor(const Gna2Tensor& t) {
    return t.Type == Gna2DataTypeNone &&
        t.Data == nullptr &&
        t.Layout[0] == '\0' &&
        t.Mode == Gna2TensorModeDefault &&
        t.Shape.NumberOfDimensions == 0;
}

const std::map<Gna2OperationType, std::vector<uint32_t>> GnaParamSize{
    {Gna2OperationTypeFullyConnectedAffine, {sizeof(Gna2BiasMode), sizeof(uint32_t)}},
    {Gna2OperationTypeConvolution, {
        sizeof(Gna2Shape),
        sizeof(Gna2BiasMode),
        sizeof(Gna2PoolingMode),
        sizeof(Gna2Shape),
        sizeof(Gna2Shape),
        sizeof(Gna2Shape)}},
    {Gna2OperationTypeCopy, {sizeof(Gna2Shape)}},
    {Gna2OperationTypeTransposition, {sizeof(Gna2Shape)}},
};

void GNAModelSerial::Export(const GnaAllocations& allocations, std::ostream& os) const {
    os.exceptions(std::ostream::failbit);

    const std::vector<Gna2Operation>
        layers(gna2Model->Operations, gna2Model->Operations + gna2Model->NumberOfOperations);

    const auto gnaGraphSize = allocations.GetSizeForExport();
    const auto& allocationsOrdered = allocations.GetAllocationsInExportOrder();

    // all offsets will be from this pointer
    auto getTensorWithProperOffset = [&allocationsOrdered](const Gna2Tensor& tensor) {
        Gna2Tensor out = tensor;
        const auto found = GnaAllocations::GetOffsetForExport(allocationsOrdered, tensor.Data);
        if (!found.first) {
            THROW_GNA_EXCEPTION << "Tensor data pointer not found in allocations\n";
        }
        out.Data = reinterpret_cast<void*>(found.second);
        return out;
    };

    auto convert_to_serial = [&allocationsOrdered](const RuntimeEndPoint& ep) {
        RuntimeEndPoint out;
        out.elements_count = ep.elements_count;
        const auto found = GnaAllocations::GetOffsetForExport(allocationsOrdered, ep.descriptor_ptr);
        if (!found.first) {
            THROW_GNA_EXCEPTION << "Endpoint data pointer not found in allocations\n";
        }
        out.descriptor_offset = found.second;
        out.scaleFactor = ep.scaleFactor;
        out.element_size = ep.element_size;
        out.shape = ep.shape;
        out.layout = ep.layout;
        out.precision = ep.precision;
        out.orientation = ep.orientation;
        return out;
    };

    /**
     * writing header
     */
    ModelHeader header;
    header.gnam[0] = 'G';
    header.gnam[1] = 'N';
    header.gnam[2] = 'A';
    header.gnam[3] = 'M';
    header.headerSize = sizeof(ModelHeader);
    header.gnaMemSize = gnaGraphSize;
    header.layersCount = layers.size();
    header.nGroup = 1; // just to support the old models
    header.nInputs = inputs.size();
    header.nOutputs = outputs.size();
    header.nTransposeInputs = transposeInputsInfo.size();
    header.nTransposeOutputs = transposeOutputsInfo.size();

    writeBits(header, os);

    for (auto &name : inputNames) {
        const auto nameSize = strlen(name.c_str()) + 1;
        writeBits(static_cast<uint32_t>(nameSize), os);
        writeNBytes(name.c_str(), nameSize , os);
    }
    ExportTranspositionInfo(os, transposeInputsInfo);
    ExportTranspositionInfo(os, transposeOutputsInfo);
    for (const auto &input : inputs) {
        writeBits(convert_to_serial(input), os);
    }
    for (auto &name : outputNames) {
        const auto nameSize = strlen(name.c_str()) + 1;
        writeBits(static_cast<uint32_t>(nameSize), os);
        writeNBytes(name.c_str(), nameSize, os);
    }
    for (const auto &output : outputs) {
        writeBits(convert_to_serial(output), os);
    }

    for (const auto & layer : layers) {
        writeBits(static_cast<uint32_t>(layer.Type), os);
        writeBits(layer.NumberOfOperands, os);

        for (uint32_t i = 0; i < layer.NumberOfOperands; i++) {
            if (layer.Operands[i] == nullptr) {
                writeBits(Gna2Tensor{}, os);
            } else {
                Gna2Tensor tensor = getTensorWithProperOffset(*layer.Operands[i]);
                // we need to remove legacy (up to & including GNA HW 2.0) CNN enforement during export
                // to avoid issues when importing and running the model on newer GNA HW with libGNA 2.1.x.y
                if (i == OutOpIdx && layer.Type == Gna2OperationTypeConvolution) {
                    memset(tensor.Layout, 0, sizeof(tensor.Layout));
                }
                writeBits(tensor, os);
            }
        }

        writeBits(layer.NumberOfParameters, os);

        // writing parameters
        switch (layer.Type) {
        case Gna2OperationTypeElementWiseAffine:
        case Gna2OperationTypeFullyConnectedAffine:
        case Gna2OperationTypeConvolution:
        case Gna2OperationTypeCopy:
        case Gna2OperationTypeTransposition:
            break;
        case Gna2OperationTypeRecurrent:
            THROW_GNA_EXCEPTION << "Exporting of recurrent operation not supported";
        default:
            THROW_GNA_EXCEPTION << "Exporting of unknown GNA operation type(" << layer.Type << ")  not supported";
        }
        for (uint32_t i = 0; i < layer.NumberOfParameters; i++) {
            if (layer.Parameters[i] == nullptr) {
                writeBits(static_cast<uint32_t>(0), os);
                continue;
            }
            const auto paramSize = GnaParamSize.at(layer.Type).at(i);
            writeBits(paramSize, os);
            writeNBytes(layer.Parameters[i], paramSize, os);
        }
    }
    // writing memory information
    writeBits(static_cast<uint32_t>(states.size()), os);
    for (auto && state : states) {
        void* gna_ptr = nullptr;
        uint32_t reserved_size = 0;
        std::string name;
        float scale_factor = 1.0f;
        std::tie(gna_ptr, reserved_size, name, scale_factor) = state;
        const auto found = GnaAllocations::GetOffsetForExport(allocationsOrdered, gna_ptr);
        if (!found.first) {
            THROW_GNA_EXCEPTION << "State data pointer not found in allocations\n";
        }
        writeBits(found.second, os);
        writeBits(reserved_size, os);
        const auto nameSize = strlen(name.c_str()) + 1;
        writeBits(static_cast<uint32_t>(nameSize), os);
        writeNBytes(name.c_str(), nameSize, os);
        writeBits(scale_factor, os);
    }

    // once structure has been written lets push gna graph
    for (const auto& a : allocationsOrdered) {
        os.write(reinterpret_cast<char*>(a.ptr), a.sizeForExport());
    }
}

std::vector<RuntimeEndPoint> GNAModelSerial::serializeOutputs(const InferenceEngine::OutputsDataMap& outputsDataMap,
        const std::vector<ov::intel_gna::OutputDesc>& outputsDesc) {
    std::vector<RuntimeEndPoint> endPoints;
    std::size_t outputIndex = 0;
    for (auto const &output : outputsDataMap) {
        auto outputName = output.first;
        auto outputDims = output.second->getTensorDesc().getDims();
        RuntimeEndPoint::Shape outputShape;
        outputShape.NumberOfDimensions = outputDims.size();
        for (size_t i=0; i < outputShape.NumberOfDimensions; ++i) {
            outputShape.Dimensions[i] = static_cast<uint32_t>(outputDims[i]);
        }
        uint32_t elementsCount = static_cast<uint32_t>(InferenceEngine::details::product(outputDims.begin(), outputDims.end()));
        InferenceEngine::Layout outputLayout = output.second->getLayout();
        InferenceEngine::Precision::ePrecision outputPrecision = InferenceEngine::Precision::FP32;
        RuntimeEndPoint endPoint(outputsDesc[outputIndex].scale_factor,
                                                 outputsDesc[outputIndex].ptrs[0],
                                                 outputsDesc[outputIndex].tensor_precision.size(),
                                                 elementsCount,
                                                 outputShape,
                                                 outputLayout,
                                                 outputPrecision,
                                                 outputsDesc[outputIndex].orientation);
        endPoints.push_back(endPoint);
        outputIndex++;
    }
    return endPoints;
}

std::vector<RuntimeEndPoint> GNAModelSerial::serializeInputs(const InferenceEngine::InputsDataMap& inputsDataMap,
                                                             const std::vector<ov::intel_gna::InputDesc>& inputDesc) {
    std::vector<RuntimeEndPoint> endPoints;

    std::size_t inputIndex = 0;
    for (auto const& input : inputsDataMap) {
        auto inputName = input.first;
        auto inputDims = input.second->getTensorDesc().getDims();
        RuntimeEndPoint::Shape inputShape;
        inputShape.NumberOfDimensions = inputDims.size();
        for (size_t i=0; i < inputShape.NumberOfDimensions; ++i) {
            inputShape.Dimensions[i] = static_cast<uint32_t>(inputDims[i]);
        }
        double scaleFactor = inputDesc[inputIndex].scale_factor;
        std::vector<void *> descriptor_ptr = inputDesc[inputIndex].ptrs;
        IE_ASSERT(descriptor_ptr.size() > 0);
        uint32_t element_size = 2u;
        uint32_t elementsCount = static_cast<uint32_t>(InferenceEngine::details::product(inputDims.begin(), inputDims.end()));
        intel_dnn_orientation_t orientation = inputDesc[inputIndex].orientation;
        InferenceEngine::Layout inputLayout = input.second->getLayout();
        InferenceEngine::Precision::ePrecision inputPrecision = InferenceEngine::Precision::FP32;
        RuntimeEndPoint endPoint(scaleFactor,
                                                 descriptor_ptr[0],
                                                 element_size,
                                                 elementsCount,
                                                 inputShape,
                                                 inputLayout,
                                                 inputPrecision,
                                                 orientation);
        endPoints.push_back(endPoint);
        inputIndex++;
    }
    return endPoints;
}

void GNAModelSerial::ExportTranspositionInfo(std::ostream &os,
        const TranspositionInfoMap &transpositionInfoMap) const {
    for (const auto &transpositionInfo : transpositionInfoMap) {
        auto nameSize = strlen(transpositionInfo.first.c_str());
        writeBits(static_cast<uint32_t>(nameSize), os);
        writeNBytes(transpositionInfo.first.c_str(), nameSize, os);
        auto fragmentsNum = transpositionInfo.second.size();
        writeBits(static_cast<uint32_t>(fragmentsNum), os);
        for (const auto &transposeFragmentInfo : transpositionInfo.second) {
            writeNBytes(&transposeFragmentInfo, sizeof(TranspositionInfo), os);
        }
    }
}