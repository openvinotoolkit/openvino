// Copyright (C) 2018-2021 Intel Corporation
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
#include <serial/headers/2dot2/gna_model_header.hpp>
#include <serial/headers/2dot5/gna_model_header.hpp>
#include <serial/headers/2dot6/gna_model_header.hpp>

#endif

#include "gna_plugin.hpp"
#include "gna_model_serial.hpp"
#include "serial/headers/latest/gna_model_header.hpp"

using namespace GNAPluginNS;

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

union {
    uint16_t s;
    uint8_t  c[2];
} constexpr static  LECheck {1};

bool is_little_endian() {
    return LECheck.c[0] == 1;
}

const int gna_header_magic = is_little_endian() ?  0x4d414e47 : 0x474e414d;

GNAPluginNS::HeaderLatest::ModelHeader GNAModelSerial::ReadHeader(std::istream &is) {
    is.exceptions(std::istream::failbit);
    auto startPos = is.tellg();
    if (startPos == -1) {
        THROW_GNA_EXCEPTION << "Can't open stream to import";
    }
    is.seekg(0, is.end);
    auto stream_len = is.tellg();
    if (stream_len == -1) {
        THROW_GNA_EXCEPTION << "Can't open file to import";
    }
    stream_len -= startPos;
    is.seekg(startPos, is.beg);

    HeaderLatest::ModelHeader header;
    header.version.major = 0u;
    header.version.minor = 0u;
    auto size_of_headers_header = sizeof(HeaderLatest::ModelHeader::gnam) + sizeof(HeaderLatest::ModelHeader::headerSize)
                                + sizeof(HeaderLatest::ModelHeader::Version);
    if (stream_len > size_of_headers_header) {
        readNBytes(&header, size_of_headers_header, is);
    } else {
        readNBytes(&header, stream_len, is);
    }
    if (*reinterpret_cast<int*>(header.gnam) != gna_header_magic) {
        THROW_GNA_EXCEPTION << "Imported file unsupported: magic number should be GNAM(0x474e414d), but was 0x"
                           << std::setfill('0') <<
                           std::hex << std::setw(2) << static_cast<short>(header.gnam[0]) <<
                           std::hex << std::setw(2) << static_cast<short>(header.gnam[1]) <<
                           std::hex << std::setw(2) << static_cast<short>(header.gnam[2]) <<
                           std::hex << std::setw(2) << static_cast<short>(header.gnam[3]);
    }

    is.seekg(startPos, is.beg);
    Header2dot1::ModelHeader tempHeader2dot1;
    switch (header.version.major) {
        case 2:
            switch (header.version.minor) {
                case 1:
                    readBits(tempHeader2dot1, is);
                    header = HeaderLatest::ModelHeader(tempHeader2dot1);
                    break;
                case 2:
                case 3:
                {
                    Header2dot3::ModelHeader tempHeader2dot3;
                    readBits(tempHeader2dot3, is);
                    header = HeaderLatest::ModelHeader(tempHeader2dot3);
                    break;
                }
                case 4:
                {
                    Header2dot4::ModelHeader tempHeader2dot4;
                    readBits(tempHeader2dot4, is);
                    header = HeaderLatest::ModelHeader(tempHeader2dot4);
                    break;
                }
                case 5:
                case 6:
                case 7:
                    readNBytes(&header, sizeof(HeaderLatest::ModelHeader), is);
                    break;
                default:
                    THROW_GNA_EXCEPTION << "Imported file unsupported. minor version should have values in range 1 to 7 and is: " << header.version.minor;
            }
            break;
        default:
            THROW_GNA_EXCEPTION << "Imported file unsupported. Import for files with major version equal to: " << header.version.major << " is not implemented";
    }

    /*
     * extra data need to be added into new header and modify check as appropriate
     */

    //  forward compatible
    if (header.headerSize > sizeof(header)) {
        is.seekg(header.headerSize - sizeof(header), std::ios_base::cur);
    }
    return header;
}

GNAPluginNS::HeaderLatest::RuntimeEndPoint GNAModelSerial::ReadEndPoint(std::istream &is) {
    is.exceptions(std::istream::failbit);

    HeaderLatest::RuntimeEndPoint endPoint;
    switch (modelHeader.version.major) {
        case 2:
            switch (modelHeader.version.minor) {
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                {
                    Header2dot6::RuntimeEndPoint tempEndPoint2dot6;
                    readBits(tempEndPoint2dot6, is);
                    endPoint = HeaderLatest::RuntimeEndPoint(tempEndPoint2dot6, modelHeader.nGroup);
                    break;
                }
                case 7:
                    readNBytes(&endPoint, sizeof(HeaderLatest::RuntimeEndPoint), is);
                    break;
                default:
                    THROW_GNA_EXCEPTION << "Imported file unsupported. minor version should have values in range 1 to 7 and is: " << modelHeader.version.minor;
            }
            break;
        default:
            THROW_GNA_EXCEPTION << "Imported file unsupported. Import for files with major version equal to: "
            << modelHeader.version.major << " is not implemented";
    }

    return endPoint;
}

#define offsetFromBase(field)\
getOffsetFromBase(field, #field)

#if GNA_LIB_VER == 2

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

void GNAModelSerial::Import(void *basePointer,
        size_t gnaGraphSize,
        std::istream & is,
        std::shared_ptr<GNAPluginNS::InputDesc> inputsDesc,
        std::vector<GNAPluginNS::OutputDesc> &desc,
        InferenceEngine::InputsDataMap& inputsDataMap,
        InferenceEngine::OutputsDataMap& outputsDataMap,
        TranspositionInfoMap& inputsTranspositionInfo,
        TranspositionInfoMap& outputsTranspositionInfo) {
    is.exceptions(std::istream::failbit);

    if (modelHeader.version.major == 2) {
        if (modelHeader.version.minor >= 3) {
            for (auto inputIndex = 0; inputIndex < modelHeader.nInputs; inputIndex++) {
                uint32_t nameSize = 0;
                readNBits<32>(nameSize, is);
                std::string inName(nameSize, '\0');
                readNBytes(&inName[0], nameSize, is);
                inputNames.push_back(inName.substr(0, nameSize - 1));
            }
        }
        if (modelHeader.version.minor >= 5) {
            for (int inputIx = 0; inputIx < modelHeader.nTransposeInputs; ++inputIx) {
                std::string inputName;
                std::vector<TranspositionInfo> transpositionInfo;
                ImportTranspositionInfo(is, inputName, transpositionInfo);
                inputsTranspositionInfo[inputName] = transpositionInfo;
            }
            for (int outputIx = 0; outputIx < modelHeader.nTransposeOutputs; ++outputIx) {
                std::string outputName;
                std::vector<TranspositionInfo> transpositionInfo;
                ImportTranspositionInfo(is, outputName, transpositionInfo);
                outputsTranspositionInfo[outputName] = transpositionInfo;
            }
        }
    }
    ImportInputs(is, basePointer, inputsDesc, inputsDataMap);

    if (modelHeader.version.major == 2) {
        if (modelHeader.version.minor >= 3) {
            for (auto inputIndex = 0; inputIndex < modelHeader.nOutputs; inputIndex++) {
                uint32_t nameSize = 0;
                readNBits<32>(nameSize, is);
                std::string outName(nameSize, '\0');
                readNBytes(&outName[0], nameSize, is);
                outputNames.push_back(outName.substr(0, nameSize - 1));
            }
        }
    }
    ImportOutputs(is, basePointer, desc, outputsDataMap);

    for (auto operation = gna2Model->Operations; operation != gna2Model->Operations + gna2Model->NumberOfOperations; ++operation) {
        readNBits<32>(operation->Type, is);
        readBits(operation->NumberOfOperands, is);
        operation->Operands = static_cast<Gna2Tensor const **>(gnaUserAllocator(sizeof(Gna2Tensor*) * operation->NumberOfOperands));
        IE_ASSERT(operation->Operands != nullptr);
        for (uint32_t i = 0; i < operation->NumberOfOperands; i++) {
            Gna2Tensor t{};
            readBits(t, is);
            if (IsEmptyTensor(t)) {
                operation->Operands[i] = nullptr;
            } else {
                operation->Operands[i] = static_cast<Gna2Tensor const *>(gnaUserAllocator(sizeof(Gna2Tensor)));
                t.Data = offsetToPointer(basePointer, reinterpret_cast<uint64_t>(t.Data));
                const_cast<Gna2Tensor&>(*operation->Operands[i]) = t;
            }
        }
        readBits(operation->NumberOfParameters, is);
        switch (operation->Type) {
        case Gna2OperationTypeElementWiseAffine:
        case Gna2OperationTypeFullyConnectedAffine:
        case Gna2OperationTypeConvolution:
        case Gna2OperationTypeCopy:
        case Gna2OperationTypeTransposition:
            break;
        case Gna2OperationTypeRecurrent:
            THROW_GNA_EXCEPTION << "Importing of recurrent operation not supported";
        default:
            THROW_GNA_EXCEPTION << "Importing of unknown GNA operation type(" << operation->Type << ")  not supported";
        }
        if (operation->NumberOfParameters > 0)
            operation->Parameters = static_cast<void **>(gnaUserAllocator(sizeof(void*) * operation->NumberOfParameters));
        else
            operation->Parameters = nullptr;
        for (uint32_t i = 0; i < operation->NumberOfParameters; i++) {
            uint32_t paramSize = 0;
            readBits(paramSize, is);
            IE_ASSERT(operation->Parameters != nullptr);
            if (paramSize == 0) {
                IE_ASSERT(operation->Parameters != nullptr);
                operation->Parameters[i] = nullptr;
                continue;
            }
            operation->Parameters[i] = gnaUserAllocator(paramSize);
            readNBytes(operation->Parameters[i], paramSize, is);

            if (GnaParamSize.at(operation->Type).size() <= i) {
                THROW_GNA_EXCEPTION << "Cannot import parameter of index: " << i;
            }
            if (paramSize != GnaParamSize.at(operation->Type).at(i)) {
                THROW_GNA_EXCEPTION << "Parameter size mismatch on import: " << i;
            }
        }
    }

    // writing memory information
    uint32_t nStates = 0;
    readBits(nStates, is);
    if (pstates != nullptr) {
        pstates->resize(nStates);
    }

    for (int i = 0; i != nStates; i++) {
        void *pSegment;
        if ( modelHeader.version.major == 2 ) {
            if ( modelHeader.version.minor < 6 ) {
                readOffset(pSegment, basePointer, is);
                uint32_t segmentSz = 0;
                readBits(segmentSz, is);
                if (pstates) {
                    (*pstates)[i] = std::make_tuple( pSegment, segmentSz, "noname", 1.0f );
                }
            } else {
                readOffset(pSegment, basePointer, is);
                uint32_t segmentSz = 0;
                readBits(segmentSz, is);
                uint32_t nameSize = 0;
                readNBits<32>(nameSize, is);
                std::string inName(nameSize, '\0');
                readNBytes(&inName[0], nameSize, is);
                float scale_factor = 1.0f;
                readBits(scale_factor, is);
                if (pstates) {
                    (*pstates)[i] = std::make_tuple( pSegment, segmentSz, inName.substr(0, nameSize - 1), scale_factor);
                }
            }
        }
    }


    // once structure has been read lets read whole gna graph
    is.read(reinterpret_cast<char*>(basePointer), gnaGraphSize);
}

void GNAModelSerial::Export(void * basePointer, size_t gnaGraphSize, std::ostream & os) const {
    os.exceptions(std::ostream::failbit);

    const std::vector<Gna2Operation>
        layers(gna2Model->Operations, gna2Model->Operations + gna2Model->NumberOfOperations);


    // all offsets will be from this pointer
    auto getOffsetFromBase = [basePointer, &gnaGraphSize](void * pointer, const char * name = nullptr) {
        auto offset = static_cast<uint64_t>(std::distance(reinterpret_cast<uint8_t*>(basePointer), reinterpret_cast<uint8_t*>(pointer)));
        if (offset > gnaGraphSize) {
            THROW_GNA_EXCEPTION << "offset to " << (name == nullptr ? "" : name) << "(0x" << pointer
                << ") not in range segment retuned from GNAAlloc(0x" << basePointer << "-0x"
                << reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(basePointer) + gnaGraphSize) << ")";
        }
        return offset;
    };

    auto getTensorWithProperOffset = [&getOffsetFromBase](const Gna2Tensor& tensor) {
        Gna2Tensor out = tensor;
        out.Data = reinterpret_cast<void*>(getOffsetFromBase(tensor.Data));
        return out;
    };

    auto convert_to_serial = [getOffsetFromBase](const HeaderLatest::RuntimeEndPoint& ep) {
        HeaderLatest::RuntimeEndPoint out;
        out.elements_count = ep.elements_count;
        out.descriptor_offset = offsetFromBase(ep.descriptor_ptr);
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
    HeaderLatest::ModelHeader header;
    header.gnam[0] = 'G';
    header.gnam[1] = 'N';
    header.gnam[2] = 'A';
    header.gnam[3] = 'M';
    header.headerSize = sizeof(HeaderLatest::ModelHeader);
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
        writeBits(offsetFromBase(gna_ptr), os);
        writeBits(reserved_size, os);
        const auto nameSize = strlen(name.c_str()) + 1;
        writeBits(static_cast<uint32_t>(nameSize), os);
        writeNBytes(name.c_str(), nameSize, os);
        writeBits(scale_factor, os);
    }

    // once structure has been written lets push gna graph
    os.write(reinterpret_cast<char*>(basePointer), gnaGraphSize);
}
#else

void GNAModelSerial::Import(void *basePointer,
        size_t gnaGraphSize,
        std::istream & is,
        std::shared_ptr<GNAPluginNS::InputDesc> inputsDesc,
        std::vector<GNAPluginNS::OutputDesc> &desc,
        InferenceEngine::InputsDataMap& inputsDataMap,
        InferenceEngine::OutputsDataMap& outputsDataMap,
        TranspositionInfoMap& inputsTranspositionInfo,
        TranspositionInfoMap& outputsTranspositionInfo) {
    is.exceptions(std::istream::failbit);

    if (modelHeader.version.major == 2) {
        if (modelHeader.version.minor >= 5) {
            for (int inputIx = 0; inputIx < modelHeader.nTransposeInputs; ++inputIx) {
                std::string inputName;
                std::vector<TranspositionInfo> transpositionInfo;
                ImportTranspositionInfo(is, inputName, transpositionInfo);
                inputsTranspositionInfo[inputName] = transpositionInfo;
            }
            for (int outputIx = 0; outputIx < modelHeader.nTransposeOutputs; ++outputIx) {
                std::string outputName;
                std::vector<TranspositionInfo> transpositionInfo;
                ImportTranspositionInfo(is, outputName, transpositionInfo);
                outputsTranspositionInfo[outputName] = transpositionInfo;
            }
        }
    }
    ImportInputs(is, basePointer, inputsDesc, inputsDataMap);
    ImportOutputs(is, basePointer, desc, outputsDataMap);

    auto readPwl = [&is, basePointer](intel_pwl_func_t & value) {
        readBits(value.nSegments, is);
        if (value.nSegments != 0) {
            readOffset(value.pSegments, basePointer, is);
        } else {
            value.pSegments = nullptr;
        }
    };

    for (auto layer = ptr_nnet->pLayers; layer != ptr_nnet->pLayers + ptr_nnet->nLayers; ++layer) {
        readBits(layer->nInputColumns, is);
        readBits(layer->nInputRows, is);
        readBits(layer->nOutputColumns, is);
        readBits(layer->nOutputRows, is);
        readBits(layer->nBytesPerInput, is);
        readBits(layer->nBytesPerOutput, is);
        readBits(layer->nBytesPerIntermediateOutput, is);
        readNBits<32>(layer->nLayerKind, is);

        // reading layers structs
        switch (layer->nLayerKind) {
        case INTEL_AFFINE_DIAGONAL:
        case INTEL_AFFINE: {
            layer->pLayerStruct = _mm_malloc(sizeof(intel_affine_layer_t), 64);
            if (layer->pLayerStruct == nullptr) {
                THROW_GNA_EXCEPTION << "could not allocate memory for intel_affine_layer_t structure.";
            }

            auto &affine = *reinterpret_cast<intel_affine_layer_t *>(layer->pLayerStruct);
            readBits(affine.affine.nBytesPerWeight, is);
            readBits(affine.affine.nBytesPerBias, is);
            readOffset(affine.affine.pWeights, basePointer, is);
            readOffset(affine.affine.pBiases, basePointer, is);
            readPwl(affine.pwl);
            break;
        }
        case INTEL_CONVOLUTIONAL: {
            layer->pLayerStruct = _mm_malloc(sizeof(intel_convolutional_layer_t), 64);
            if (layer->pLayerStruct == nullptr) {
                THROW_GNA_EXCEPTION << "could not allocate memory for intel_convolutional_layer_t structure.";
            }

            auto &convolution = *reinterpret_cast<intel_convolutional_layer_t *>(layer->pLayerStruct);
            readBits(convolution.nFilterCoefficients, is);
            readBits(convolution.nBytesFilterCoefficient, is);
            readBits(convolution.nBytesBias, is);
            readBits(convolution.nFilters, is);
            readBits(convolution.nFeatureMaps, is);
            readBits(convolution.nFeatureMapRows, is);
            readBits(convolution.nFeatureMapColumns, is);
            readBits(convolution.nFilterRows, is);
            readOffset(convolution.pFilters, basePointer, is);
            readOffset(convolution.pBiases, basePointer, is);
            readBits(convolution.nPoolSize, is);
            readBits(convolution.nPoolStride, is);
            readBits(convolution.poolType, is);
            readPwl(convolution.pwl);
            break;
        }

        case INTEL_COPY: {
            layer->pLayerStruct = _mm_malloc(sizeof(intel_copy_layer_t), 64);
            if (layer->pLayerStruct == nullptr) {
                THROW_GNA_EXCEPTION << "could not allocate memory for intel_copy_layer_t structure.";
            }

            auto &copy = *reinterpret_cast<intel_copy_layer_t *>(layer->pLayerStruct);
            readBits(copy.nCopyRows, is);
            readBits(copy.nCopyCols, is);
            break;
        }

        case INTEL_RECURRENT:
            THROW_GNA_EXCEPTION << "Importing of recurrent layer not supported";
        case INTEL_INTERLEAVE:
            THROW_GNA_EXCEPTION << "Importing of interleave layer not supported";
        case INTEL_DEINTERLEAVE:
            THROW_GNA_EXCEPTION << "Importing of deinterleave layer not supported";
        default:
            THROW_GNA_EXCEPTION << "Importing of unknown GNA layer kind(" << layer->nLayerKind << ")  not supported";
        }

        // reading offsets of inputs/outputs
        readOffset(layer->pInputs, basePointer, is);
        if (layer->nLayerKind == INTEL_COPY) {
            layer->pOutputsIntermediate = nullptr;
        } else {
            readOffset(layer->pOutputsIntermediate, basePointer, is);
        }
        readOffset(layer->pOutputs, basePointer, is);
    }

    // writing memory information
    uint32_t nStates = 0;
    readBits(nStates, is);
    if (pstates != nullptr) {
        pstates->resize(nStates);
    }

    for (int i = 0; i != nStates; i++) {
        void *pSegment;
        if ( modelHeader.version.major == 2 ) {
            if ( modelHeader.version.minor < 6 ) {
                readOffset(pSegment, basePointer, is);
                uint32_t segmentSz = 0;
                readBits(segmentSz, is);
                if (pstates) {
                    (*pstates)[i] = std::make_tuple( pSegment, segmentSz, "noname", 1.0f);
                }
            } else {
                readOffset(pSegment, basePointer, is);
                uint32_t segmentSz = 0;
                readBits(segmentSz, is);
                uint32_t nameSize = 0;
                readNBits<32>(nameSize, is);
                std::string inName(nameSize, '\0');
                readNBytes(&inName[0], nameSize, is);
                float scale_factor = 1.0f;
                readBits(scale_factor, is);
                if (pstates) {
                    (*pstates)[i] = std::make_tuple( pSegment, segmentSz, inName.substr(0, nameSize - 1), scale_factor );
                }
            }
        }
    }


    // once structure has been read lets read whole gna graph
    is.read(reinterpret_cast<char*>(basePointer), gnaGraphSize);
}

/**
 *
 * @param ptr_nnet
 * @param gnaAllocSize - it can be calculated based on nnet, however it will overcomplicate export
 * about base adress it is relatively easy to calculate
 * @param os
 */

void GNAModelSerial::Export(void * basePointer, size_t gnaGraphSize, std::ostream & os) const {
    os.exceptions(std::ostream::failbit);

    std::vector<intel_nnet_layer_t>
        layers(ptr_nnet->pLayers, ptr_nnet->pLayers + ptr_nnet->nLayers);


    // all offsets will be from this pointer
    auto getOffsetFromBase = [basePointer, &gnaGraphSize](void * pointer, const char * name = nullptr) {
        auto offset = static_cast<uint64_t >(std::distance(reinterpret_cast<uint8_t*>(basePointer), reinterpret_cast<uint8_t*>(pointer)));
        if (offset > gnaGraphSize) {
            THROW_GNA_EXCEPTION << "offset to " << (name == nullptr ? "" : name) << "(0x" << pointer
                               << ") not in range segment returned from GNAAlloc(0x" << basePointer << "-0x"
                               << reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(basePointer) + gnaGraphSize) << ")";
        }
        return offset;
    };

    auto writePwl = [&os, getOffsetFromBase] (intel_pwl_func_t & value) {
        writeBits(value.nSegments, os);
        // export require certain offset, since offset from base to nullptr cannot be correct, we are not store it at all
        if (value.nSegments != 0) {
            writeBits(offsetFromBase(value.pSegments), os);
        }
    };

    auto convert_to_serial = [getOffsetFromBase](const HeaderLatest::RuntimeEndPoint& ep){
        HeaderLatest::RuntimeEndPoint out;
        out.elements_count = ep.elements_count;
        out.element_size = ep.element_size;
        out.descriptor_offset = offsetFromBase(ep.descriptor_ptr);
        out.scaleFactor = ep.scaleFactor;
        out.orientation = ep.orientation;
        return out;
    };

    /**
     * writing header
     */
    HeaderLatest::ModelHeader header;
    header.gnam[0] = 'G';
    header.gnam[1] = 'N';
    header.gnam[2] = 'A';
    header.gnam[3] = 'M';
    header.version.major = 1u;
    header.version.minor = 1u;
    header.gnaMemSize = gnaGraphSize;
    header.layersCount = layers.size();
    header.nGroup = ptr_nnet->nGroup;
    header.nInputs = 1;
    header.nOutputs = 1;
    header.headerSize = sizeof(HeaderLatest::ModelHeader);
    header.nTransposeInputs = transposeInputsInfo.size();
    header.nTransposeOutputs = transposeOutputsInfo.size();

    ExportTranspositionInfo(os, transposeInputsInfo);
    ExportTranspositionInfo(os, transposeOutputsInfo);

    writeBits(header, os);
    writeBits(convert_to_serial(inputs[0]), os);
    writeBits(convert_to_serial(outputs[0]), os);

    for (auto & layer : layers) {
        writeBits(layer.nInputColumns, os);
        writeBits(layer.nInputRows, os);
        writeBits(layer.nOutputColumns, os);
        writeBits(layer.nOutputRows, os);
        writeBits(layer.nBytesPerInput, os);
        writeBits(layer.nBytesPerOutput, os);
        writeBits(layer.nBytesPerIntermediateOutput, os);
        writeBits(static_cast<uint32_t>(layer.nLayerKind), os);

        // writing layers structs
        switch (layer.nLayerKind) {
            case INTEL_AFFINE_DIAGONAL:
            case INTEL_AFFINE: {
                auto &affine = *reinterpret_cast<intel_affine_layer_t *>(layer.pLayerStruct);
                writeBits(affine.affine.nBytesPerWeight, os);
                writeBits(affine.affine.nBytesPerBias, os);
                writeBits(offsetFromBase(affine.affine.pWeights), os);
                writeBits(offsetFromBase(affine.affine.pBiases), os);
                writePwl(affine.pwl);
                break;
            }
            case INTEL_CONVOLUTIONAL: {
                auto &convolution = *reinterpret_cast<intel_convolutional_layer_t *>(layer.pLayerStruct);
                writeBits(convolution.nFilterCoefficients, os);
                writeBits(convolution.nBytesFilterCoefficient, os);
                writeBits(convolution.nBytesBias, os);
                writeBits(convolution.nFilters, os);
                writeBits(convolution.nFeatureMaps, os);
                writeBits(convolution.nFeatureMapRows, os);
                writeBits(convolution.nFeatureMapColumns, os);
                writeBits(convolution.nFilterRows, os);
                writeBits(offsetFromBase(convolution.pFilters), os);
                writeBits(offsetFromBase(convolution.pBiases), os);
                writeBits(convolution.nPoolSize, os);
                writeBits(convolution.nPoolStride, os);
                writeBits(convolution.poolType, os);
                writePwl(convolution.pwl);
                break;
            }

            case INTEL_COPY: {
                auto &copy = *reinterpret_cast<intel_copy_layer_t *>(layer.pLayerStruct);
                writeBits(copy.nCopyRows, os);
                writeBits(copy.nCopyCols, os);
                break;
            }

            case INTEL_RECURRENT:
                THROW_GNA_EXCEPTION << "Exporting of recurrent layer not supported";
            case INTEL_INTERLEAVE:
                THROW_GNA_EXCEPTION << "Exporting of interleave layer not supported";
            case INTEL_DEINTERLEAVE:
                THROW_GNA_EXCEPTION << "Exporting of deinterleave layer not supported";
            default:
                THROW_GNA_EXCEPTION << "Exporting of unknown GNA layer kind(" << layer.nLayerKind << ")  not supported";
        }

        // writing offsets from base.
        writeBits(offsetFromBase(layer.pInputs), os);
        if (layer.nLayerKind != INTEL_COPY) {
            writeBits(offsetFromBase(layer.pOutputsIntermediate), os);
        }
        writeBits(offsetFromBase(layer.pOutputs), os);
    }
    // writing memory information
    writeBits(static_cast<uint32_t>(states.size()), os);
    for (auto && state : states) {
        void* gna_ptr = nullptr;
        uint32_t reserved_size = 0;
        std::string name;
        float scale_factor = 1.0f;
        std::tie(gna_ptr, reserved_size, name, scale_factor) = state;
        writeBits(offsetFromBase(gna_ptr), os);
        writeBits(reserved_size, os);
        const auto nameSize = strlen(name.c_str()) + 1;
        writeBits(static_cast<uint32_t>(nameSize), os);
        writeNBytes(name.c_str(), nameSize, os);
        writeBits(scale_factor, os);
    }

    // once structure has been written lets push gna graph
    os.write(reinterpret_cast<char*>(basePointer), gnaGraphSize);
}

#endif

std::vector<HeaderLatest::RuntimeEndPoint> GNAModelSerial::serializeOutputs(const InferenceEngine::OutputsDataMap& outputsDataMap,
        const std::vector<GNAPluginNS::OutputDesc>& outputsDesc) {
    std::vector<HeaderLatest::RuntimeEndPoint> endPoints;
    std::size_t outputIndex = 0;
    for (auto const &output : outputsDataMap) {
        auto outputName = output.first;
        auto outputDims = output.second->getTensorDesc().getDims();
        HeaderLatest::RuntimeEndPoint::Shape outputShape;
        outputShape.NumberOfDimensions = outputDims.size();
        for (size_t i=0; i < outputShape.NumberOfDimensions; ++i) {
            outputShape.Dimensions[i] = static_cast<uint32_t>(outputDims[i]);
        }
        uint32_t elementsCount = static_cast<uint32_t>(InferenceEngine::details::product(outputDims.begin(), outputDims.end()));
        InferenceEngine::Layout outputLayout = output.second->getLayout();
        InferenceEngine::Precision::ePrecision outputPrecision = InferenceEngine::Precision::FP32;
        HeaderLatest::RuntimeEndPoint endPoint(outputsDesc[outputIndex].scale_factor,
                                                 outputsDesc[outputIndex].ptrs[0],
                                                 outputsDesc[outputIndex].num_bytes_per_element,
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

std::vector<HeaderLatest::RuntimeEndPoint> GNAModelSerial::serializeInputs(const InferenceEngine::InputsDataMap& inputsDataMap,
                                                                             std::shared_ptr<GNAPluginNS::InputDesc> inputDesc) {
    std::vector<HeaderLatest::RuntimeEndPoint> endPoints;

    std::size_t inputIndex = 0;
    for (auto const& input : inputsDataMap) {
        auto inputName = input.first;
        auto inputDims = input.second->getTensorDesc().getDims();
        HeaderLatest::RuntimeEndPoint::Shape inputShape;
        inputShape.NumberOfDimensions = inputDims.size();
        for (size_t i=0; i < inputShape.NumberOfDimensions; ++i) {
            inputShape.Dimensions[i] = static_cast<uint32_t>(inputDims[i]);
        }
        double scaleFactor = inputDesc->getScaleFactor(inputIndex);
        std::vector<void *> descriptor_ptr = inputDesc->getPtrInputsGlobal(inputName);
        IE_ASSERT(descriptor_ptr.size() > 0);
        uint32_t element_size = 2u;
        uint32_t elementsCount = static_cast<uint32_t>(InferenceEngine::details::product(inputDims.begin(), inputDims.end()));
        intel_dnn_orientation_t orientation = inputDesc->getOrientation(inputName);
        InferenceEngine::Layout inputLayout = input.second->getLayout();
        InferenceEngine::Precision::ePrecision inputPrecision = InferenceEngine::Precision::FP32;
        HeaderLatest::RuntimeEndPoint endPoint(scaleFactor,
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

void GNAModelSerial::ImportInputs(std::istream &is,
        void* basePtr,
        std::shared_ptr<GNAPluginNS::InputDesc> inputsDesc,
        InferenceEngine::InputsDataMap& dataMap) {
    dataMap.clear();

    for (uint32_t inputIndex = 0; inputIndex < modelHeader.nInputs; inputIndex++) {
        const std::string& name = (modelHeader.version.major == 2 && modelHeader.version.minor >= 3)
                ? inputNames.at(inputIndex) : std::string("input" + std::to_string(inputIndex));

        HeaderLatest::RuntimeEndPoint input = ReadEndPoint(is);
        inputsDesc->getPtrInputsGlobal(name).push_back(reinterpret_cast<float*>(reinterpret_cast<uint8_t *> (basePtr) + input.descriptor_offset));
        inputsDesc->orientation_in[name] = input.orientation;
        inputsDesc->bytes_allocated_for_input[name] = input.element_size * input.elements_count;

        auto inputDims = InferenceEngine::SizeVector();
        for (auto i = 0; i < input.shape.NumberOfDimensions; ++i) {
            inputDims.push_back(input.shape.Dimensions[i]);
        }
        InferenceEngine::Layout inputLayout = static_cast<InferenceEngine::Layout>(input.layout);
        InferenceEngine::Precision inputPresicion = InferenceEngine::Precision(static_cast<InferenceEngine::Precision::ePrecision>(input.precision));
        dataMap[name] = std::make_shared<InferenceEngine::InputInfo>();
        dataMap[name]->setInputData(std::make_shared<InferenceEngine::Data>(name,
                                                            InferenceEngine::TensorDesc(
                                                                    inputPresicion,
                                                                    inputDims,
                                                                    inputLayout)));
        inputsDesc->inputScaleFactors.push_back(input.scaleFactor);
    }
}

void GNAModelSerial::ImportOutputs(std::istream &is,
        void* basePtr,
        std::vector<GNAPluginNS::OutputDesc> &desc,
        InferenceEngine::OutputsDataMap& dataMap) {
    desc.clear();
    dataMap.clear();
    desc.resize(modelHeader.nOutputs);

    for (uint32_t outputIndex = 0; outputIndex < modelHeader.nOutputs; outputIndex++) {
        const std::string& name = (modelHeader.version.major == 2 && modelHeader.version.minor >= 3)
                                  ? outputNames.at(outputIndex) : std::string("output" + std::to_string(outputIndex));

        HeaderLatest::RuntimeEndPoint output = ReadEndPoint(is);
        OutputDesc description;
        description.ptrs.push_back(reinterpret_cast<float*>(reinterpret_cast<uint8_t *> (basePtr) + output.descriptor_offset));
        description.orientation = kDnnInterleavedOrientation;
        description.orientation = output.orientation;
        description.num_bytes_per_element = output.element_size;
        description.scale_factor = output.scaleFactor;

        auto outputDims = InferenceEngine::SizeVector();
        for (auto i = 0; i < output.shape.NumberOfDimensions; ++i) {
            outputDims.push_back(output.shape.Dimensions[i]);
        }
        InferenceEngine::Layout outputLayout = static_cast<InferenceEngine::Layout>(output.layout);
        InferenceEngine::Precision outputPresicion =  InferenceEngine::Precision(static_cast<InferenceEngine::Precision::ePrecision>(output.precision));
        dataMap[name] = std::make_shared<InferenceEngine::Data>(name,
                                                 InferenceEngine::TensorDesc(
                                                         outputPresicion,
                                                         outputDims,
                                                         outputLayout));
        desc.at(outputIndex) = description;
    }
}

void GNAModelSerial::ImportTranspositionInfo(std::istream &is,
        std::string &name,
        std::vector<TranspositionInfo> &transpositionInfo) {
    uint32_t nameSize = 0;
    readNBits<32>(nameSize, is);
    name.resize(nameSize, '\0');
    readNBytes(&name[0], nameSize, is);
    uint32_t transposeFragmentsSize = 0;
    readNBits<32>(transposeFragmentsSize, is);
    for (int rotFragmIx = 0; rotFragmIx < transposeFragmentsSize; ++rotFragmIx) {
        TranspositionInfo fragmentTranspositionInfo;
        readNBytes(&fragmentTranspositionInfo, sizeof(TranspositionInfo), is);
        transpositionInfo.push_back(fragmentTranspositionInfo);
    }
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

void GNAModelSerial::setHeader(HeaderLatest::ModelHeader header) {
    modelHeader = header;
}
