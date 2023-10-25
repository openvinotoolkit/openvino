// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_common.h>

#include <array>
#include <ie_algorithm.hpp>
#include <ie_precision.hpp>
#include <iomanip>
#include <ios>
#include <map>
#include <vector>

#if defined __INTEL_COMPILER || defined _MSC_VER
#    include <malloc.h>
#else
#    include <mm_malloc.h>
#endif

#include "common/versioning.hpp"
#include "gna2_model_helper.hpp"
#include "gna_model_serial.hpp"
#include "gna_plugin.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/core.hpp"
#include "serial/headers/2dot2/gna_model_header.hpp"
#include "serial/headers/2dot5/gna_model_header.hpp"
#include "serial/headers/2dot7/gna_model_header.hpp"
#include "serial/headers/2dot8/gna_model_header.hpp"
#include "serial/headers/2dot9/gna_model_header.hpp"
#include "serial/headers/latest/gna_model_header.hpp"

using namespace ov::intel_gna;

inline void writeNBytes(const void* ptr, uint32_t size, std::ostream& os) {
    os.write(static_cast<const char*>(ptr), size);
}

template <class T>
inline void writeBits(const T& obj, std::ostream& os) {
    os.write(reinterpret_cast<const char*>(&obj), sizeof(T));
}

inline void writeString(const std::string& str, std::ostream& os) {
    const char* c_str = str.c_str();
    const uint32_t str_len = static_cast<uint32_t>(strlen(c_str)) + 1;
    writeBits(str_len, os);
    writeNBytes(c_str, str_len, os);
}

inline void write_pre_processing_model(const std::shared_ptr<ov::Model>& model, std::ostream& os) {
    // allocate buffer for ir.xml
    std::ostringstream xml_buf;
    // allocate buffer for ir.bin
    std::ostringstream bin_buf;

    // serialize IR to stream buffer (.xml + .bin)
    ov::pass::Serialize serializer(xml_buf, bin_buf);
    serializer.run_on_model(model);

    // write IR
    writeString(xml_buf.str(), os);

    // write BIN
    size_t ir_bin_size = bin_buf.str().size();
    writeBits(ir_bin_size, os);
    writeNBytes(bin_buf.str().c_str(), static_cast<uint32_t>(ir_bin_size), os);
}

template <class T>
inline void readBits(T& obj, std::istream& is) {
    is.read(reinterpret_cast<char*>(&obj), sizeof(T));
}

inline void readNBytes(void* ptr, uint32_t size, std::istream& is) {
    is.read(reinterpret_cast<char*>(ptr), size);
}

template <int nBits, class T>
inline void readNBits(T& obj, std::istream& is) {
    std::array<uint8_t, nBits / 8> tmp;
    is.read(reinterpret_cast<char*>(&tmp), nBits / 8);

    obj = *reinterpret_cast<T*>(&tmp.front());
}

inline std::string readString(std::istream& is) {
    uint32_t str_len = 0;
    readNBits<32>(str_len, is);
    std::string str(str_len, '\0');
    readNBytes(&str[0], str_len, is);
    return str.substr(0, str_len - 1);
}

inline void* offsetToPointer(void* const base, uint64_t offset) {
    return reinterpret_cast<uint8_t*>(base) + offset;
}

template <class T>
inline void readOffset(T& ptr, void* base, std::istream& is) {
    uint64_t offset = 0ull;
    readBits(offset, is);
    ptr = reinterpret_cast<T>(offsetToPointer(base, offset));
}

union {
    uint16_t s;
    uint8_t c[2];
} constexpr static LECheck{1};

inline bool is_little_endian() {
    return LECheck.c[0] == 1;
}

void GNAVersionSerializer::Export(std::ostream& os) const {
    writeString(ov::intel_gna::common::get_openvino_version_string(), os);
    writeString(GNADeviceHelper::GetGnaLibraryVersion(), os);
}

std::string GNAVersionSerializer::Import(std::istream& is) const {
    std::string version;
    if (is.peek() && !is.eof()) {
        version = "The model was exported with OpenVINO version:\n" + readString(is) + "\n";
        version += "GNA Library version:\n" + readString(is) + "\n";
    }
    return version;
}

const int gna_header_magic = is_little_endian() ? 0x4d414e47 : 0x474e414d;

header_latest::ModelHeader GNAModelSerial::ReadHeader(std::istream& is) {
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

    header_latest::ModelHeader header;
    header.version.major = 0u;
    header.version.minor = 0u;
    auto size_of_headers_header = sizeof(header_latest::ModelHeader::gnam) +
                                  sizeof(header_latest::ModelHeader::headerSize) +
                                  sizeof(header_latest::ModelHeader::Version);
    if (static_cast<uint32_t>(stream_len) > size_of_headers_header) {
        readNBytes(&header, static_cast<uint32_t>(size_of_headers_header), is);
    } else {
        readNBytes(&header, static_cast<uint32_t>(stream_len), is);
    }
    if (*reinterpret_cast<int*>(header.gnam) != gna_header_magic) {
        THROW_GNA_EXCEPTION << "Imported file unsupported: magic number should be GNAM(0x474e414d), but was 0x"
                            << std::setfill('0') << std::hex << std::setw(2) << static_cast<short>(header.gnam[0])
                            << std::hex << std::setw(2) << static_cast<short>(header.gnam[1]) << std::hex
                            << std::setw(2) << static_cast<short>(header.gnam[2]) << std::hex << std::setw(2)
                            << static_cast<short>(header.gnam[3]);
    }

    is.seekg(startPos, is.beg);
    header_2_dot_1::ModelHeader tempheader_2_dot_1;
    switch (header.version.major) {
    case 2:
        switch (header.version.minor) {
        case 1:
            readBits(tempheader_2_dot_1, is);
            header = header_latest::ModelHeader(tempheader_2_dot_1);
            break;
        case 2:
        case 3: {
            header_2_dot_3::ModelHeader tempheader_2_dot_3;
            readBits(tempheader_2_dot_3, is);
            header = header_latest::ModelHeader(tempheader_2_dot_3);
            break;
        }
        case 4: {
            header_2_dot_4::ModelHeader tempheader_2_dot_4;
            readBits(tempheader_2_dot_4, is);
            header = header_latest::ModelHeader(tempheader_2_dot_4);
            break;
        }
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
            readNBytes(&header, sizeof(header_latest::ModelHeader), is);
            break;
        default:
            THROW_GNA_EXCEPTION
                << "Imported file unsupported. minor version should have values in range 1 to 9 and is: "
                << header.version.minor;
        }
        break;
    default:
        THROW_GNA_EXCEPTION << "Imported file unsupported. Import for files with major version equal to: "
                            << header.version.major << " is not implemented";
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

header_latest::RuntimeEndPoint GNAModelSerial::ReadEndPoint(std::istream& is) {
    is.exceptions(std::istream::failbit);

    header_latest::RuntimeEndPoint endPoint;
    switch (model_header_.version.major) {
    case 2:
        switch (model_header_.version.minor) {
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
        case 6: {
            header_2_dot_6::RuntimeEndPoint tempEndPoint2dot6;
            readBits(tempEndPoint2dot6, is);
            endPoint = header_latest::RuntimeEndPoint(tempEndPoint2dot6, model_header_.nGroup);
            break;
        }
        case 7: {
            header_2_dot_7::RuntimeEndPoint tempEndPoint2dot7;
            readBits(tempEndPoint2dot7, is);
            endPoint = header_latest::RuntimeEndPoint(tempEndPoint2dot7);
            break;
        }
        case 8:
        case 9:
            readNBytes(&endPoint, sizeof(header_latest::RuntimeEndPoint), is);
            break;
        default:
            THROW_GNA_EXCEPTION
                << "Imported file unsupported. minor version should have values in range 1 to 9 and is: "
                << model_header_.version.minor;
        }
        break;
    default:
        THROW_GNA_EXCEPTION << "Imported file unsupported. Import for files with major version equal to: "
                            << model_header_.version.major << " is not implemented";
    }

    return endPoint;
}

#define offsetFromBase(field) getOffsetFromBase(field, #field)

inline bool IsEmptyTensor(const Gna2Tensor& t) {
    return t.Type == Gna2DataTypeNone && t.Data == nullptr && t.Layout[0] == '\0' && t.Mode == Gna2TensorModeDefault &&
           t.Shape.NumberOfDimensions == 0;
}

static const std::map<Gna2OperationType, std::vector<uint32_t>> GnaParamSize{
    {Gna2OperationTypeFullyConnectedAffine, {sizeof(Gna2BiasMode), sizeof(uint32_t)}},
    {Gna2OperationTypeConvolution,
     {sizeof(Gna2Shape),
      sizeof(Gna2BiasMode),
      sizeof(Gna2PoolingMode),
      sizeof(Gna2Shape),
      sizeof(Gna2Shape),
      sizeof(Gna2Shape)}},
    {Gna2OperationTypeCopy, {sizeof(Gna2Shape)}},
    {Gna2OperationTypeTransposition, {sizeof(Gna2Shape)}},
};

void GNAModelSerial::Import(void* basePointer,
                            size_t gnaGraphSize,
                            std::istream& is,
                            GnaInputs& inputs,
                            GnaOutputs& outputs,
                            TranspositionInfoMap& inputsTranspositionInfo,
                            TranspositionInfoMap& outputsTranspositionInfo,
                            std::string& libVersionFromFile) {
    is.exceptions(std::istream::failbit);
    // 2. Read inputs names
    if (model_header_.version.major == 2) {
        for (uint32_t inputIndex = 0; inputIndex < model_header_.nInputs; inputIndex++) {
            std::string name =
                (model_header_.version.minor >= 3) ? readString(is) : std::string("input" + std::to_string(inputIndex));
            inputs[name] = InputDesc(name);
        }
        // Plugin uses ngraph pre/post-processing function to transpose inputs/outputs starting from version 2.9
        if (model_header_.version.minor >= 5 && model_header_.version.minor <= 8) {
            // 3. Read transposition input info
            for (uint32_t inputIx = 0; inputIx < model_header_.nTransposeInputs; ++inputIx) {
                std::string inputName;
                std::vector<TranspositionInfo> transpositionInfo;
                ImportTranspositionInfo(is, inputName, transpositionInfo);
                inputsTranspositionInfo[inputName] = transpositionInfo;
            }
            // 4. Read transposition output info
            for (uint32_t outputIx = 0; outputIx < model_header_.nTransposeOutputs; ++outputIx) {
                std::string outputName;
                std::vector<TranspositionInfo> transpositionInfo;
                ImportTranspositionInfo(is, outputName, transpositionInfo);
                outputsTranspositionInfo[outputName] = transpositionInfo;
            }
        }
    }
    // 5. Read Inputs endpoints
    ImportNodes(is, basePointer, inputs);
    // 6. Read output names
    if (model_header_.version.major == 2) {
        for (uint32_t outputIndex = 0; outputIndex < model_header_.nOutputs; outputIndex++) {
            std::string name = (model_header_.version.minor >= 3) ? readString(is)
                                                                  : std::string("output" + std::to_string(outputIndex));
            outputs[name] = OutputDesc(name);
        }
    }
    // 7. Read outputs
    ImportNodes(is, basePointer, outputs);

    for (auto operation = gna2model_->Operations; operation != gna2model_->Operations + gna2model_->NumberOfOperations;
         ++operation) {
        readNBits<32>(operation->Type, is);
        readBits(operation->NumberOfOperands, is);
        operation->Operands =
            static_cast<Gna2Tensor const**>(gnaUserAllocator(sizeof(Gna2Tensor*) * operation->NumberOfOperands));
        IE_ASSERT(operation->Operands != nullptr);
        for (uint32_t i = 0; i < operation->NumberOfOperands; i++) {
            Gna2Tensor t{};
            readBits(t, is);
            if (IsEmptyTensor(t)) {
                operation->Operands[i] = nullptr;
            } else {
                operation->Operands[i] = static_cast<Gna2Tensor const*>(gnaUserAllocator(sizeof(Gna2Tensor)));
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
            operation->Parameters =
                static_cast<void**>(gnaUserAllocator(sizeof(void*) * operation->NumberOfParameters));
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
    if (pstates_ != nullptr) {
        pstates_->resize(nStates);
    }

    for (uint32_t i = 0; i != nStates; i++) {
        void* pSegment;
        if (model_header_.version.major == 2) {
            if (model_header_.version.minor < 6) {
                readOffset(pSegment, basePointer, is);
                uint32_t segmentSz = 0;
                readBits(segmentSz, is);
                if (pstates_) {
                    (*pstates_)[i] = std::make_tuple(pSegment, segmentSz, "noname", 1.0f);
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
                if (pstates_) {
                    (*pstates_)[i] = std::make_tuple(pSegment, segmentSz, inName.substr(0, nameSize - 1), scale_factor);
                }
            }
        }
    }

    // once structure has been read lets read whole gna graph
    is.read(reinterpret_cast<char*>(basePointer), gnaGraphSize);

    // read OV and GNA versions if available in model file
    if (model_header_.version.major == 2 && model_header_.version.minor >= 8) {
        libVersionFromFile = version_.Import(is);
    }
}

void GNAModelSerial::Export(const GnaAllocations& allocations, std::ostream& os) const {
    os.exceptions(std::ostream::failbit);

    const std::vector<Gna2Operation> layers(gna2model_->Operations,
                                            gna2model_->Operations + gna2model_->NumberOfOperations);

    const auto gnaGraphSize = allocations.GetSizeForExport();
    const auto& allocationsOrdered = allocations.GetAllocationsInExportOrder();

    auto getTensorWithProperOffset = [&allocationsOrdered](const Gna2Tensor& tensor) {
        Gna2Tensor out = tensor;
        const auto found = GnaAllocations::GetOffsetForExport(allocationsOrdered, tensor.Data);
        if (!found.first) {
            THROW_GNA_EXCEPTION << "Tensor data pointer not found in allocations\n";
        }
        out.Data = reinterpret_cast<void*>(found.second);
        return out;
    };

    auto convert_to_serial = [&allocationsOrdered](const GnaDesc& desc) {
        header_latest::RuntimeEndPoint ep;
        ep.elements_count = desc.num_elements;
        ep.scaleFactor = desc.scale_factor;
        ep.element_size = static_cast<uint32_t>(desc.tensor_precision.size());
        ep.layout = desc.model_layout;
        ep.precision = desc.model_precision;
        ep.orientation = desc.orientation;
        ep.tensor_names_count = static_cast<uint8_t>(desc.tensor_names.size());
        const auto found = GnaAllocations::GetOffsetForExport(allocationsOrdered, *desc.ptrs.begin());
        if (!found.first) {
            THROW_GNA_EXCEPTION << "Endpoint data pointer not found in allocations\n";
        }
        ep.descriptor_offset = found.second;
        // shape
        ep.shape.NumberOfDimensions = static_cast<uint32_t>(desc.dims.size());
        for (size_t i = 0; i < ep.shape.NumberOfDimensions; ++i) {
            ep.shape.Dimensions[i] = static_cast<uint32_t>(desc.dims[i]);
        }
        return ep;
    };

    /**
     * writing header
     */
    header_latest::ModelHeader header;
    header.gnam[0] = 'G';
    header.gnam[1] = 'N';
    header.gnam[2] = 'A';
    header.gnam[3] = 'M';
    header.headerSize = sizeof(header_latest::ModelHeader);
    header.gnaMemSize = gnaGraphSize;
    header.layersCount = layers.size();
    header.nGroup = 1;  // just to support the old models
    header.nInputs = static_cast<uint32_t>(inputs_.size());
    header.nOutputs = static_cast<uint32_t>(outputs_.size());
    header.nTransposeInputs = static_cast<uint32_t>(inputs_transpose_info_.size());
    header.nTransposeOutputs = static_cast<uint32_t>(outputs_transpose_info_.size());
    // 1. Write header
    writeBits(header, os);
    // 2. Write input names
    for (const auto& input : inputs_.Get()) {
        // Write the input name
        writeString(input.name, os);
    }
    // 3. Write transposition input info - removed in v.2.9
    // 4. Write transposition output info - removed in v.2.9
    // 5. Write input endpoints and tensor names
    for (const auto& input : inputs_.Get()) {
        // write RuntimeEndPoint
        writeBits(convert_to_serial(input), os);
        // write the input tensor names
        for (const auto& tname : input.tensor_names) {
            writeString(tname, os);
        }
        // write pre-processing model
        if (input.pre_post_process_model) {
            write_pre_processing_model(input.pre_post_process_model, os);
        } else {
            // write empty string to detect  that model is absent during the import
            writeString("", os);
        }
    }
    // 6. Write outputs names
    for (auto& output : outputs_.Get()) {
        // write the output name
        writeString(output.name, os);
    }
    // 7. Write outputs endpoints and tensor names
    for (auto& output : outputs_.Get()) {
        // write RuntimeEndPoint
        writeBits(convert_to_serial(output), os);
        // write the output tensor names
        for (auto& tname : output.tensor_names) {
            writeString(tname, os);
        }

        // write post-processing model
        if (output.pre_post_process_model) {
            write_pre_processing_model(output.pre_post_process_model, os);
        } else {
            // write empty string to detect  that model is absent during the import
            writeString("", os);
        }
    }
    // 8. Write layers
    for (const auto& layer : layers) {
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
    for (auto&& state : states) {
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
        const auto nameSize = static_cast<uint32_t>(strlen(name.c_str()) + 1);
        writeBits(nameSize, os);
        writeNBytes(name.c_str(), nameSize, os);
        writeBits(scale_factor, os);
    }

    // once structure has been written let's push gna graph memory
    for (const auto& a : allocationsOrdered) {
        os.write(reinterpret_cast<char*>(a.ptr), a.sizeForExport());
    }

    // write OV & GNA versions
    version_.Export(os);
}

template <class T>
void GNAModelSerial::ImportNodes(std::istream& is, void* base_ptr, T& nodes) {
    for (auto& node : nodes.Get()) {
        header_latest::RuntimeEndPoint ep = ReadEndPoint(is);

        node.ptrs.push_back(reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(base_ptr) + ep.descriptor_offset));
        node.orientation = ep.orientation;
        node.num_elements = ep.elements_count;
        node.scale_factor = ep.scaleFactor;
        node.model_precision =
            InferenceEngine::Precision(static_cast<InferenceEngine::Precision::ePrecision>(ep.precision));
        node.set_precision(ep.element_size);
        node.model_layout = static_cast<InferenceEngine::Layout>(ep.layout);
        node.allocated_size = node.get_required_size();

        auto inputDims = InferenceEngine::SizeVector();
        for (uint32_t i = 0; i < ep.shape.NumberOfDimensions; ++i) {
            inputDims.push_back(ep.shape.Dimensions[i]);
        }
        node.dims = inputDims;

        // read tensor names
        for (uint8_t tId = 0; tId < ep.tensor_names_count; ++tId) {
            node.tensor_names.insert(readString(is));
        }
        AppendTensorNameIfNeeded(node);

        // read pre-sprocessing model
        if (model_header_.version.major == 2 && model_header_.version.minor >= 9) {
            std::string ir_xml_str = readString(is);
            if (!ir_xml_str.empty()) {
                // read IR bin
                size_t ir_bin_size = 0;
                readBits(ir_bin_size, is);

                ov::Tensor ir_bin_tensor(ov::element::u8, ov::Shape({ir_bin_size}));
                readNBytes(ir_bin_tensor.data(), static_cast<uint32_t>(ir_bin_size), is);

                // restore model
                ov::Core core;
                node.pre_post_process_model = core.read_model(ir_xml_str, ir_bin_tensor);
            }
        }
    }
}

void GNAModelSerial::ImportTranspositionInfo(std::istream& is,
                                             std::string& name,
                                             std::vector<TranspositionInfo>& transpositionInfo) {
    uint32_t nameSize = 0;
    readNBits<32>(nameSize, is);
    name.resize(nameSize, '\0');
    readNBytes(&name[0], nameSize, is);
    uint32_t transposeFragmentsSize = 0;
    readNBits<32>(transposeFragmentsSize, is);
    for (uint32_t rotFragmIx = 0; rotFragmIx < transposeFragmentsSize; ++rotFragmIx) {
        TranspositionInfo fragmentTranspositionInfo;
        readNBytes(&fragmentTranspositionInfo, sizeof(TranspositionInfo), is);
        transpositionInfo.push_back(fragmentTranspositionInfo);
    }
}

void GNAModelSerial::AppendTensorNameIfNeeded(GnaDesc& nodeDesc) const {
    static constexpr header_2_dot_8::ModelHeader::Version kHasTensorNamesVersion;

    if (header_latest::IsFirstVersionLower(model_header_.version, kHasTensorNamesVersion) &&
        nodeDesc.tensor_names.empty()) {
        nodeDesc.tensor_names.insert(nodeDesc.name);
    }
}
