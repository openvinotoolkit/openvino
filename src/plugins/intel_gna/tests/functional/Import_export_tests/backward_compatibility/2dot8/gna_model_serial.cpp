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
#include "serial/headers/2dot8/gna_model_header.hpp"

#ifdef GNA_DEBUG
#include <ngraph/pass/manager.hpp>
#include "transformations/serialize.hpp"
#endif

using namespace ov::intel_gna::header_2_dot_8;

inline void writeNBytes(const void *ptr, uint32_t size, std::ostream & os) {
    os.write(static_cast<const char*>(ptr), size);
}

template <class T>
inline void writeBits(const T & obj, std::ostream & os) {
    size_t size = sizeof(T);
    os.write(reinterpret_cast<const char *>(&obj), size);
}

inline void writeString(const std::string &str, std::ostream &os) {
    const char *c_str = str.c_str();
    const size_t str_len = strlen(c_str) + 1;
    writeBits(static_cast<uint32_t>(str_len), os);
    writeNBytes(c_str, str_len, os);
}

void GNAVersionSerializer::Export(std::ostream& os) const {
    writeString(ov::intel_gna::common::get_openvino_version_string(), os);
    writeString(GNADeviceHelper::GetGnaLibraryVersion(), os);
}

static const std::map<Gna2OperationType, std::vector<uint32_t>> GnaParamSize{
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
        layers(gna2model_->Operations, gna2model_->Operations + gna2model_->NumberOfOperations);

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
        ov::intel_gna::header_2_dot_8::RuntimeEndPoint ep;
        ep.elements_count = desc.num_elements;
        ep.scaleFactor = desc.scale_factor;
        ep.element_size = desc.tensor_precision.size();
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
        ep.shape.NumberOfDimensions = desc.dims.size();
        for (size_t i=0; i < ep.shape.NumberOfDimensions; ++i) {
            ep.shape.Dimensions[i] = desc.dims[i];
        }
        return ep;
    };

    /**
     * writing header
     */
    ov::intel_gna::header_2_dot_8::ModelHeader header;
    header.gnam[0] = 'G';
    header.gnam[1] = 'N';
    header.gnam[2] = 'A';
    header.gnam[3] = 'M';
    header.headerSize = sizeof(ov::intel_gna::header_2_dot_8::ModelHeader);
    header.gnaMemSize = gnaGraphSize;
    header.layersCount = layers.size();
    header.nGroup = 1; // just to support the old models
    header.nInputs = inputs_.size();
    header.nOutputs = outputs_.size();
    header.nTransposeInputs = inputs_transpose_info_.size();
    header.nTransposeOutputs = outputs_transpose_info_.size();
    // 1. Write header
    writeBits(header, os);
    // 2. Write input names
    for (const auto &input : inputs_.Get()) {
        // Write the input name
        writeString(input.name, os);
    }
    // 3. Write transposition input info
    ExportTranspositionInfo(os, inputs_transpose_info_);
    // 4. Write transposition output info
    ExportTranspositionInfo(os, outputs_transpose_info_);
    // 5. Write input endpoints and tensor names
    for (const auto &input : inputs_.Get()) {
        // write RuntimeEndPoint
        writeBits(convert_to_serial(input), os);
        // write the input tensor names
        for (const auto &tname : input.tensor_names) {
            writeString(tname, os);
        }
    }
    // 6. Write outputs names
    for (auto &output : outputs_.Get()) {
        // write the output name
        writeString(output.name, os);
    }
    // 7. Write outputs endpoints and tensor names
    for (auto &output : outputs_.Get()) {
        // write RuntimeEndPoint
        writeBits(convert_to_serial(output), os);
        // write the output tensor names
        for (auto &tname : output.tensor_names) {
            writeString(tname, os);
        }
    }
    // 8. Write layers
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

    // once structure has been written let's push gna graph memory
    for (const auto& a : allocationsOrdered) {
        os.write(reinterpret_cast<char*>(a.ptr), a.sizeForExport());
    }

    // write OV & GNA versions
    version_.Export(os);
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