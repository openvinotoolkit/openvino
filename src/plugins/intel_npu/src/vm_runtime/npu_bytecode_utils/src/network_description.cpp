//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npu_bytecode_utils/network_description.hpp"
#include "npu_bytecode_utils/bytecode_reader.hpp"
#include "npu_bytecode_utils/except.hpp"
#include "npu_bytecode_utils/logger.hpp"
#include "npu_bytecode_utils/section_header_table.hpp"
#include "npu_bytecode_utils/serialization_utils.hpp"
#include "npu_bytecode_utils/span.hpp"
#include "npu_bytecode_utils/type_section.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <vector>

#include <openvino/core/dimension.hpp>
#include <openvino/core/partial_shape.hpp>
#include <openvino/core/type/element_type.hpp>

namespace intel_npu::vm {
void NetworkMetadata::bindRelatedDescriptors() {
    size_t ioIndex = 0;

    for (IODescriptor& input : inputs) {
        if (input.relatedDescriptorIndex.has_value()) {
            ++ioIndex;
            continue;
        }

        if (input.isStateInput) {
            const auto relatedDescriptorIterator =
                    std::find_if(outputs.begin(), outputs.end(), [&](const IODescriptor& output) {
                        return output.isStateOutput && (output.nameFromCompiler == input.nameFromCompiler);
                    });

            if (relatedDescriptorIterator != outputs.end()) {
                input.relatedDescriptorIndex = std::distance(outputs.begin(), relatedDescriptorIterator);
                outputs.at(*input.relatedDescriptorIndex).relatedDescriptorIndex = ioIndex;
            }
        } else if (input.isShapeTensor) {
            const auto relatedDescriptorIterator =
                    std::find_if(inputs.begin(), inputs.end(), [&](const IODescriptor& candidate) {
                        return !candidate.isShapeTensor && (candidate.nameFromCompiler == input.nameFromCompiler);
                    });

            if (relatedDescriptorIterator != inputs.end()) {
                input.relatedDescriptorIndex = std::distance(inputs.begin(), relatedDescriptorIterator);
                inputs.at(*input.relatedDescriptorIndex).relatedDescriptorIndex = ioIndex;
            }
        }

        ++ioIndex;
    }

    ioIndex = 0;

    for (IODescriptor& output : outputs) {
        if (output.relatedDescriptorIndex.has_value()) {
            ++ioIndex;
            continue;
        }

        if (output.isShapeTensor) {
            const auto relatedDescriptorIterator =
                    std::find_if(outputs.begin(), outputs.end(), [&](const IODescriptor& candidate) {
                        return !candidate.isShapeTensor && (candidate.nameFromCompiler == output.nameFromCompiler);
                    });

            if (relatedDescriptorIterator != outputs.end()) {
                output.relatedDescriptorIndex = std::distance(outputs.begin(), relatedDescriptorIterator);
                outputs.at(*output.relatedDescriptorIndex).relatedDescriptorIndex = ioIndex;
            }
        }

        ++ioIndex;
    }
}

namespace {
ov::element::Type parsePrecision(const BytecodeReader& reader, size_t typeIndex) {
    constexpr uint8_t int8Bitwidth = 8;
    constexpr uint8_t int16Bitwidth = 16;
    constexpr uint8_t int32Bitwidth = 32;
    constexpr uint8_t int64Bitwidth = 64;
    constexpr uint8_t fp16Bitwidth = 16;
    constexpr uint8_t fp32Bitwidth = 32;
    constexpr uint8_t fp64Bitwidth = 64;

    auto typeEntry = reader.getDataSectionEntry(SectionType::TypeSection, typeIndex);
    NPU_VM_THROW_WHEN(typeEntry.empty(), "Type entry at index {} is not found", typeIndex);

    TypeCode typeCode = TypeCode::OPAQUE;
    NPU_VM_THROW_UNLESS(parseValueFrom(typeEntry, typeCode), "Failed to parse type code for type index {}", typeIndex);

    switch (typeCode) {
    case TypeCode::FLOAT: {
        FloatType floatType{};
        NPU_VM_THROW_UNLESS(floatType.parseFrom(typeEntry), "Failed to parse float type for type index {}", typeIndex);

        // Map float types using both bitwidth and format
        if (floatType.bitWidth == fp16Bitwidth) {
            switch (floatType.format) {
            case FloatTypeFormat::IEEE754:
                return ov::element::f16;
            case FloatTypeFormat::BFloat:
                return ov::element::bf16;
            case FloatTypeFormat::TFloat:
            case FloatTypeFormat::E4M3:
            case FloatTypeFormat::E5M2:
            case FloatTypeFormat::E2M1:
            case FloatTypeFormat::E8M0:
            case FloatTypeFormat::NF4:
                // Extended formats not yet supported in OV element types; preserve exact format awareness
                // by returning dynamic for now. This prevents silent misinterpretation.
                NPU_VM_LOG_WARN("Type index {}: extended float format {} not yet supported in OV element types, "
                                "falling back to dynamic",
                                typeIndex, static_cast<int>(floatType.format));
                return ov::element::dynamic;
            }
        }
        if (floatType.bitWidth == fp32Bitwidth) {
            return ov::element::f32;
        }
        if (floatType.bitWidth == fp64Bitwidth) {
            return ov::element::f64;
        }
        NPU_VM_LOG_WARN("Type index {}: unsupported float bitwidth {}, falling back to dynamic", typeIndex,
                        floatType.bitWidth);
        return ov::element::dynamic;
    }
    case TypeCode::INTEGER: {
        IntegerType intType{};
        NPU_VM_THROW_UNLESS(intType.parseFrom(typeEntry), "Failed to parse integer type for type index {}", typeIndex);

        // Map common integer bit-widths. NOTE: signedness (signed vs unsigned) is NOT preserved in bytecode
        // metadata; only bitWidth is stored during serialization. We use conservative signed mappings
        // (i8, i16, i32, i64). If the actual signedness is critical, it must be inferred from context
        // (e.g., operations using the value) or preserved separately.
        switch (intType.bitWidth) {
        case int8Bitwidth:
            return (intType.isSigned) ? ov::element::i8 : ov::element::u8;
        case int16Bitwidth:
            return (intType.isSigned) ? ov::element::i16 : ov::element::u16;
        case int32Bitwidth:
            return (intType.isSigned) ? ov::element::i32 : ov::element::u32;
        case int64Bitwidth:
            return (intType.isSigned) ? ov::element::i64 : ov::element::u64;
        default:
            NPU_VM_LOG_WARN("Type index {}: unsupported integer bitwidth {}, signed {}, falling back to dynamic",
                            typeIndex, intType.bitWidth, intType.isSigned);
            return ov::element::dynamic;
        }
    }
    case TypeCode::OPAQUE:
    case TypeCode::BUFFER:
    case TypeCode::FUNCTION:
        // OPAQUE, BUFFER, and FUNCTION types lack well-defined OV element mappings.
        NPU_VM_LOG_WARN("Type index {}: type code {} lacks well-defined OV element mapping, falling back to dynamic",
                        typeIndex, static_cast<int>(typeCode));
        return ov::element::dynamic;
    }

    return ov::element::dynamic;
}

ov::PartialShape parseShape(const BytecodeReader& reader, size_t shapeIndex) {
    auto shapeEntry = reader.getDataSectionEntry(SectionType::ConstantSection, shapeIndex);
    NPU_VM_THROW_WHEN(shapeEntry.empty(), "Shape entry at index {} is not found", shapeIndex);

    NPU_VM_THROW_UNLESS(shapeEntry.size() % sizeof(int64_t) == 0,
                        "Shape entry at index {} has invalid size {}, expected multiple of {}", shapeIndex,
                        shapeEntry.size(), sizeof(int64_t));

    const auto numDims = shapeEntry.size() / sizeof(int64_t);
    ov::PartialShape shape;
    shape.reserve(numDims);

    auto isDynamicDim = [](int64_t value) {
        constexpr std::array<int64_t, 2> dynamicDimMarkers = {
                std::numeric_limits<int64_t>::min(),  // Legacy marker
                -1                                    // MLIR sentinel (mlir::ShapedType::kDynamic)
        };
        return std::find(dynamicDimMarkers.begin(), dynamicDimMarkers.end(), value) != dynamicDimMarkers.end();
    };

    for (size_t i = 0; i < numDims; ++i) {
        int64_t dimValue = 0;
        NPU_VM_THROW_UNLESS(parseValueFrom(shapeEntry, dimValue),
                            "Failed to parse shape dimension {} for shape entry {}", i, shapeIndex);
        if (isDynamicDim(dimValue)) {
            shape.push_back(ov::Dimension::dynamic());
        } else {
            shape.push_back(ov::Dimension(dimValue));
        }
    }

    return shape;
}

IODescriptor parseDescriptor(const BytecodeReader& reader, Span<uint8_t>& entry) {
    uint64_t nameIndex = 0;
    uint64_t typeIndex = 0;
    uint64_t shapeIndex = 0;
    uint32_t indexUsedByDriver = 0;
    uint8_t hasDynamicStrides = 0;
    uint8_t tensorNameCount = 0;
    uint8_t hasShapeFromIRModel = 0;
    uint64_t shapeFromIRModelIndex = 0;
    uint8_t hasNodeFriendlyName = 0;
    uint64_t nodeFriendlyNameIndex = 0;

    NPU_VM_THROW_UNLESS(parseValueFrom(entry, nameIndex), "Failed to parse descriptor name index");
    NPU_VM_THROW_UNLESS(parseValueFrom(entry, typeIndex), "Failed to parse descriptor type index");
    NPU_VM_THROW_UNLESS(parseValueFrom(entry, shapeIndex), "Failed to parse descriptor shape index");
    NPU_VM_THROW_UNLESS(parseValueFrom(entry, indexUsedByDriver), "Failed to parse descriptor index used by driver");
    NPU_VM_THROW_UNLESS(parseValueFrom(entry, hasDynamicStrides), "Failed to parse descriptor dynamic strides flag");
    NPU_VM_THROW_UNLESS(parseValueFrom(entry, tensorNameCount), "Failed to parse descriptor tensor name count");
    auto name = reader.getString(nameIndex);
    NPU_VM_THROW_UNLESS(name.has_value(), "String entry for descriptor name at index {} is not found", nameIndex);

    IODescriptor descriptor;
    descriptor.nameFromCompiler = name.value();
    descriptor.precision = parsePrecision(reader, typeIndex);
    descriptor.shapeFromCompiler = parseShape(reader, shapeIndex);
    descriptor.indexUsedByDriver = indexUsedByDriver;
    descriptor.supportsStridedLayout = hasDynamicStrides != 0;

    for (uint8_t i = 0; i < tensorNameCount; ++i) {
        uint64_t tensorNameIndex = 0;
        NPU_VM_THROW_UNLESS(parseValueFrom(entry, tensorNameIndex),
                            "Failed to parse tensor name index {} for descriptor", i);
        auto tensorName = reader.getString(tensorNameIndex);
        NPU_VM_THROW_UNLESS(tensorName.has_value(), "String entry for tensor name at index {} is not found",
                            tensorNameIndex);
        descriptor.outputTensorNames.insert(tensorName.value());
    }

    NPU_VM_THROW_UNLESS(parseValueFrom(entry, hasShapeFromIRModel), "Failed to parse hasShapeFromIRModel");
    if (hasShapeFromIRModel != 0) {
        NPU_VM_THROW_UNLESS(parseValueFrom(entry, shapeFromIRModelIndex), "Failed to parse shape from IR model index");
        descriptor.shapeFromIRModel = parseShape(reader, shapeFromIRModelIndex);
    }

    NPU_VM_THROW_UNLESS(parseValueFrom(entry, hasNodeFriendlyName), "Failed to parse hasNodeFriendlyName");
    if (hasNodeFriendlyName != 0) {
        NPU_VM_THROW_UNLESS(parseValueFrom(entry, nodeFriendlyNameIndex),
                            "Failed to parse descriptor friendly name index");
        auto nodeFriendlyName = reader.getString(nodeFriendlyNameIndex);
        NPU_VM_THROW_UNLESS(nodeFriendlyName.has_value(), "String entry for descriptor name at index {} is not found",
                            nodeFriendlyNameIndex);
        descriptor.nodeFriendlyName = nodeFriendlyName.value();
    }

    return descriptor;
}

const details::DataSectionInfo* getMetadataSectionInfo(BytecodeReader& reader) {
    const auto& headers = reader.getSectionHeaderTable().getSectionHeaders();
    for (const auto& header : headers) {
        if (header.type == SectionType::MetadataSection) {
            return dynamic_cast<const details::DataSectionInfo*>(header.info.get());
        }
    }
    return nullptr;
}

void dumpDescriptors([[maybe_unused]] const char* kind, const std::vector<IODescriptor>& descriptors) {
    NPU_VM_LOG_TRACE("{} count: {}", kind, descriptors.size());
    [[maybe_unused]] size_t index = 0;
    for ([[maybe_unused]] const auto& descriptor : descriptors) {
        NPU_VM_LOG_TRACE("{}[{}] name='{}', precision={}, shape={}, driver_index={}, "
                         "dynamic_strides={}, state_in={}, state_out={}, shape_tensor={}",
                         kind, index, descriptor.nameFromCompiler, descriptor.precision.get_type_name(),
                         descriptor.shapeFromCompiler.to_string(), descriptor.indexUsedByDriver,
                         descriptor.supportsStridedLayout ? "true" : "false",
                         descriptor.isStateInput ? "true" : "false", descriptor.isStateOutput ? "true" : "false",
                         descriptor.isShapeTensor ? "true" : "false");
        ++index;
    }
}

void dumpNetworkMetadata(const NetworkMetadata& network) {
    NPU_VM_LOG_TRACE("deserialized metadata: name='{}', numStreams={}, numCmdLists={}", network.name,
                     network.numStreams, network.numCmdLists);
    dumpDescriptors("input", network.inputs);
    dumpDescriptors("output", network.outputs);
    dumpDescriptors("profiling_output", network.profilingOutputs);
}

}  // namespace

std::optional<NetworkMetadata> parseNetworkMetadata(const uint8_t* bytecodePtr, size_t bytecodeSize) {
    if (bytecodePtr == nullptr || bytecodeSize == 0) {
        return std::nullopt;
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast) - function receives pointer to const
    Span<uint8_t> bytecodeView(const_cast<uint8_t*>(bytecodePtr), bytecodeSize);
    BytecodeReader reader(bytecodeView);
    if (!reader.parseFile()) {
        return std::nullopt;
    }
    return parseNetworkMetadata(reader);
}

std::optional<NetworkMetadata> parseNetworkMetadata(BytecodeReader& reader) {
    NetworkMetadata network{};

    const auto* metadataInfo = getMetadataSectionInfo(reader);
    if (metadataInfo == nullptr) {
        NPU_VM_LOG_DEBUG("Bytecode blob does not contain metadata section");
        return std::nullopt;
    }

    for (size_t i = 0; i < metadataInfo->dataInfos.size(); ++i) {
        auto entry = reader.getDataSectionEntry(SectionType::MetadataSection, i);
        if (entry.empty()) {
            NPU_VM_LOG_WARN("Metadata entry at index {} is empty", i);
            return std::nullopt;
        }

        MetadataRecordKind recordKind = MetadataRecordKind::Network;
        if (!parseValueFrom(entry, recordKind)) {
            NPU_VM_LOG_WARN("Failed to parse metadata record kind for entry {}", i);
            return std::nullopt;
        }

        switch (recordKind) {
        case MetadataRecordKind::Network: {
            uint64_t nameIndex = 0;
            uint64_t numStreams = 0;
            uint64_t numCmdLists = 0;
            if (!parseValueFrom(entry, nameIndex)) {
                NPU_VM_LOG_WARN("Failed to parse network metadata name field for entry {}", i);
                return std::nullopt;
            }
            if (!parseValueFrom(entry, numStreams)) {
                NPU_VM_LOG_WARN("Failed to parse network metadata numStreams field for entry {}", i);
                return std::nullopt;
            }
            if (!parseValueFrom(entry, numCmdLists)) {
                NPU_VM_LOG_WARN("Failed to parse network metadata numCmdLists field for entry {}", i);
                return std::nullopt;
            }

            auto name = reader.getString(nameIndex);
            if (!name.has_value()) {
                NPU_VM_LOG_WARN("String entry for network name at index {} is not found", nameIndex);
                return std::nullopt;
            }
            network.name = name.value();
            network.numStreams = static_cast<size_t>(numStreams);
            network.numCmdLists = static_cast<size_t>(numCmdLists);
            break;
        }
        case MetadataRecordKind::Input:
            network.inputs.push_back(parseDescriptor(reader, entry));
            break;
        case MetadataRecordKind::Output:
            network.outputs.push_back(parseDescriptor(reader, entry));
            break;
        case MetadataRecordKind::ProfilingOutput:
            network.profilingOutputs.push_back(parseDescriptor(reader, entry));
            break;
        default:
            NPU_VM_LOG_WARN("Unknown metadata record kind {} for entry {}", static_cast<uint8_t>(recordKind), i);
            return std::nullopt;
        }

        if (!entry.empty()) {
            NPU_VM_LOG_WARN("Trailing bytes are present in metadata entry {} after parsing expected fields", i);
            return std::nullopt;
        }
    }

    network.bindRelatedDescriptors();
    dumpNetworkMetadata(network);
    return network;
}

void printMetadataEntry(Span<uint8_t> metadata) {
    if (metadata.empty()) {
        std::cout << "(empty)";
        return;
    }

    MetadataRecordKind kind{};
    if (!parseValueFrom(metadata, kind)) {
        NPU_VM_LOG_ERROR("Failed to parse metadata record kind");
        return;
    }

    switch (kind) {
    case MetadataRecordKind::Network: {
        uint64_t nameIndex = 0;
        uint64_t numStreams = 0;
        uint64_t numCmdlists = 0;
        if (!parseValueFrom(metadata, nameIndex) || !parseValueFrom(metadata, numStreams) ||
            !parseValueFrom(metadata, numCmdlists)) {
            NPU_VM_LOG_ERROR("Failed to parse network_metadata fields");
            return;
        }
        std::cout << "networkMetadata<nameIndex=" << nameIndex << ", numStreams=" << numStreams
                  << ", numCmdlists=" << numCmdlists << ">";
        break;
    }
    case MetadataRecordKind::Input:
    case MetadataRecordKind::Output:
    case MetadataRecordKind::ProfilingOutput: {
        uint64_t nameIndex = 0;
        uint64_t typeIndex = 0;
        uint64_t shapeIndex = 0;
        uint32_t indexUsedByDriver = 0;
        uint8_t hasDynamicStrides = 0;
        uint8_t tensorNameCount = 0;
        if (!parseValueFrom(metadata, nameIndex) || !parseValueFrom(metadata, typeIndex) ||
            !parseValueFrom(metadata, shapeIndex) || !parseValueFrom(metadata, indexUsedByDriver) ||
            !parseValueFrom(metadata, hasDynamicStrides) || !parseValueFrom(metadata, tensorNameCount)) {
            NPU_VM_LOG_ERROR("Failed to parse metadata descriptor fields");
            return;
        }

        const char* kindStr = (kind == MetadataRecordKind::Input)    ? "inputMetadata"
                              : (kind == MetadataRecordKind::Output) ? "outputMetadata"
                                                                     : "profilingOutputMetadata";
        std::cout << kindStr << "<nameIndex=" << nameIndex << ", typeIndex=" << typeIndex
                  << ", shapeIndex=" << shapeIndex << ", driverIdx=" << indexUsedByDriver
                  << ", dynamicStrides=" << (hasDynamicStrides ? "true" : "false")
                  << ", tensorNameCount=" << static_cast<uint32_t>(tensorNameCount);

        for (uint8_t i = 0; i < tensorNameCount; ++i) {
            uint64_t tensorNameIndex = 0;
            if (!parseValueFrom(metadata, tensorNameIndex)) {
                NPU_VM_LOG_ERROR("Failed to parse tensor name index at position {}", static_cast<uint32_t>(i));
                return;
            }
            std::cout << ", tensorName[" << static_cast<uint32_t>(i) << "]=" << tensorNameIndex;
        }

        uint8_t hasShapeFromIRModel = 0;
        if (!parseValueFrom(metadata, hasShapeFromIRModel)) {
            NPU_VM_LOG_ERROR("Failed to parse hasShapeFromIRModel field");
            return;
        }
        std::cout << ", hasShapeFromIRModel=" << (hasShapeFromIRModel ? "true" : "false");
        if (hasShapeFromIRModel) {
            uint64_t shapeFromIRModelIndex = 0;
            if (!parseValueFrom(metadata, shapeFromIRModelIndex)) {
                NPU_VM_LOG_ERROR("Failed to parse shapeFromIRModelIndex field");
                return;
            }
            std::cout << ", shapeFromIRModelIndex=" << shapeFromIRModelIndex;
        }

        uint8_t hasNodeFriendlyName = 0;
        if (!parseValueFrom(metadata, hasNodeFriendlyName)) {
            NPU_VM_LOG_ERROR("Failed to parse hasNodeFriendlyName field");
            return;
        }
        std::cout << ", hasNodeFriendlyName=" << (hasNodeFriendlyName ? "true" : "false");
        if (hasNodeFriendlyName) {
            uint64_t nodeFriendlyNameIndex = 0;
            if (!parseValueFrom(metadata, nodeFriendlyNameIndex)) {
                NPU_VM_LOG_ERROR("Failed to parse nodeFriendlyNameIndex field");
                return;
            }
            std::cout << ", nodeFriendlyNameIndex=" << nodeFriendlyNameIndex;
        }

        std::cout << ">";
        break;
    }
    default:
        NPU_VM_LOG_ERROR("Unknown metadata kind: {}", static_cast<uint8_t>(kind));
        break;
    }
}

}  // namespace intel_npu::vm
