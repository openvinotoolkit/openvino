// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npu_bytecode_utils/magic_number.hpp"
#include "npu_bytecode_utils/network_description.hpp"
#include "npu_bytecode_utils/section_header_table.hpp"
#include "npu_bytecode_utils/serialization_utils.hpp"
#include "npu_bytecode_utils/type_section.hpp"
#include "npu_bytecode_utils/version.hpp"
#include "vm_runtime/virtual_machine.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

using namespace ov::intel_npu::vm;

namespace {

SectionHeader makeTypeSectionHeader(const std::vector<uint8_t>& typeSection) {
    details::DataSectionInfo sectionInfo;
    sectionInfo.numData = 1;
    sectionInfo.dataInfos.push_back({/*offset=*/0, /*size=*/static_cast<uint64_t>(typeSection.size())});

    SectionHeader header{};
    header.type = SectionType::TypeSection;
    header.nameIndex = 0;
    header.offset = 0;
    header.size = static_cast<uint64_t>(typeSection.size());
    header.info = std::make_unique<details::DataSectionInfo>(sectionInfo);
    return header;
}

std::vector<uint8_t> makeBytecodeWithDuplicateTypeSections() {
    std::vector<uint8_t> firstTypeSection;
    IntegerType{/*bitWidth=*/32, true}.appendTo(firstTypeSection);

    std::vector<uint8_t> secondTypeSection;
    FloatType{/*bitWidth=*/32, /*format=*/FloatTypeFormat::IEEE754}.appendTo(secondTypeSection);

    SectionHeaderTable sectionHeaderTable;
    sectionHeaderTable.addSectionHeader(makeTypeSectionHeader(firstTypeSection));
    sectionHeaderTable.addSectionHeader(makeTypeSectionHeader(secondTypeSection));
    sectionHeaderTable.computeOffsets();

    std::vector<uint8_t> bytecodeBuffer;
    MagicNumber(MAGIC_NUMBER).appendTo(bytecodeBuffer);
    Version(/*major=*/1, /*minor=*/0, /*patch=*/0).appendTo(bytecodeBuffer);
    sectionHeaderTable.appendTo(bytecodeBuffer);
    bytecodeBuffer.insert(bytecodeBuffer.end(), firstTypeSection.begin(), firstTypeSection.end());
    bytecodeBuffer.insert(bytecodeBuffer.end(), secondTypeSection.begin(), secondTypeSection.end());
    return bytecodeBuffer;
}

template <typename ElementType>
void checkMetadataPrecision(const ElementType& elementType, ov::element::Type expectedPrecision) {
    SCOPED_TRACE(expectedPrecision.get_type_name());
    for (const auto recordKind : {MetadataRecordKind::Input, MetadataRecordKind::Output}) {
        SCOPED_TRACE(static_cast<int>(recordKind));
        std::vector<uint8_t> typeSection;
        elementType.appendTo(typeSection);
        std::vector<uint8_t> stringSection{'t', 'e', 'n', 's', 'o', 'r', 0};
        std::vector<uint8_t> shapeSection;
        appendValueTo(shapeSection, int64_t{2});
        appendValueTo(shapeSection, int64_t{3});

        std::vector<uint8_t> metadataSection;
        appendValueTo(metadataSection, recordKind);
        appendValueTo(metadataSection, uint64_t{0});
        appendValueTo(metadataSection, uint64_t{0});
        appendValueTo(metadataSection, uint64_t{0});
        appendValueTo(metadataSection, uint32_t{0});
        appendValueTo(metadataSection, uint8_t{0});
        appendValueTo(metadataSection, uint8_t{0});
        appendValueTo(metadataSection, uint8_t{0});
        appendValueTo(metadataSection, uint8_t{0});

        SectionHeaderTable sectionHeaderTable;
        const auto addSection = [&](SectionType sectionType, const std::vector<uint8_t>& payload) {
            auto header = makeTypeSectionHeader(payload);
            header.type = sectionType;
            sectionHeaderTable.addSectionHeader(std::move(header));
        };
        addSection(SectionType::TypeSection, typeSection);
        addSection(SectionType::StringSection, stringSection);
        addSection(SectionType::ConstantSection, shapeSection);
        addSection(SectionType::MetadataSection, metadataSection);
        sectionHeaderTable.computeOffsets();

        std::vector<uint8_t> bytecode;
        MagicNumber(MAGIC_NUMBER).appendTo(bytecode);
        Version(1, 0, 0).appendTo(bytecode);
        sectionHeaderTable.appendTo(bytecode);
        for (const auto* payload : {&shapeSection, &stringSection, &typeSection, &metadataSection}) {
            bytecode.insert(bytecode.end(), payload->begin(), payload->end());
        }

        const auto metadata = parseNetworkMetadata(bytecode.data(), bytecode.size());
        ASSERT_TRUE(metadata.has_value());
        const auto& descriptors = recordKind == MetadataRecordKind::Input ? metadata->inputs : metadata->outputs;
        ASSERT_EQ(descriptors.size(), 1);
        EXPECT_EQ(descriptors.front().precision, expectedPrecision);
        EXPECT_EQ(descriptors.front().shapeFromCompiler, ov::PartialShape({2, 3}));
        EXPECT_EQ(descriptors.front().nameFromCompiler, "tensor");
    }
}

}  // namespace

TEST(VirtualMachineParseTest, RejectsDuplicateTypeSections) {
    npu_vm_module* module{nullptr};
    const auto bytecode = makeBytecodeWithDuplicateTypeSections();
    auto status = npu_vm_parse_module(bytecode.data(), static_cast<uint32_t>(bytecode.size()), &module);
    EXPECT_FALSE(status == NPU_VM_SUCCESS);
    if (status == NPU_VM_SUCCESS) {
        npu_vm_destroy_module(module);
    }
}

TEST(VirtualMachineParseTest, MetadataPreservesI4Precision) {
    checkMetadataPrecision(IntegerType{4, true}, ov::element::i4);
}

TEST(VirtualMachineParseTest, MetadataPreservesU4Precision) {
    checkMetadataPrecision(IntegerType{4, false}, ov::element::u4);
}

TEST(VirtualMachineParseTest, MetadataPreservesNF4Precision) {
    checkMetadataPrecision(FloatType{4, FloatTypeFormat::NF4}, ov::element::nf4);
}

TEST(VirtualMachineParseTest, MetadataPreservesF16Precision) {
    checkMetadataPrecision(FloatType{16, FloatTypeFormat::IEEE754}, ov::element::f16);
}

