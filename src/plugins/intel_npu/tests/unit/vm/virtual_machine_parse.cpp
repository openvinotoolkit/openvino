//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npu_bytecode_utils/magic_number.hpp"
#include "npu_bytecode_utils/section_header_table.hpp"
#include "npu_bytecode_utils/type_section.hpp"
#include "npu_bytecode_utils/version.hpp"
#include "vm_runtime/virtual_machine.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

using namespace intel_npu::vm;

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

}  // namespace

TEST(VirtualMachineParseTest, RejectsDuplicateTypeSections) {
    npu_vm_module* module{nullptr};
    const auto bytecode = makeBytecodeWithDuplicateTypeSections();
    auto status = npu_vm_parse_module(bytecode.data(), bytecode.size(), &module);
    EXPECT_FALSE(status == NPU_VM_SUCCESS);
    if (status == NPU_VM_SUCCESS) {
        npu_vm_destroy_module(module);
    }
}
