//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bytecode_builder.hpp"
#include "npu_bytecode_utils/bytecode_reader.hpp"
#include "npu_bytecode_utils/span.hpp"
#include "npu_bytecode_utils/version.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace intel_npu::vm;

namespace {

// Builds minimal valid bytecode with the specified version
std::vector<uint8_t> buildBytecodeWithVersion(Version version) {
    return utils::BytecodeBuilder{}
            .setVersion(version)
            .addFunction(utils::BytecodeBuilder::FunctionBuilder("main", /*numGeneralRegisters=*/1,
                                                                 /*paramTypes=*/{},
                                                                 /*resultTypes=*/{},
                                                                 /*isEntrypoint=*/true)
                                 .ret())
            .build();
}

// Checks if bytecode with given version is compatible with the specified VM range
bool isCompatible(Version version, Version vmMin, Version vmMax) {
    auto bytecode = buildBytecodeWithVersion(version);
    const auto bytecodeView = Span<uint8_t>{bytecode.data(), bytecode.size()};
    return BytecodeReader::isVersionSupported(bytecodeView, vmMin, vmMax);
}

// --- Tests with production VM version (1.0.0) ---

TEST(VirtualMachineCompatibilityCheck, ExactVersionMatch) {
    EXPECT_TRUE(isCompatible(Version{1, 0, 0}, Version{1, 0, 0}, Version{1, 0, 0}));
}

TEST(VirtualMachineCompatibilityCheck, PatchVersionIsIgnored) {
    EXPECT_TRUE(isCompatible(Version{1, 0, 5}, Version{1, 0, 0}, Version{1, 0, 0}));
}

TEST(VirtualMachineCompatibilityCheck, HigherMinorVersionRejected) {
    EXPECT_FALSE(isCompatible(Version{1, 1, 0}, Version{1, 0, 0}, Version{1, 0, 0}));
}

TEST(VirtualMachineCompatibilityCheck, HigherMajorVersionRejected) {
    EXPECT_FALSE(isCompatible(Version{2, 0, 0}, Version{1, 0, 0}, Version{1, 0, 0}));
}

TEST(VirtualMachineCompatibilityCheck, LowerMajorVersionRejected) {
    EXPECT_FALSE(isCompatible(Version{0, 9, 0}, Version{1, 0, 0}, Version{1, 0, 0}));
}

TEST(VirtualMachineCompatibilityCheck, ZeroVersionRejected) {
    EXPECT_FALSE(isCompatible(Version{0, 0, 0}, Version{1, 0, 0}, Version{1, 0, 0}));
}

// --- Tests simulating VM version increase ---

TEST(VirtualMachineCompatibilityCheck, IncreasedVmVersionOldBytecodeStillSupported) {
    EXPECT_TRUE(isCompatible(Version{1, 0, 0}, Version{1, 0, 0}, Version{1, 3, 0}));
}

TEST(VirtualMachineCompatibilityCheck, IncreasedVmVersionIntermediateMinorVersionSupported) {
    EXPECT_TRUE(isCompatible(Version{1, 2, 0}, Version{1, 0, 0}, Version{1, 3, 0}));
}

TEST(VirtualMachineCompatibilityCheck, IncreasedVmVersionCurrentVmMinorVersionSupported) {
    EXPECT_TRUE(isCompatible(Version{1, 3, 0}, Version{1, 0, 0}, Version{1, 3, 0}));
}

TEST(VirtualMachineCompatibilityCheck, IncreasedVmVersionFutureMinorVersionRejected) {
    EXPECT_FALSE(isCompatible(Version{1, 4, 0}, Version{1, 0, 0}, Version{1, 3, 0}));
}

TEST(VirtualMachineCompatibilityCheck, IncreasedVmVersionFutureMajorVersionRejected) {
    EXPECT_FALSE(isCompatible(Version{2, 0, 0}, Version{1, 0, 0}, Version{1, 3, 0}));
}

TEST(VirtualMachineCompatibilityCheck, IncreasedVmVersionPatchVersionIgnoredAtMaxMinor) {
    EXPECT_TRUE(isCompatible(Version{1, 3, 99}, Version{1, 0, 0}, Version{1, 3, 0}));
}

// --- Tests simulating backward compatibility floor being raised ---

TEST(VirtualMachineCompatibilityCheck, RaisedMinVersionBytecodeBelowFloorRejected) {
    EXPECT_FALSE(isCompatible(Version{1, 1, 0}, Version{1, 2, 0}, Version{1, 5, 0}));
}

TEST(VirtualMachineCompatibilityCheck, RaisedMinVersionBytecodeAtFloorSupported) {
    EXPECT_TRUE(isCompatible(Version{1, 2, 0}, Version{1, 2, 0}, Version{1, 5, 0}));
}

TEST(VirtualMachineCompatibilityCheck, RaisedMinVersionBytecodeAboveFloorSupported) {
    EXPECT_TRUE(isCompatible(Version{1, 4, 0}, Version{1, 2, 0}, Version{1, 5, 0}));
}

TEST(VirtualMachineCompatibilityCheck, RaisedMinVersionBytecodeAtMaxSupported) {
    EXPECT_TRUE(isCompatible(Version{1, 5, 0}, Version{1, 2, 0}, Version{1, 5, 0}));
}

TEST(VirtualMachineCompatibilityCheck, RaisedMinVersionBytecodeAboveMaxRejected) {
    EXPECT_FALSE(isCompatible(Version{1, 6, 0}, Version{1, 2, 0}, Version{1, 5, 0}));
}

TEST(VirtualMachineCompatibilityCheck, RaisedMinVersionDeprecatedVersionWithHighPatchStillRejected) {
    // Version 1.1.99 is below the floor 1.2.0 regardless of patch
    EXPECT_FALSE(isCompatible(Version{1, 1, 99}, Version{1, 2, 0}, Version{1, 5, 0}));
}

// --- Tests simulating major version bump ---

TEST(VirtualMachineCompatibilityCheck, MajorVersionBumpOldMajorVersionRejected) {
    EXPECT_FALSE(isCompatible(Version{1, 0, 0}, Version{2, 0, 0}, Version{2, 1, 0}));
}

TEST(VirtualMachineCompatibilityCheck, MajorVersionBumpOldMajorHighMinorRejected) {
    EXPECT_FALSE(isCompatible(Version{1, 9, 0}, Version{2, 0, 0}, Version{2, 1, 0}));
}

TEST(VirtualMachineCompatibilityCheck, MajorVersionBumpNewMajorAtMinSupported) {
    EXPECT_TRUE(isCompatible(Version{2, 0, 0}, Version{2, 0, 0}, Version{2, 1, 0}));
}

TEST(VirtualMachineCompatibilityCheck, MajorVersionBumpNewMajorAtMaxSupported) {
    EXPECT_TRUE(isCompatible(Version{2, 1, 0}, Version{2, 0, 0}, Version{2, 1, 0}));
}

TEST(VirtualMachineCompatibilityCheck, MajorVersionBumpNewMajorAboveMaxRejected) {
    EXPECT_FALSE(isCompatible(Version{2, 2, 0}, Version{2, 0, 0}, Version{2, 1, 0}));
}

TEST(VirtualMachineCompatibilityCheck, MajorVersionBumpFutureMajorVersionRejected) {
    EXPECT_FALSE(isCompatible(Version{3, 0, 0}, Version{2, 0, 0}, Version{2, 1, 0}));
}

// --- Tests simulating cross-major backward compatibility ---

TEST(VirtualMachineCompatibilityCheck, CrossMajorBackwardCompatOldMajorAboveFloorSupported) {
    EXPECT_TRUE(isCompatible(Version{1, 7, 0}, Version{1, 5, 0}, Version{2, 2, 0}));
}

TEST(VirtualMachineCompatibilityCheck, CrossMajorBackwardCompatOldMajorAtFloorSupported) {
    EXPECT_TRUE(isCompatible(Version{1, 5, 0}, Version{1, 5, 0}, Version{2, 2, 0}));
}

TEST(VirtualMachineCompatibilityCheck, CrossMajorBackwardCompatOldMajorBelowFloorRejected) {
    EXPECT_FALSE(isCompatible(Version{1, 4, 0}, Version{1, 5, 0}, Version{2, 2, 0}));
}

TEST(VirtualMachineCompatibilityCheck, CrossMajorBackwardCompatNewMajorSupported) {
    EXPECT_TRUE(isCompatible(Version{2, 1, 0}, Version{1, 5, 0}, Version{2, 2, 0}));
}

TEST(VirtualMachineCompatibilityCheck, CrossMajorBackwardCompatNewMajorAtMaxSupported) {
    EXPECT_TRUE(isCompatible(Version{2, 2, 0}, Version{1, 5, 0}, Version{2, 2, 0}));
}

TEST(VirtualMachineCompatibilityCheck, CrossMajorBackwardCompatNewMajorAboveMaxRejected) {
    EXPECT_FALSE(isCompatible(Version{2, 3, 0}, Version{1, 5, 0}, Version{2, 2, 0}));
}

TEST(VirtualMachineCompatibilityCheck, CrossMajorBackwardCompatVeryOldMajorRejected) {
    EXPECT_FALSE(isCompatible(Version{0, 9, 0}, Version{1, 5, 0}, Version{2, 2, 0}));
}

}  // namespace
