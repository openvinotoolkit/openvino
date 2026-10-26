//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npu_bytecode_utils/except.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

namespace {

TEST(VirtualMachineExceptTest, ThrowAlwaysThrowsRuntimeError) {
    EXPECT_THROW(NPU_VM_THROW("error happened"), std::runtime_error);
}

TEST(VirtualMachineExceptTest, ThrowFormatsMessageWithPlaceholders) {
    try {
        NPU_VM_THROW("parse failed at {}:{}", "file.vm", 42);
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error& ex) {
        EXPECT_THAT(ex.what(), testing::HasSubstr("parse failed at file.vm:42"));
    }
}

TEST(VirtualMachineExceptTest, ThrowAppendsExtraArgumentsWithoutPlaceholders) {
    try {
        NPU_VM_THROW("bad input", 7, "token");
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error& ex) {
        EXPECT_THAT(ex.what(), testing::HasSubstr("bad input 7 token"));
    }
}

TEST(VirtualMachineExceptTest, ThrowKeepsUnmatchedPlaceholders) {
    try {
        NPU_VM_THROW("missing {} {}", "arg");
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error& ex) {
        EXPECT_THAT(ex.what(), testing::HasSubstr("missing arg {}"));
    }
}

TEST(VirtualMachineExceptTest, ThrowConvertsNullCStringToEmptyText) {
    const char* text = nullptr;

    try {
        NPU_VM_THROW("null='{}'", text);
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error& ex) {
        EXPECT_THAT(ex.what(), testing::HasSubstr("null=''"));
    }
}

TEST(VirtualMachineExceptTest, ThrowWhenThrowsWhenConditionIsTrue) {
    EXPECT_THROW(NPU_VM_THROW_WHEN(true, "condition met: {}", 3), std::runtime_error);
}

TEST(VirtualMachineExceptTest, ThrowWhenDoesNotThrowWhenConditionIsFalse) {
    EXPECT_NO_THROW(NPU_VM_THROW_WHEN(false, "should not throw"));
}

TEST(VirtualMachineExceptTest, ThrowUnlessThrowsWhenConditionIsFalse) {
    EXPECT_THROW(NPU_VM_THROW_UNLESS(false, "condition failed: {}", "x"), std::runtime_error);
}

TEST(VirtualMachineExceptTest, ThrowUnlessDoesNotThrowWhenConditionIsTrue) {
    EXPECT_NO_THROW(NPU_VM_THROW_UNLESS(true, "should not throw"));
}

}  // namespace
