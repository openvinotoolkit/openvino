// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <exception>
#include <memory>

#include "intel_npu/utils/vcl/vcl_api.hpp"

namespace {

using ::intel_npu::VCLFunctionTable;
using ::intel_npu::VCLLoaderHolder;

TEST(VCLLoaderHolderFuncTests, Functions_CalledTwice_ReturnsTheSameTable) {
    VCLLoaderHolder holder;
    std::shared_ptr<const VCLFunctionTable> first;
    try {
        first = holder.functions();
    } catch (const std::exception&) {
        GTEST_SKIP() << "Couldn't load compiler library";
    }
    EXPECT_EQ(first, holder.functions());
}

TEST(VCLLoaderHolderFuncTests, Functions_AfterHolderDestroyed_TableStaysUsable) {
    std::shared_ptr<const VCLFunctionTable> table;
    try {
        table = VCLLoaderHolder().functions();
    } catch (const std::exception&) {
        GTEST_SKIP() << "Couldn't load compiler library";
    }
    EXPECT_TRUE(table->hasAllRequiredSymbols());
}

}  // namespace
