// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "ze_graph_ext_wrappers.hpp"
#include "driver_compiler_adapter.hpp"

using namespace intel_npu;

class ZeroWrappersTest : public ::testing::TestWithParam<int> {
protected:
    void SetUp() override;
    
    void TearDown() override;
    
public:
    std::shared_ptr<ZeGraphExtWrappers> zeGraphExt;
    
    std::unique_ptr<DriverCompilerAdapter> compilerAdapter;
    
    SerializedIR serializedIR;
    
    GraphDescriptor graphDescriptor;
    
    std::shared_ptr<ov::Model> model;
    
    std::string buildFlags;
};

SerializedIR serializeIR(const std::shared_ptr<const ov::Model>& model,
                                                ze_graph_compiler_version_info_t compilerVersion,
                                                const uint32_t supportedOpsetVersion);

void checkedMemcpy(void* destination, size_t destinationSize, const void* source, size_t numberOfBytes);
