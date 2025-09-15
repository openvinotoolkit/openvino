// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "ze_graph_ext_wrappers.hpp"
#include "zero_init_mock.hpp"

using namespace intel_npu;

class ZeroWrappersTest : public ::testing::TestWithParam<std::tuple<int, std::string>> {
protected:
    void SetUp() override;

    void TearDown() override;

public:
    std::shared_ptr<ZeroInitStructsMock> zeroInitMock;
    
    std::shared_ptr<ZeroInitStructsHolder> zeroInitStruct;

    std::shared_ptr<ZeGraphExtWrappers> zeGraphExt;

    SerializedIR serializedIR;

    GraphDescriptor graphDescriptor;

    std::shared_ptr<ov::Model> model;

    std::string buildFlags;

    static std::string getTestCaseName(testing::TestParamInfo<std::tuple<int, std::string>> obj) {
        int flag;
        std::string version;
        std::tie(flag, version) = obj.param;
        return "graphDescriptorFlag=" + std::to_string(flag) + "_extVersion=" + version;
    }
};

SerializedIR serializeIR(const std::shared_ptr<const ov::Model>& model,
                         ze_graph_compiler_version_info_t compilerVersion,
                         const uint32_t supportedOpsetVersion);

void checkedMemcpy(void* destination, size_t destinationSize, const void* source, size_t numberOfBytes);
