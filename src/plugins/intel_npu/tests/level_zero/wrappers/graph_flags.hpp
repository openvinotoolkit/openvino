// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "ze_graph_ext_wrappers.hpp"
#include "zero_init_mock.hpp"

using namespace intel_npu;

class ZeroGraphFlagsTest : public ::testing::TestWithParam<std::string> {
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

    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj) {
        std::string version = obj.param;
        return "extVersion=" + version;
    }

    GraphDescriptor getGraphDesc(const uint32_t flag);
};
