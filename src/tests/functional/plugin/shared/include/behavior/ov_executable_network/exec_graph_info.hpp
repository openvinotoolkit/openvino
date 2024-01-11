// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifcorer: Apache-2.0
//
#pragma once

#include <fstream>

#include "exec_graph_info.hpp"
#include "base/ov_behavior_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
namespace behavior {

typedef std::tuple<
        ov::element::Type_t,                // Element type
        std::string,                        // Device name
        ov::AnyMap                          // Config
> OVExecGraphImportExportTestParams;

class OVExecGraphImportExportTest : public testing::WithParamInterface<OVExecGraphImportExportTestParams>,
                                    public OVCompiledNetworkTestBase {
    public:
    static std::string getTestCaseName(testing::TestParamInfo<OVExecGraphImportExportTestParams> obj);

    void SetUp() override;

    void TearDown() override;

    protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    ov::element::Type_t elementType;
    std::shared_ptr<ov::Model> function;
};

class OVExecGraphUniqueNodeNames : public testing::WithParamInterface<ov::test::BasicParams>,
                                   public OVCompiledNetworkTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::BasicParams> obj);
    void SetUp() override;

protected:
    std::shared_ptr<ov::Model> fnPtr;
};

}  // namespace behavior
}  // namespace test
}  // namespace ov
