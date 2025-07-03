// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <unordered_set>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
namespace behavior {

struct PluginParameter {
    std::string _name;
    std::string _location;
};

struct FunctionParameter {
    std::unordered_set<std::string>  _majorPluginNodeIds;
    std::shared_ptr<ov::Model>       _function;
    bool                             _dynamic_batch;
    uint32_t                         _seed;
};

using OVHeteroSyntheticTestParameters = std::tuple<
    std::vector<PluginParameter>,
    FunctionParameter
>;

class OVHeteroSyntheticTest : public testing::WithParamInterface<OVHeteroSyntheticTestParameters>,
                             virtual public ov::test::SubgraphBaseStaticTest {
protected:
    enum {Plugin, Function};

    ~OVHeteroSyntheticTest() override = default;
    void SetUp() override;
    void TearDown() override;

    std::string SetUpAffinity();

    std::vector<std::string> _registredPlugins;

public:
    static std::string getTestCaseName(const ::testing::TestParamInfo<OVHeteroSyntheticTestParameters>& obj);

    static std::vector<FunctionParameter> singleMajorNodeFunctions(
        const std::vector<std::function<std::shared_ptr<ov::Model>()>>& builders, bool dynamic_batch  = false);

    static std::vector<FunctionParameter> randomMajorNodeFunctions(
        const std::vector<std::function<std::shared_ptr<ov::Model>()>>& builders, bool dynamic_batch = false, uint32_t seed = 0);

    static std::vector<FunctionParameter> withMajorNodesFunctions(
        const std::function<std::shared_ptr<ov::Model>()>& builder,
        const std::unordered_set<std::string>& majorNodes,
        bool dynamic_batch = false);

    static std::vector<FunctionParameter> _singleMajorNodeFunctions;
    static std::vector<FunctionParameter> _randomMajorNodeFunctions;
};

}  // namespace behavior
}  // namespace test
}  // namespace ov
