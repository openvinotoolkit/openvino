// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <unordered_set>
#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace HeteroTests {

struct PluginParameter {
    std::string _name;
    std::string _location;
};

struct FunctionParameter {
    std::unordered_set<std::string>     _majorPluginNodeIds;
    std::shared_ptr<ngraph::Function>   _function;
};

using HeteroSyntheticTestParameters = std::tuple<
    std::vector<PluginParameter>,
    FunctionParameter
>;

struct HeteroSyntheticTest : public testing::WithParamInterface<HeteroSyntheticTestParameters>,
                             virtual public LayerTestsUtils::LayerTestsCommon {
    enum {Plugin, Function};
    ~HeteroSyntheticTest() override = default;
    void SetUp() override;
    void TearDown() override;
    std::string SetUpAffinity();
    static std::string getTestCaseName(const ::testing::TestParamInfo<HeteroSyntheticTestParameters>& obj);
    static std::vector<FunctionParameter> _singleMajorNodeFunctions;
    static std::vector<FunctionParameter> _randomMajorNodeFunctions;
    std::vector<std::string> _registredPlugins;
};

}  //  namespace HeteroTests
