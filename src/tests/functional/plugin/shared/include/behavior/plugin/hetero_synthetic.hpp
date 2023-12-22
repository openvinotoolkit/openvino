// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <unordered_set>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace HeteroTests {

struct PluginParameter {
    std::string _name;
    std::string _location;
};

struct FunctionParameter {
    std::unordered_set<std::string>         _majorPluginNodeIds;
    std::shared_ptr<ngraph::Function>       _function;
    bool                                    _dynamic_batch;
    uint32_t                                _seed;
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
    static std::vector<FunctionParameter> singleMajorNodeFunctions(
        const std::vector<std::function<std::shared_ptr<ngraph::Function>()>>& builders, bool dynamic_batch = false);
    static std::vector<FunctionParameter> randomMajorNodeFunctions(
        const std::vector<std::function<std::shared_ptr<ngraph::Function>()>>& builders, bool dynamic_batch = false, uint32_t seed = 0);
    static std::vector<FunctionParameter> withMajorNodesFunctions(
        const std::function<std::shared_ptr<ngraph::Function>()>& builder,
        const std::unordered_set<std::string>& majorNodes,
        bool dynamic_batch = false);
    std::vector<std::string> _registredPlugins;
};

}  //  namespace HeteroTests
