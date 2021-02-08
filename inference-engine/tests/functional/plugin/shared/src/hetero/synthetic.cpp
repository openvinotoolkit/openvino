// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "hetero/synthetic.hpp"
#include <ngraph/op/util/op_types.hpp>
#include <ngraph/variant.hpp>
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include <random>
namespace HeteroTests {

static std::vector<std::function<std::shared_ptr<ngraph::Function>()>> builders = {
    [] {return ngraph::builder::subgraph::makeSplitMultiConvConcat();},
    [] {return ngraph::builder::subgraph::makeNestedSplitConvConcat();},
    [] {return ngraph::builder::subgraph::makeSplitConvConcatNestedInBranch();},
    [] {return ngraph::builder::subgraph::makeSplitConvConcatNestedInBranchNestedOut();},
};

std::vector<FunctionParameter> HeteroSyntheticTest::_singleMajorNodeFunctions{[] {
    std::vector<FunctionParameter> result;
    for (auto&& builder : builders) {
        auto function = builder();
        for (auto&& node : function->get_ordered_ops()) {
            if (!ngraph::op::is_constant(node) &&
                    !(ngraph::op::is_parameter(node)) &&
                    !(ngraph::op::is_output(node))) {
                result.push_back(FunctionParameter{{node->get_friendly_name()}, function});
            }
        }
    }
    return result;
} ()};

std::vector<FunctionParameter> HeteroSyntheticTest::_randomMajorNodeFunctions{[] {
    std::vector<FunctionParameter> results;
    for (auto p = 0.2; p < 1.; p+=0.2) {
        std::mt19937 e{std::random_device {} ()};
        std::bernoulli_distribution d{p};
        for (auto&& builder : builders) {
            auto function = builder();
            auto ordered_ops = function->get_ordered_ops();
            for (std::size_t i = 0; i < ordered_ops.size(); ++i) {
                std::unordered_set<std::string> majorPluginNodeIds;
                for (auto&& node : ordered_ops) {
                    if (!(ngraph::op::is_constant(node)) &&
                            !(ngraph::op::is_parameter(node)) &&
                            !(ngraph::op::is_output(node)) && d(e)) {
                        majorPluginNodeIds.emplace(node->get_friendly_name());
                    }
                }
                if (std::any_of(std::begin(results), std::end(results), [&] (const FunctionParameter& param) {
                    return majorPluginNodeIds == param._majorPluginNodeIds;
                })) {
                    continue;
                }
                results.push_back(FunctionParameter{majorPluginNodeIds, function});
            }
        }
    }
    return results;
} ()};

std::string HeteroSyntheticTest::getTestCaseName(const ::testing::TestParamInfo<HeteroSyntheticTestParameters>& obj) {
    std::vector<PluginParameter> pluginParameters;
    FunctionParameter functionParamter;
    std::tie(pluginParameters, functionParamter) = obj.param;
    std::string name = "function=" + functionParamter._function->get_friendly_name();
    name += "_layers=";
    std::size_t num = functionParamter._majorPluginNodeIds.size() - 1;
    for (auto&& id : functionParamter._majorPluginNodeIds) {
        name += id + ((num !=0) ? "," : "");
        num--;
    }
    name += "_targetDevice=HETERO:";
    num = pluginParameters.size() - 1;
    for (auto&& pluginParameter : pluginParameters) {
        name += pluginParameter._name + ((num !=0) ? "," : "");
        num--;
    }
    return name;
}

void HeteroSyntheticTest::SetUp() {
    auto& param = GetParam();
    targetDevice = "HETERO:";
    int num = std::get<Plugin>(param).size() - 1;
    for (auto&& pluginParameter : std::get<Plugin>(param)) {
        bool registred = true;
        try {
            PluginCache::get().ie()->RegisterPlugin(pluginParameter._location, pluginParameter._name);
        } catch (InferenceEngine::details::InferenceEngineException& ex) {
            if (std::string{ex.what()}.find("Device with \"" + pluginParameter._name
                                             + "\"  is already registered in the InferenceEngine")
                == std::string::npos) {
                throw ex;
            } else {
                registred = false;
            }
        }
        if (registred) {
            _registredPlugins.push_back(pluginParameter._name);
        }
        targetDevice += pluginParameter._name;
        targetDevice += ((num !=0) ? "," : "");
        --num;
    }
    function = std::get<Function>(param)._function;
}

void HeteroSyntheticTest::TearDown() {
    if (!FuncTestUtils::SkipTestsConfig::currentTestIsDisabled()) {
        for (auto&& pluginName : _registredPlugins) {
            PluginCache::get().ie()->UnregisterPlugin(pluginName);
        }
    }
}

std::string HeteroSyntheticTest::SetUpAffinity() {
    int id = 0;
    auto& param = GetParam();
    std::string affinities;
    auto& pluginParameters = std::get<Plugin>(param);
    affinities += "\n{\n";
    for (auto&& node : std::get<Function>(param)._function->get_ordered_ops()) {
        if (!ngraph::op::is_constant(node) &&
                !(ngraph::op::is_parameter(node)) &&
                !(ngraph::op::is_output(node))) {
            std::string affinity;
            if (std::get<Function>(param)._majorPluginNodeIds.end() !=
                std::get<Function>(param)._majorPluginNodeIds.find(node->get_friendly_name())) {
                affinity = pluginParameters.at(0)._name;
            } else {
                affinity = pluginParameters.at(1)._name;
            }
            node->get_rt_info()["affinity"] = std::make_shared<ngraph::VariantWrapper<std::string>>(affinity);
            affinities += "\t{" + node->get_friendly_name() + ",\t\t" + affinity + "}\n";
        }
    }
    affinities += "}";
    return affinities;
}

TEST_P(HeteroSyntheticTest, someLayersToMajorPluginOthersToFallback) {
    auto affinities = SetUpAffinity();
    SCOPED_TRACE(affinities);
    Run();
    if (!FuncTestUtils::SkipTestsConfig::currentTestIsDisabled()) {
        ASSERT_NE(nullptr, cnnNetwork.getFunction());
    }
}

}  //  namespace HeteroTests
