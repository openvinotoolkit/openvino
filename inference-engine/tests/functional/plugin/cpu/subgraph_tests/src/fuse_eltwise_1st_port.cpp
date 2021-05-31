// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <debug.h>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <ie_precision.hpp>
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "ie_system_conf.h"

//#include "low_precision/network_helper.hpp"

// NB! For debug purposes
#include <transformations/serialize.hpp>

using namespace CPUTestUtils;
using namespace ngraph;
using InferenceEngine::Precision;
using ngraph::helpers::EltwiseTypes;

namespace CPUSubgraphTestsDefinitions {

typedef std::tuple<
        Shape,                      // Input shapes
        element::Type,              // Parent precision
        element::Type,              // Child precision
        std::vector<EltwiseTypes>,  // Types of eltwise operations
        bool,                       // True if child and parent are connected on the first port (false if on the zero)
        std::string                 // Device name
> FuseEltwiseOn1stPortTuple;

class FuseEltwiseOn1stPortTest : public testing::WithParamInterface<FuseEltwiseOn1stPortTuple>,
                         virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FuseEltwiseOn1stPortTuple> &obj) {
        Shape inputShape;
        element::Type  parent_precision, child_precision;
        std::vector<EltwiseTypes> eltwiseOpTypes;
        std::string targetDevice;
        bool connected_on_1st_port;
        std::tie(inputShape, parent_precision, child_precision, eltwiseOpTypes, connected_on_1st_port, targetDevice) = obj.param;
        std::ostringstream results;
        results << "IS" << CommonTestUtils::vec2str(inputShape) << "_";
        results << "ParPRC" << "=" << parent_precision << "_";
        results << "ChildPRC" << "=" << child_precision << "_";
        for (int i = 0; i < eltwiseOpTypes.size(); i++) {
            results << "Op" << std::to_string(i) << "=" << eltwiseOpTypes[i] << "_";
        }
        results << "ConnectedOnPort" << "=" << (connected_on_1st_port ? 1 : 0) << "_";
        results << "targetDevice=" << targetDevice;
        return results.str();
    }

protected:
    void SetUp() {
        threshold = 0.1f;

        Shape inputShape;
        element::Type parent_precision, child_precision;
        std::vector<EltwiseTypes> eltwiseOpTypes;
        bool connected_on_1st_port;
        std::tie(inputShape, parent_precision, child_precision, eltwiseOpTypes, connected_on_1st_port, targetDevice) = this->GetParam();

        ASSERT_EQ(eltwiseOpTypes.size(), 2) << "Only two eltwise types are accepted";

        auto param_parent = std::make_shared<opset6::Parameter>(parent_precision, inputShape);
        auto param_child = std::make_shared<opset6::Parameter>(child_precision, inputShape);

        std::vector<float> Input1Data2(1);
        auto const_parent = ngraph::builder::makeConstant(parent_precision, Shape{1}, Input1Data2, true);
        auto parent = ngraph::builder::makeEltwise(const_parent, param_parent, eltwiseOpTypes[0]);

        std::shared_ptr<ngraph::Node> child;
        if (connected_on_1st_port)
            child = ngraph::builder::makeEltwiseRelaxed(param_child, parent, eltwiseOpTypes[1]);
        else
            child = ngraph::builder::makeEltwiseRelaxed(parent, param_child, eltwiseOpTypes[1]);

        std::vector<std::shared_ptr<ngraph::Node>> nodes {parent, child};
        for (auto node : nodes) {
            ASSERT_EQ(node->get_input_size(), 2) <<
                                                 "This test supports eltwise ops only with two inputs.";
        }
        ngraph::ResultVector results{std::make_shared<ngraph::opset6::Result>(child)};
        ngraph::ParameterVector params{param_parent, param_child};

        function = std::make_shared<ngraph::Function>(results, params, "FuseOnSecondPort");
        #define cover_precisions(PARENT_PRECISION, CHILD_PRECISION) \
        case element::CHILD_PRECISION: \
            additionalPasses.push_back(std::make_shared<ngraph::pass::ConvertPrecision<element::PARENT_PRECISION, element::CHILD_PRECISION>>());\
            break

        switch (parent_precision) {
            case element::i32: {
                switch (child_precision) {
                    cover_precisions(i32, f32);
                    default:
                        GTEST_FAIL() << "Child precision " << child_precision << " is not supported.";
                }
                break;
            }
            default:
                GTEST_FAIL() << "Parent precision " << parent_precision << " is not supported.";
        }
        #undef cover_precisions
    }
    void GenerateInputs() {
        const auto& inputsInfo = executableNetwork.GetInputsInfo();
        const auto& functionParams = function->get_parameters();
        for (int i = 0; i < functionParams.size(); ++i) {
            const auto& param = functionParams[i];
            const auto infoIt = inputsInfo.find(param->get_friendly_name());
            GTEST_ASSERT_NE(infoIt, inputsInfo.cend());

            const auto& desc = infoIt->second->getTensorDesc();
            uint32_t range = 10;
            int32_t resolution = 1;
            if (desc.getPrecision().is_float()) {
                range = 1;
                resolution = 1000;
            }
            const int32_t start_from = 0;
            auto blob = FuncTestUtils::createAndFillBlob(desc, range, start_from, resolution);
            inputs.push_back(blob);
        }
    }
};

TEST_P(FuseEltwiseOn1stPortTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
}

namespace {
/* This test checks that an eltwise_node correctly sets input precision if it fused other op (e.g. another eltwise) on non-zero port.
 * inputShapes - any shape;
 * childPrecision - must be chosen so that conversion it to parentPrecision would loose information;
 * eltwiseOps - any two eltwise ops, each of them should accept two inputs;
 * connectedOn1stPort - defines the port that connects two eltwise nodes;
*/
std::vector<Shape> inputShapes {{1, 1, 2, 3}};
std::vector<element::Type> childPrecisions = {element::f32};
std::vector<element::Type> parentPrecisions = {element::i32};
std::vector<std::vector<EltwiseTypes>> eltwiseOps = {{ EltwiseTypes::ADD, EltwiseTypes::MULTIPLY}};
std::vector<bool> connectedOn1stPort = {false, true};

INSTANTIATE_TEST_CASE_P(smoke_FuseEltwiseOn1stPort, FuseEltwiseOn1stPortTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes),
                                ::testing::ValuesIn(parentPrecisions),
                                ::testing::ValuesIn(childPrecisions),
                                ::testing::ValuesIn(eltwiseOps),
                                ::testing::ValuesIn(connectedOn1stPort),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        FuseEltwiseOn1stPortTest::getTestCaseName);

} // namespace
} // namespace CPUSubgraphTestsDefinitions
