/// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// #include <functional_test_utils/ov_tensor_utils.hpp>
// TODO: Remove ?
#include "test_utils/cpu_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include <ngraph/opsets/opset9.hpp>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
namespace {
    std::vector<InputShape> inputShape;
}  // namespace

using EyeLayerTestParams = std::tuple<
        std::vector<InputShape>,
        ElementType,         // Net precision
        TargetDevice>;       // Device name

using EyeLikeLayerCPUTestParamsSet = std::tuple<
        CPULayerTestsDefinitions::EyeLayerTestParams,
        CPUSpecificParams>;

class EyeLikeLayerCPUTest : public testing::WithParamInterface<EyeLikeLayerCPUTestParamsSet>,
                            virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<EyeLikeLayerCPUTestParamsSet> obj) {
        CPULayerTestsDefinitions::EyeLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;
        std::string td;
        ElementType netPr;
        std::tie(inputShape, netPr, td) = basicParamsSet;
        std::ostringstream result;
        result << "EyeTest_";
        result << "IS=(";
        for (const auto& shape : inputShape) {
            result << CommonTestUtils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=(";
        for (const auto& shape : inputShape) {
            for (const auto& item : shape.second) {
                result << CommonTestUtils::vec2str(item) << "_";
            }
        }
        result << netPr << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams) << "_";
        result << std::to_string(obj.index);
        return result.str();
    }
protected:
    void SetUp() override {
        CPULayerTestsDefinitions::EyeLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        ElementType netPrecision;
        std::tie(inputShape, netPrecision, targetDevice) = basicParamsSet;

        init_input_shapes(inputShape);

        selectedType = std::string("unknown_I32");
        function = createFunction(true);
    }

    std::shared_ptr<ngraph::Function> createFunction(bool secondInputConst) {
        auto rowsPar = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i32, ngraph::Shape{});
        rowsPar->set_friendly_name("rows");
        auto colsPar = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i32, ngraph::Shape{});
        colsPar->set_friendly_name("cols");
        auto diagPar = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i32, ngraph::Shape{});
        diagPar->set_friendly_name("diagInd");
        auto eyelike = std::make_shared<ngraph::op::v9::Eye>(rowsPar, colsPar, diagPar, ngraph::element::i32);
        eyelike->get_rt_info() = getCPUInfo();

        auto function2 = std::make_shared<ngraph::Function>(eyelike->outputs(),
            ngraph::ParameterVector{rowsPar, colsPar, diagPar}, "EyeLike");

        //
        return function2;
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (int i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], 1, 3);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(EyeLikeLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    CheckPluginRelatedResults(compiledModel, "EyeLike");
}

namespace {

/* CPU PARAMS */
std::vector<CPUSpecificParams> filterCPUInfoForDevice(std::string dims = "", std::string modeStr = "") {
        std::vector<CPUSpecificParams> resCPUParams;
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {}, {}});
        return resCPUParams;
}

const std::vector<ElementType> netPrecisions = {
        ElementType::i32
};

std::vector<std::vector<ov::Shape>> staticInput3DShapeVector = {{{}, {}, {}}};

std::vector<std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>> inShapesDynamic = {
       {{ngraph::PartialShape(), ngraph::PartialShape(), ngraph::PartialShape()},
        {{ngraph::Shape{}, ngraph::Shape{}, ngraph::Shape{}}, {ngraph::Shape{}, ngraph::Shape{}, ngraph::Shape{}}}}
};

INSTANTIATE_TEST_SUITE_P(x_smoke_EyeTest, EyeLikeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         ::testing::ValuesIn(static_shapes_to_test_representation(staticInput3DShapeVector)),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice("", ""))),
                         EyeLikeLayerCPUTest::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions