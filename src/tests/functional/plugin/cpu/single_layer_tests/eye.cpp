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
    std::vector<int> pooledSpatialShape;
    std::string mode;
    std::vector<InputShape> inputShape;
}  // namespace

using AdaPoolSpecificParams = std::tuple<
        std::vector<int>,        // pooled vector
        std::vector<InputShape>>;      // feature map shape

using AdaPoolLayerTestParams = std::tuple<
        AdaPoolSpecificParams,
        std::string,         // mode
        bool,                // second Input is Constant
        ElementType,         // Net precision
        TargetDevice>;       // Device name

using EyeLikeLayerCPUTestParamsSet = std::tuple<
        CPULayerTestsDefinitions::AdaPoolLayerTestParams,
        CPUSpecificParams>;

class EyeLikeLayerCPUTest : public testing::WithParamInterface<EyeLikeLayerCPUTestParamsSet>,
                            virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<EyeLikeLayerCPUTestParamsSet> obj) {
        CPULayerTestsDefinitions::AdaPoolLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;
        std::string td;
        ElementType netPr;
        bool isStatic;
        AdaPoolSpecificParams adaPar;
        std::tie(adaPar, mode, isStatic, netPr, td) = basicParamsSet;
        std::tie(pooledSpatialShape, inputShape) = adaPar;
        std::ostringstream result;
        result << "AdaPoolTest_";
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
        result << "OS=" << CommonTestUtils::vec2str(pooledSpatialShape) << "(spat.)_";
        result << netPr << "_";
        result << mode << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams) << "_";
        result << std::to_string(obj.index);
        return result.str();
    }
protected:
    void SetUp() override {
        CPULayerTestsDefinitions::AdaPoolLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        CPULayerTestsDefinitions::AdaPoolSpecificParams adaPoolParams;
        ElementType netPrecision;
        bool isStatic;
        std::tie(adaPoolParams, mode, isStatic, netPrecision, targetDevice) = basicParamsSet;
        std::tie(pooledVector, inputShape) = adaPoolParams;

        init_input_shapes(inputShape);

        selectedType = std::string("unknown_FP32");
        if (netPrecision == ElementType::bf16) {
            rel_threshold = 1e-2;
        }
        function = createFunction(isStatic);
    }

    std::shared_ptr<ngraph::Function> createFunction(bool secondInputConst) {
        // auto params = ngraph::builder::makeDynamicParams(ngraph::element::i32, { inputDynamicShapes[0] });
        // params.front()->set_friendly_name("ParamsInput");
        // std::shared_ptr<ov::Node> secondInput;
        // if (secondInputConst) {
        //     secondInput = ngraph::op::Constant::create(ngraph::element::i32, ngraph::Shape{pooledVector.size()}, pooledVector);
        // } else {
        //     auto pooledParam = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i32, ngraph::Shape{pooledVector.size()});
        //     pooledParam->set_friendly_name("ParamSecondInput");
        //     params.push_back(pooledParam);
        //     secondInput = pooledParam;
        // }
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

    void init_ref_function(std::shared_ptr<ov::Model> &funcRef, const std::vector<ov::Shape>& targetInputStaticShapes) override {
        if (function->get_parameters().size() == 2) {
            funcRef = createFunction(true);
        }
        ngraph::helpers::resize_function(funcRef, targetInputStaticShapes);
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

private:
    std::vector<int> pooledVector;
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
        {{ngraph::Shape{}, ngraph::Shape{}}, {ngraph::Shape{}, ngraph::Shape{}}}}
};

const std::vector<std::vector<int>> pooled3DVector = {
        { 1 },
        { 3 },
        { 5 }
};

const auto staticParams = ::testing::Combine(
        ::testing::ValuesIn(pooled3DVector),         // output spatial shape
        ::testing::ValuesIn(static_shapes_to_test_representation(staticInput3DShapeVector))
);

INSTANTIATE_TEST_SUITE_P(x_smoke_StaticAdaPoolAvg3DLayoutTest, EyeLikeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         staticParams,
                                         ::testing::Values("avg"),
                                         ::testing::Values(true),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice("", ""))),
                         EyeLikeLayerCPUTest::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions