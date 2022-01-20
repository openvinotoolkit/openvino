// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "test_utils/cpu_test_utils.hpp"

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
using topKParams = std::tuple<
    ngraph::element::Type, // data precision
    InputShape, // data shape
    std::int64_t, // axis
    ngraph::opset4::TopK::Mode,
    ngraph::opset4::TopK::SortType>;

class TopKLayerCPUTest : public testing::WithParamInterface<topKParams>, public ov::test::SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<topKParams> obj) {
        ngraph::element::Type inputPrecision;
        InputShape inputShape;
        std::int64_t axis;
        ngraph::opset4::TopK::Mode mode;
        ngraph::opset4::TopK::SortType sortType;
        std::tie(inputPrecision, inputShape, axis, mode, sortType) = obj.param;

        std::ostringstream result;
        result << inputPrecision << "_" << "IS=" << CommonTestUtils::partialShape2str({ inputShape.first }) << "_" << "TS=(";
        for (const auto& shape : inputShape.second) {
            result << CommonTestUtils::vec2str(shape) << "_";
        }

        result << ")_axis=" << axis << "_mode=" << mode << "_sortType=" << sortType;
        return result.str();
    }

protected:
    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        ov::runtime::Tensor data_tensor;
        const auto& dataPrecision = funcInputs[0].get_element_type();
        const auto& dataShape = targetInputStaticShapes.front();
        if (funcInputs[0].get_element_type().is_real()) {
            data_tensor = ov::test::utils::create_and_fill_tensor(dataPrecision, dataShape, 10, 0, 1000);
        } else {
            data_tensor = ov::test::utils::create_and_fill_tensor(dataPrecision, dataShape);
        }

        const auto axis = std::get<2>(this->GetParam());
        const auto& kPrecision = funcInputs[1].get_element_type();
        const auto& kShape = targetInputStaticShapes[1];

        const size_t startFrom = 1;
        const size_t range = targetInputStaticShapes[0][axis] - 1;
        const size_t seed = inferRequestNum++;
        const auto kTensor = ov::test::utils::create_and_fill_tensor(kPrecision, kShape, range, startFrom, 1, seed);

        inputs.insert({ funcInputs[0].get_node_shared_ptr(), data_tensor });
        inputs.insert({ funcInputs[1].get_node_shared_ptr(), kTensor });
    }

    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;
        ngraph::element::Type inputPrecision;
        InputShape inputShape;
        std::int64_t axis;
        ngraph::opset4::TopK::Mode mode;
        ngraph::opset4::TopK::SortType sortType;
        std::tie(inputPrecision, inputShape, axis, mode, sortType) = this->GetParam();

        inputDynamicShapes = { inputShape.first, {} };
        for (size_t i = 0; i < inputShape.second.size(); ++i) {
            targetStaticShapes.push_back({ inputShape.second[i], {} });
        }

        selectedType = makeSelectedTypeStr("ref_any", inputPrecision);

        auto params = ngraph::builder::makeDynamicParams(inputPrecision, { inputDynamicShapes[0] });
        auto k_param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i32, inputDynamicShapes[1]);
        params.push_back(k_param);
        auto topk = std::make_shared<ngraph::opset1::TopK>(params[0], k_param, axis, mode, sortType);

        ngraph::ResultVector results;
        for (size_t i = 0; i < topk->get_output_size(); ++i) {
            results.push_back(std::make_shared<ngraph::opset1::Result>(topk->output(i)));
        }

        function = std::make_shared<ngraph::Function>(results, params, "TopKLayerCPUTest");
        functionRefs = ngraph::clone_function(*function);
    }

private:
    size_t inferRequestNum = 0;
};

TEST_P(TopKLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
    CheckPluginRelatedResults(executableNetwork, "TopK");
}

const ngraph::element::TypeVector inputPrecisions = {
    ngraph::element::f32,
};

const std::vector<int64_t> axes = { 0, 1, 2 };

const std::vector<ngraph::opset4::TopK::Mode> modes = {
    ngraph::opset4::TopK::Mode::MIN,
    ngraph::opset4::TopK::Mode::MAX
};

const std::vector<ngraph::opset4::TopK::SortType> sortTypes = {
    ngraph::opset4::TopK::SortType::SORT_INDICES,
    ngraph::opset4::TopK::SortType::SORT_VALUES,
};

const std::vector<InputShape> inShapes = {
    InputShape{
        // dynamic
        {-1, -1, -1},
        // target
        {
            {10, 13, 11},
            {15, 10, 5},
            {5, 5, 8}
        }
    },
    InputShape{
        // dynamic
        {-1, { 7, 25 }, -1},
        // target
        {
            {10, 7, 10},
            {15, 20, 5},
            {5, 25, 7}
        }
    },
    InputShape{
        // dynamic
        {{8, 10}, {7, 17}, {12, 27}},
        // target
        {
            {8, 7, 12},
            {9, 12, 25},
            {10, 17, 27}
        }
    },
};

const auto testCases = ::testing::Combine(
    ::testing::ValuesIn(inputPrecisions),
    ::testing::ValuesIn(inShapes),
    ::testing::ValuesIn(axes),
    ::testing::ValuesIn(modes),
    ::testing::ValuesIn(sortTypes)
);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs, TopKLayerCPUTest, testCases, TopKLayerCPUTest::getTestCaseName);

} // namespace CPULayerTestsDefinitions
