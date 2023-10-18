// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/single_layer/topk.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

typedef std::tuple<
        int64_t,                            // keepK
        int64_t,                            // axis
        ngraph::opset4::TopK::Mode,         // mode
        ngraph::opset4::TopK::SortType,     // sort
        ElementType,                        // Net precision
        ElementType,                        // Input precision
        ElementType,                        // Output precision
        InputShape,                         // inputShape
        TargetDevice,                       // Device name
        ngraph::helpers::InputLayerType     // Input type
> TopKLayerTestParamsSet;

class TopKLayerGPUTest : public testing::WithParamInterface<TopKLayerTestParamsSet>,
                         virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TopKLayerTestParamsSet>& obj) {
        TopKLayerTestParamsSet basicParamsSet = obj.param;

        int64_t keepK, axis;
        ngraph::opset4::TopK::Mode mode;
        ngraph::opset4::TopK::SortType sort;
        ElementType netPrecision, inPrc, outPrc;
        InputShape inputShape;
        TargetDevice targetDevice;
        ngraph::helpers::InputLayerType inputType;
        std::tie(keepK, axis, mode, sort, netPrecision, inPrc, outPrc, inputShape, targetDevice, inputType) = basicParamsSet;

        std::ostringstream result;
        result << "k=" << keepK << "_";
        result << "axis=" << axis << "_";
        result << "mode=" << mode << "_";
        result << "sort=" << sort << "_";
        result << "netPRC=" << netPrecision << "_";
        result << "inPRC=" << inPrc << "_";
        result << "outPRC=" << outPrc << "_";
        result << "IS=" << ov::test::utils::partialShape2str({inputShape.first}) << "_" << "TS=(";
        for (const auto& shape : inputShape.second) {
            result << ov::test::utils::vec2str(shape) << "_";
        }
        result << ")_";
        result << "inputType=" << inputType;
        result << "TargetDevice=" << targetDevice;

        return result.str();
    }

protected:
    void SetUp() override {
        TopKLayerTestParamsSet basicParamsSet = this->GetParam();

        int64_t keepK;
        ngraph::opset4::TopK::Mode mode;
        ngraph::opset4::TopK::SortType sort;
        ElementType inPrc, outPrc;
        InputShape inputShape;
        std::tie(keepK, axis, mode, sort, netPrecision, inPrc, outPrc, inputShape, targetDevice, inputType) = basicParamsSet;

        if (inputType == ngraph::helpers::InputLayerType::CONSTANT) {
            init_input_shapes({inputShape});
        } else {
            inputDynamicShapes = {inputShape.first, {}};
            for (size_t i = 0; i < inputShape.second.size(); ++i) {
                targetStaticShapes.push_back({inputShape.second[i], {}});
            }
        }

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(netPrecision, inputDynamicShapes[0])};

        std::shared_ptr<ngraph::opset4::TopK> topk;
        if (inputType == ngraph::helpers::InputLayerType::CONSTANT) {
            auto k = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{}, &keepK);
            topk = std::dynamic_pointer_cast<ngraph::opset4::TopK>(std::make_shared<ngraph::opset4::TopK>(params[0], k, axis, mode, sort));
        } else {
            auto k = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::Type_t::i64, inputDynamicShapes[1]);
            params.push_back(k);
            topk = std::dynamic_pointer_cast<ngraph::opset4::TopK>(
                    std::make_shared<ngraph::opset4::TopK>(params[0], k, axis, mode, sort));
        }

        ngraph::ResultVector results;
        for (size_t i = 0; i < topk->get_output_size(); i++) {
            results.push_back(std::make_shared<ngraph::opset4::Result>(topk->output(i)));
        }

        function = std::make_shared<ngraph::Function>(results, params, "TopK");
    }

    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        auto shape = targetInputStaticShapes.front();
        ov::Tensor tensor;
        tensor = ov::test::utils::create_and_fill_tensor(funcInputs[0].get_element_type(), shape);
        size_t size = tensor.get_size();

        if (netPrecision == ElementType::f32) {
            std::vector<int> data(size);

            int start = - static_cast<int>(size / 2);
            std::iota(data.begin(), data.end(), start);
            std::mt19937 gen(0);
            std::shuffle(data.begin(), data.end(), gen);

            auto *rawBlobDataPtr = static_cast<float *>(tensor.data());
            for (size_t i = 0; i < size; ++i) {
                rawBlobDataPtr[i] = static_cast<float>(data[i]);
            }
        } else {
            FAIL() << "generate_inputs for " << netPrecision << " precision isn't supported";
        }
        inputs.insert({funcInputs[0].get_node_shared_ptr(), tensor});

        if (inputType == ngraph::helpers::InputLayerType::PARAMETER) {
            const auto& kPrecision = funcInputs[1].get_element_type();
            const auto& kShape = targetInputStaticShapes[1];

            const size_t startFrom = 1;
            const size_t range = targetInputStaticShapes[0][axis];
            const size_t seed = inferRequestNum++;
            const auto kTensor = ov::test::utils::create_and_fill_tensor(kPrecision, kShape, range, startFrom, 1, seed);

            inputs.insert({funcInputs[1].get_node_shared_ptr(), kTensor});
        }
    }

private:
    int64_t axis;
    size_t inferRequestNum = 0;
    ElementType netPrecision;
    ngraph::helpers::InputLayerType inputType;
};

TEST_P(TopKLayerGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
}

namespace {

const std::vector<ElementType> netPrecisions = {
    ElementType::f32,
};

const std::vector<int64_t> axes = {0, 3};
const std::vector<int64_t> k = {3, 5, 7};

const std::vector<ngraph::opset4::TopK::Mode> modes = {
    ngraph::opset4::TopK::Mode::MIN,
    ngraph::opset4::TopK::Mode::MAX
};

const std::vector<ngraph::opset4::TopK::SortType> sortTypes = {
    ngraph::opset4::TopK::SortType::SORT_VALUES,
    ngraph::opset4::TopK::SortType::SORT_INDICES,
};

std::vector<ov::test::InputShape> inputShapesDynamic = {
    {
        {ov::PartialShape::dynamic(4), {{7, 7, 7, 7}, {7, 8, 7, 9}}},
        {{-1, -1, -1, -1}, {{8, 9, 10, 11}, {11, 7, 8, 9}}}
    }
};

INSTANTIATE_TEST_CASE_P(smoke_TopK_constant_dynamic, TopKLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(k),
        ::testing::ValuesIn(axes),
        ::testing::ValuesIn(modes),
        ::testing::ValuesIn(sortTypes),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ElementType::undefined),
        ::testing::Values(ElementType::undefined),
        ::testing::ValuesIn(inputShapesDynamic),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT)),
    TopKLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_TopK_parameter_dynamic, TopKLayerGPUTest,
    ::testing::Combine(
        ::testing::Values(1),
        ::testing::ValuesIn(axes),
        ::testing::ValuesIn(modes),
        ::testing::ValuesIn(sortTypes),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ElementType::undefined),
        ::testing::Values(ElementType::undefined),
        ::testing::ValuesIn(inputShapesDynamic),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER)),
    TopKLayerGPUTest::getTestCaseName);

} // namespace
} // namespace GPULayerTestsDefinitions
