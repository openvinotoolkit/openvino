// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/single_layer/topk.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

typedef std::tuple<
        int64_t,                           // keepK
        int64_t,                           // axis
        ngraph::opset4::TopK::Mode,        // mode
        ngraph::opset4::TopK::SortType,    // sort
        ElementType,                       // Net precision
        ElementType,                       // Input precision
        ElementType,                       // Output precision
        InputShape,                        // inputShape
        TargetDevice,                      // Device name
        std::map<std::string, std::string> // Additional network configuration
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
        std::map<std::string, std::string> additionalConfig;
        std::tie(keepK, axis, mode, sort, netPrecision, inPrc, outPrc, inputShape, targetDevice, additionalConfig) = basicParamsSet;

        std::ostringstream result;
        result << "k=" << keepK << "_";
        result << "axis=" << axis << "_";
        result << "mode=" << mode << "_";
        result << "sort=" << sort << "_";
        result << "netPRC=" << netPrecision << "_";
        result << "inPRC=" << inPrc << "_";
        result << "outPRC=" << outPrc << "_";
        result << "IS=" << CommonTestUtils::partialShape2str({inputShape.first}) << "_" << "TS=(";
        for (const auto& shape : inputShape.second) {
            result << CommonTestUtils::vec2str(shape) << "_";
        }
        result << "config=(";
        for (const auto configEntry : additionalConfig) {
            result << configEntry.first << ", " << configEntry.second << ":";
        }
        result << ")_";
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
        std::map<std::string, std::string> additionalConfig;
        std::tie(keepK, axis, mode, sort, netPrecision, inPrc, outPrc, inputShape, targetDevice, additionalConfig) = basicParamsSet;

        init_input_shapes({inputShape});

        auto params = ngraph::builder::makeDynamicParams(netPrecision, {inputDynamicShapes[0]});

        std::shared_ptr<ngraph::opset4::TopK> topk;
        auto k = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{}, &keepK);
        topk = std::dynamic_pointer_cast<ngraph::opset4::TopK>(std::make_shared<ngraph::opset4::TopK>(params[0], k, axis, mode, sort));

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

        if (netPrecision == ElementType::f32 || netPrecision == ElementType::i32) {
            std::vector<int> data(size);

            // For int32, deliberately set big numbers which are not accurately representable in fp32
            int start = netPrecision == ElementType::i32 ? pow(2, 30) + 1 : - static_cast<int>(size / 2);
            std::iota(data.begin(), data.end(), start);
            std::mt19937 gen(0);
            std::shuffle(data.begin(), data.end(), gen);

            if (netPrecision == ElementType::f32) {
                auto *rawBlobDataPtr = static_cast<float *>(tensor.data());
                for (size_t i = 0; i < size; ++i) {
                    rawBlobDataPtr[i] = static_cast<float>(data[i]);
                }
            } else {
                auto *rawBlobDataPtr = static_cast<int32_t *>(tensor.data());
                for (size_t i = 0; i < size; ++i) {
                    rawBlobDataPtr[i] = static_cast<int32_t>(data[i]);
                }
            }
        } else if (netPrecision == ElementType::bf16) {
            size_t O = 1, A = 1, I = 1;
            A = shape[axis];
            for (size_t i = 0; i < axis; i++)
                O *= shape[i];
            for (size_t i = axis + 1; i < shape.size(); i++)
                I *= shape[i];
            if (O * A * I != size)
                FAIL() << "Incorrect blob shape " << shape;

            auto *rawBlobDataPtr = static_cast<ngraph::bfloat16 *>(tensor.data());
            for (size_t o = 0; o < O; o++) {
                for (size_t i = 0; i < I; i++) {
                    std::vector<int> data(A);
                    int start = - static_cast<int>(A / 2);
                    std::iota(data.begin(), data.end(), start);
                    const size_t seed = (o + 1) * (i + 1);
                    std::mt19937 gen(seed);
                    std::shuffle(data.begin(), data.end(), gen);
                    for (size_t a = 0; a < A; a++) {
                        rawBlobDataPtr[o * A * I + a * I + i] = static_cast<ngraph::bfloat16>(data[a]);
                    }
                }
            }
        } else {
            FAIL() << "generate_inputs for " << netPrecision << " precision isn't supported";
        }
        inputs.insert({funcInputs[0].get_node_shared_ptr(), tensor});
    }

private:
    int64_t axis;
    ElementType netPrecision;
    bool staticShape;
};

TEST_P(TopKLayerGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
}

namespace {

std::map<std::string, std::string> emptyAdditionalConfig;

const std::vector<ElementType> netPrecisions = {
    ElementType::f32,
};

const std::vector<int64_t> axes = {0, 1, 2, 3};
const std::vector<int64_t> k = {1, 5, 7, 18, 21};

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
        {{21, {20, 25}, 21, {20, 25}}, {{21, 21, 21, 21}, {21, 22, 21, 23}}},
        {ov::PartialShape::dynamic(4), {{21, 21, 21, 21}, {21, 22, 21, 23}}}
    }
};

INSTANTIATE_TEST_CASE_P(smoke_TopK_dynamic, TopKLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(k),
        ::testing::ValuesIn(axes),
        ::testing::ValuesIn(modes),
        ::testing::ValuesIn(sortTypes),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ElementType::undefined),
        ::testing::Values(ElementType::undefined),
        ::testing::ValuesIn(inputShapesDynamic),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::Values(emptyAdditionalConfig)),
    TopKLayerGPUTest::getTestCaseName);

} // namespace
} // namespace GPULayerTestsDefinitions
