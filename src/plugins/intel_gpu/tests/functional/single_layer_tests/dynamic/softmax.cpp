// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/softmax.hpp"

namespace {

typedef std::tuple<
    ov::element::Type,     // model type
    ov::test::InputShape,  // inputShape
    int64_t>               // axis
softmaxGPUTestParamsSet;

class SoftMaxLayerGPUTest : public testing::WithParamInterface<softmaxGPUTestParamsSet>,
                            virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<softmaxGPUTestParamsSet>& obj) {
        ov::element::Type model_type;
        ov::test::InputShape inShape;
        int64_t axis;
        std::tie(model_type, inShape, axis) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << model_type << "_";
        result << "IS=" << ov::test::utils::partialShape2str({inShape.first}) << "_";
        result << "TS=";
        for (const auto& shape : inShape.second) {
            result << "(";
            result << ov::test::utils::vec2str(shape);
            result << ")_";
        }
        result << "axis=" << axis << "_";
        return result.str();
    }
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        ov::element::Type model_type;
        ov::test::InputShape inShape;
        int64_t axis;
        std::tie(model_type, inShape, axis) = this->GetParam();

        if (model_type == ov::element::f16) {
            abs_threshold = 0.005;
        }

        init_input_shapes({inShape});
        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes)
            params.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));

        const auto softMax = std::make_shared<ov::op::v1::Softmax>(params.at(0), axis);
        auto makeFunction = [](ov::ParameterVector &params, const std::shared_ptr<ov::Node> &lastNode) {
            ov::ResultVector results;

            for (size_t i = 0; i < lastNode->get_output_size(); i++)
                results.push_back(std::make_shared<ov::op::v0::Result>(lastNode->output(i)));

            return std::make_shared<ov::Model>(results, params, "ShapeOfLayerGPUTest");
        };
        function = makeFunction(params, softMax);
    }
};

TEST_P(SoftMaxLayerGPUTest, Inference) {
    run();
}

const std::vector<ov::element::Type> netPrecisions = {
       ov::element::f32, ov::element::f16
};

const std::vector<int64_t> axis2D = {0, 1};

const std::vector<ov::test::InputShape> inputShapes2D = {
    {
        {-1, -1},
        {{10, 10}, {10, 306}, {10, 10}, {10, 5}}
    },
    {
        {{2, 100}, {5, 400}},
        {{10, 10}, {10, 306}, {10, 10}, {10, 5}}
    }
};

const std::vector<int64_t> axis4D = {1, 2, 3};

const std::vector<ov::test::InputShape> inputShapes4D = {
    {
        {-1, -1, -1, -1},
        {{10, 10, 20, 20}, {10, 306, 10, 10}, {3, 304, 4, 4}, {4, 18, 19, 19}}
    },
    {
        {{20, 100}, 12, {1, 400}, {1, 300}},
        {{30, 12, 10, 20}, {20, 12, 5, 40}}
    },
    { // Excessive upper bound case
        {3, 8, {128, 16384}, {128, 16384}},
        {{3, 8, 128, 128}}
    },
};

const std::vector<int64_t> axis5D = {0, 1, 2, 4};

const std::vector<ov::test::InputShape> inputShapes5D = {
    {
        {-1, -1, -1, -1, -1},
        {{10, 1, 10, 20, 20}, {10, 306, 2, 2, 1}, {3, 304, 4, 1, 4}}
    },
    {
        {{20, 30}, 12, 10, {1, 10}, {1, 40}},
        {{30, 12, 10, 4, 20}, {20, 12, 10, 5, 40}}
    }
};

INSTANTIATE_TEST_SUITE_P(softMaxGPUDynamicTest2D,
        SoftMaxLayerGPUTest,
                         ::testing::Combine(testing::ValuesIn(netPrecisions),
                                            testing::ValuesIn(inputShapes2D),
                                            testing::ValuesIn(axis2D)),
                         SoftMaxLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(softMaxGPUDynamicTest4D,
        SoftMaxLayerGPUTest,
                         ::testing::Combine(testing::ValuesIn(netPrecisions),
                                            testing::ValuesIn(inputShapes4D),
                                            testing::ValuesIn(axis4D)),
                         SoftMaxLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(softMaxGPUDynamicTest5D,
        SoftMaxLayerGPUTest,
                         ::testing::Combine(testing::ValuesIn(netPrecisions),
                                            testing::ValuesIn(inputShapes5D),
                                            testing::ValuesIn(axis5D)),
                         SoftMaxLayerGPUTest::getTestCaseName);
}  // namespace
