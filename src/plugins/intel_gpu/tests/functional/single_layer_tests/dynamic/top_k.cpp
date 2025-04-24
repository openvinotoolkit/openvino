// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <random>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/topk.hpp"

namespace {
using ov::test::InputShape;

typedef std::tuple<
        int64_t,                           // keepK
        int64_t,                           // axis
        ov::op::v1::TopK::Mode,            // mode
        ov::op::v1::TopK::SortType,        // sort
        ov::element::Type,                 // Model type
        ov::element::Type,                 // Input precision
        ov::element::Type,                 // Output precision
        InputShape,                        // input_shape
        std::string,                       // Device name
        ov::test::utils::InputLayerType    // Input type
> TopKLayerTestParamsSet;

class TopKLayerGPUTest : public testing::WithParamInterface<TopKLayerTestParamsSet>,
                         virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TopKLayerTestParamsSet>& obj) {
        TopKLayerTestParamsSet basicParamsSet = obj.param;

        int64_t keepK, axis;
        ov::op::v1::TopK::Mode mode;
        ov::op::v1::TopK::SortType sort;
        ov::element::Type model_type, inPrc, outPrc;
        InputShape input_shape;
        std::string targetDevice;
        ov::test::utils::InputLayerType input_type;
        std::tie(keepK, axis, mode, sort, model_type, inPrc, outPrc, input_shape, targetDevice, input_type) = basicParamsSet;

        std::ostringstream result;
        result << "k=" << keepK << "_";
        result << "axis=" << axis << "_";
        result << "mode=" << mode << "_";
        result << "sort=" << sort << "_";
        result << "netPRC=" << model_type << "_";
        result << "inPRC=" << inPrc << "_";
        result << "outPRC=" << outPrc << "_";
        result << "IS=" << ov::test::utils::partialShape2str({input_shape.first}) << "_" << "TS=(";
        for (const auto& shape : input_shape.second) {
            result << ov::test::utils::vec2str(shape) << "_";
        }
        result << ")_";
        result << "input_type=" << input_type;
        result << "TargetDevice=" << targetDevice;

        return result.str();
    }

protected:
    void SetUp() override {
        TopKLayerTestParamsSet basicParamsSet = this->GetParam();

        int64_t keepK;
        ov::op::v1::TopK::Mode mode;
        ov::op::v1::TopK::SortType sort;
        ov::element::Type inPrc, outPrc;
        InputShape input_shape;
        std::tie(keepK, axis, mode, sort, model_type, inPrc, outPrc, input_shape, targetDevice, input_type) = basicParamsSet;

        if (input_type == ov::test::utils::InputLayerType::CONSTANT) {
            init_input_shapes({input_shape});
        } else {
            inputDynamicShapes = {input_shape.first, {}};
            for (size_t i = 0; i < input_shape.second.size(); ++i) {
                targetStaticShapes.push_back({input_shape.second[i], {}});
            }
        }

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0])};

        std::shared_ptr<ov::op::v1::TopK> topk;
        if (input_type == ov::test::utils::InputLayerType::CONSTANT) {
            auto k = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, &keepK);
            topk = ov::as_type_ptr<ov::op::v1::TopK>(std::make_shared<ov::op::v1::TopK>(params[0], k, axis, mode, sort));
        } else {
            auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputDynamicShapes[1]);
            params.push_back(k);
            topk = ov::as_type_ptr<ov::op::v1::TopK>(
                    std::make_shared<ov::op::v1::TopK>(params[0], k, axis, mode, sort));
        }

        ov::ResultVector results;
        for (size_t i = 0; i < topk->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::op::v0::Result>(topk->output(i)));
        }

        function = std::make_shared<ov::Model>(results, params, "TopK");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        auto shape = targetInputStaticShapes.front();
        ov::Tensor tensor;
        tensor = ov::test::utils::create_and_fill_tensor(funcInputs[0].get_element_type(), shape);
        size_t size = tensor.get_size();

        if (model_type == ov::element::f32) {
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
            FAIL() << "generate_inputs for " << model_type << " precision isn't supported";
        }
        inputs.insert({funcInputs[0].get_node_shared_ptr(), tensor});

        if (input_type == ov::test::utils::InputLayerType::PARAMETER) {
            const auto& kPrecision = funcInputs[1].get_element_type();
            const auto& kShape = targetInputStaticShapes[1];

            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = 1;
            in_data.range = targetInputStaticShapes[0][axis];
            in_data.seed = inferRequestNum++;
            const auto kTensor = ov::test::utils::create_and_fill_tensor(kPrecision, kShape, in_data);

            inputs.insert({funcInputs[1].get_node_shared_ptr(), kTensor});
        }
    }

private:
    int64_t axis;
    size_t inferRequestNum = 0;
    ov::element::Type model_type;
    ov::test::utils::InputLayerType input_type;
};

TEST_P(TopKLayerGPUTest, Inference) {
    run();
}

const std::vector<ov::element::Type> model_types = {
    ov::element::f32,
};

const std::vector<int64_t> axes = {0, 3};
const std::vector<int64_t> k = {3, 5, 7};

const std::vector<ov::op::v1::TopK::Mode> modes = {
    ov::op::v1::TopK::Mode::MIN,
    ov::op::v1::TopK::Mode::MAX
};

const std::vector<ov::op::v1::TopK::SortType> sortTypes = {
    ov::op::v1::TopK::SortType::SORT_VALUES,
    ov::op::v1::TopK::SortType::SORT_INDICES,
};

std::vector<ov::test::InputShape> input_shapesDynamic = {
    {
        {ov::PartialShape::dynamic(4), {{7, 7, 7, 7}, {7, 8, 7, 9}}},
        {{-1, -1, -1, -1}, {{8, 9, 10, 11}, {11, 7, 8, 9}}}
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_TopK_constant_dynamic,
                         TopKLayerGPUTest,
                         ::testing::Combine(::testing::ValuesIn(k),
                                            ::testing::ValuesIn(axes),
                                            ::testing::ValuesIn(modes),
                                            ::testing::ValuesIn(sortTypes),
                                            ::testing::ValuesIn(model_types),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::ValuesIn(input_shapesDynamic),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::Values(ov::test::utils::InputLayerType::CONSTANT)),
                         TopKLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TopK_parameter_dynamic,
                         TopKLayerGPUTest,
                         ::testing::Combine(::testing::Values(1),
                                            ::testing::ValuesIn(axes),
                                            ::testing::ValuesIn(modes),
                                            ::testing::ValuesIn(sortTypes),
                                            ::testing::ValuesIn(model_types),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::ValuesIn(input_shapesDynamic),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::Values(ov::test::utils::InputLayerType::PARAMETER)),
                         TopKLayerGPUTest::getTestCaseName);

} // namespace
