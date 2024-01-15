// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/adaptive_avg_pool.hpp"
#include "openvino/op/adaptive_max_pool.hpp"

namespace {
using ov::test::InputShape;

// struct AdaptivePoolingShapeParams {
//     InputShape inputShapes;
//     std::vector<int> pooledVector;
// };

using AdaptivePoolingShapeParams = std::tuple<std::vector<int>,          // pooled vector
                                         std::vector<InputShape>>;  // feature map shape

typedef std::tuple<
        AdaptivePoolingShapeParams,
        ov::element::Type,     // Model type
        bool,                  // Is Max Pooling
        ov::element::Type     // Index Elements Type
> AdaptivePoolingGPUTestParams;


class AdaptivePoolingGPUTest : public testing::WithParamInterface<AdaptivePoolingGPUTestParams>,
                        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<AdaptivePoolingGPUTestParams> obj) {
        AdaptivePoolingShapeParams Shapes;
        ov::element::Type model_type;
        bool isMaxPooling;
        ov::element::Type index_type;
        std::vector<int> pooledSpatialShape;
        InputShape inputShapes;

        std::tie(Shapes, model_type, isMaxPooling, index_type) = obj.param;
        std::tie(pooledSpatialShape, inputShapes) = Shapes;

        std::ostringstream result;
        result << "IS=(";
        result << ov::test::utils::partialShape2str({inputShapes.first}) << "_";
        for (size_t i = 0lu; i < inputShapes.second.size(); i++) {
            result << "{";
            result << ov::test::utils::vec2str(inputShapes.second[i]) << "_";
            result << "}_";
        }
        result << "TS=(";
        result << ov::test::utils::vec2str(pooledSpatialShape) << "_";
        result << "netPrc=" << model_type << "_";
        result << "MaxPooling=" << (isMaxPooling ? "True" : "False") << "_";

        if (isMaxPooling == true)
            result << "idxType=" << index_type << "_";

        return result.str();
    }

protected:
    void SetUp() override {
        AdaptivePoolingShapeParams Shapes;
        ov::element::Type model_type;
        bool isMaxPooling;
        ov::element::Type index_type = ov::element::i64;
        std::vector<int> pooledSpatialShape;
        InputShape inputShapes;

        std::tie(Shapes, model_type, isMaxPooling, index_type) = this->GetParam();
        std::tie(pooledVector, inputShapes) = Shapes;
        targetDevice = ov::test::utils::DEVICE_GPU;

        init_input_shapes({inputShapes});

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0])};
        params.back()->set_friendly_name("data");

        params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{pooledVector.size()}););
        params.back()->set_friendly_name("output_shape");


        auto adaptiveAvgPoolNode = std::make_shared<ov::op::v8::AdaptiveAvgPool>(params[0], params[1]);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(adaptiveAvgPoolNode)};
        function = std::make_shared<ov::Model>(results, params, "AdaptivePooling");
    }

    // void generatePooledVector() {
    //     std::random_device rd;
    //     std::uniform_int_distribution<int32_t> distribution(1, 5);
    //     for (size_t i = 0; i < pooledVector.size(); i++) {
    //         pooledVector[i] = distribution(rd);
    //     }
    // }

    // std::shared_ptr<ov::Model> createFunction(bool secondInputConst) {
    //     ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0])};
    //     params.front()->set_friendly_name("ParamsInput");
    //     std::shared_ptr<ov::Node> secondInput;
    //     if (secondInputConst) {
    //         secondInput = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{pooledVector.size()}, pooledVector);
    //     } else {
    //         auto pooledParam =
    //             std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{pooledVector.size()});
    //         pooledParam->set_friendly_name("ParamSecondInput");
    //         params.push_back(pooledParam);
    //         secondInput = pooledParam;
    //     }

    //     auto adapoolMax = std::make_shared<ov::op::v8::AdaptiveMaxPool>(params[0], secondInput, ov::element::i32);
    //     adapoolMax->get_rt_info() = getCPUInfo();
    //     auto adapoolAvg = std::make_shared<ov::op::v8::AdaptiveAvgPool>(params[0], secondInput);
    //     adapoolAvg->get_rt_info() = getCPUInfo();

    //     auto function = (mode == "max" ? std::make_shared<ov::Model>(adapoolMax->outputs(), params, "AdaPoolMax")
    //                                    : std::make_shared<ov::Model>(adapoolAvg->outputs(), params, "AdaPoolAvg"));
    //     return function;
    // }

    void validate() override {
        auto actualOutputs = get_plugin_outputs();
        if (function->get_parameters().size() == 2) {
            auto pos = std::find_if(inputs.begin(),
                                    inputs.end(),
                                    [](const std::pair<std::shared_ptr<ov::Node>, ov::Tensor>& params) {
                                        return params.first->get_friendly_name() == "ParamSecondInput";
                                    });
            OPENVINO_ASSERT(pos != inputs.end());
            inputs.erase(pos);
        }
        auto expectedOutputs = calculate_refs();
        if (expectedOutputs.empty()) {
            return;
        }
        ASSERT_EQ(actualOutputs.size(), expectedOutputs.size())
            << "model interpreter has " << expectedOutputs.size() << " outputs, while IE " << actualOutputs.size();

        compare(expectedOutputs, actualOutputs);
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;

            if (i == 1) {
                tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
                auto* dataPtr = tensor.data<int32_t>();
                for (size_t i = 0; i < pooledVector.size(); i++) {
                    dataPtr[i] = pooledVector[i];
                }
            } else {
                ov::test::utils::InputGenerateData in_data;
                in_data.start_from = 0;
                in_data.range = 2560;
                in_data.resolution = 256;
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

private:
    std::vector<int> pooledVector;
};

TEST_P(AdaptivePoolingGPUTest, Inference) {
    run();
}

const std::vector<ov::element::Type> model_types = {
    ov::element::f32,
    // ov::element::f16
};

const std::vector<ov::element::Type> index_type = {
    ov::element::i64,
    ov::element::i32
};

const std::vector<AdaptivePoolingShapeParams> dynamicInputShapes = {
    {
        ov::test::InputShape(ov::PartialShape({-1, -1, -1}), {{1, 3, 64}}),
        {1},
    },
    // {
    //     ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1}), {{1, 3, 64, 64}}),
    //     ov::test::InputShape(ov::PartialShape({-1}), {{2}}),
    // },
    // {
    //     ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1, -1}), {{1, 5, 64, 64, 64}}),
    //     ov::test::InputShape(ov::PartialShape({-1}), {{3}}),
    // },
};

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_avg, AdaptivePoolingGPUTest,
                ::testing::Combine(
                    ::testing::ValuesIn(dynamicInputShapes),    // input shapes
                    ::testing::ValuesIn(model_types),          // network precision
                    ::testing::Values(false),                     // is Max pool
                    ::testing::Values(ov::element::i32)),
                AdaptivePoolingGPUTest::getTestCaseName);
} // namespace
