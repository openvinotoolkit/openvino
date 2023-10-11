// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

using namespace ngraph;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

typedef std::tuple<
        std::vector<InputShape>,            // input shapes
        std::vector<float>,                 // input values
        ElementType,                        // Network precision
        TargetDevice,                       // Device name
        std::map<std::string, std::string>  // Additional network configuration
> RangeDynamicGPUTestParamsSet;

class RangeDynamicGPUTest : public testing::WithParamInterface<RangeDynamicGPUTestParamsSet>,
                            virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RangeDynamicGPUTestParamsSet>& obj) {
        RangeDynamicGPUTestParamsSet basicParamsSet = obj.param;
        std::ostringstream result;
        std::vector<InputShape> inputShapes;
        std::vector<float> inputValues;
        ElementType netType;
        TargetDevice targetDevice;
        std::map<std::string, std::string> additionalConfig;
        std::tie(inputShapes, inputValues, netType, targetDevice, additionalConfig) = basicParamsSet;

        result << "IS=";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
            for (const auto& actual_shape : shape.second) {
                result << ov::test::utils::partialShape2str({actual_shape}) << "_";
            }
        }
        result << "IV=";
        for (const auto& v : inputValues) {
            result << v << "_";
        }
        result << "NetType=" << netType << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

protected:
    void init_input_shapes(const std::vector<InputShape>& shapes) {
        if (shapes.empty()) {
            targetStaticShapes = {{}};
            return;
        }
        size_t targetStaticShapeSize = shapes.front().second.size();
        for (size_t i = 1; i < shapes.size(); ++i) {
            if (targetStaticShapeSize < shapes[i].second.size()) {
                targetStaticShapeSize = shapes[i].second.size();
            }
        }
        targetStaticShapes.resize(targetStaticShapeSize);

        for (const auto& shape : shapes) {
            auto dynShape = shape.first;
            inputDynamicShapes.push_back(dynShape);
            for (size_t i = 0; i < targetStaticShapeSize; ++i) {
                targetStaticShapes[i].push_back(i < shape.second.size() ? shape.second.at(i) : shape.second.back());
            }
        }
    }

    template<typename T>
    void add_scalar_to_tensor(T scalar, ov::Tensor& tensor) {
        #define CASE(X)                                                                   \
            case X: {                                                                     \
                auto *dataPtr = tensor.data<element_type_traits<X>::value_type>();        \
                dataPtr[0] = static_cast<element_type_traits<X>::value_type>(scalar);     \
                break;                                                                    \
            }

        switch (tensor.get_element_type()) {
            CASE(ElementType::boolean)
            CASE(ElementType::i8)
            CASE(ElementType::i16)
            CASE(ElementType::i32)
            CASE(ElementType::i64)
            CASE(ElementType::u8)
            CASE(ElementType::u16)
            CASE(ElementType::u32)
            CASE(ElementType::u64)
            CASE(ElementType::bf16)
            CASE(ElementType::f16)
            CASE(ElementType::f32)
            CASE(ElementType::f64)
            CASE(ElementType::u1)
            CASE(ElementType::i4)
            CASE(ElementType::u4)
            default: OPENVINO_THROW("Unsupported element type: ", tensor.get_element_type());
        }
    }

    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        auto generate_input = [&](size_t index, ElementType element_type) {
            ov::Tensor tensor(element_type, targetInputStaticShapes[index]);
            add_scalar_to_tensor<float>(input_values[index], tensor);
            inputs.insert({funcInputs[index].get_node_shared_ptr(), tensor});
        };

        // net_type=undifined means mixed type test
        if (net_type == ElementType::undefined) {
            generate_input(0, ElementType::f32);
            generate_input(1, ElementType::i32);
            generate_input(2, ElementType::f32);
        } else {
            for (size_t i = 0; i < funcInputs.size(); ++i) {
                generate_input(i, funcInputs[i].get_element_type());
            }
        }
    }

    void SetUp() override {
        RangeDynamicGPUTestParamsSet basicParamsSet = this->GetParam();
        std::vector<InputShape> inputShapes;
        std::vector<float> inputValues;
        ElementType netType;
        std::map<std::string, std::string> additionalConfig;
        ov::ParameterVector params;
        std::tie(inputShapes, inputValues, netType, targetDevice, additionalConfig) = basicParamsSet;

        input_values = inputValues;
        net_type = netType;

        init_input_shapes(inputShapes);

        if (netType == ElementType::undefined) {
            std::vector<element::Type> types = { ElementType::f32, ElementType::i32, ElementType::f32 };
            for (size_t i = 0; i < types.size(); i++) {
                auto paramNode = std::make_shared<ov::op::v0::Parameter>(types[i], inputDynamicShapes[i]);
                params.push_back(paramNode);
            }
            netType = ElementType::f32;
        } else {
            for (auto&& shape : inputDynamicShapes) {
                params.push_back(std::make_shared<ov::op::v0::Parameter>(netType, shape));
            }
        }
        const auto range = std::make_shared<ngraph::opset8::Range>(params[0], params[1], params[2], netType);

        ngraph::ResultVector results = {std::make_shared<ngraph::opset1::Result>(range)};
        function = std::make_shared<ngraph::Function>(results, params, "shapeof_out");
    }

private:
    std::vector<float> input_values;
    ElementType net_type;
};


TEST_P(RangeDynamicGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

namespace {
std::map<std::string, std::string> emptyAdditionalConfig;
const std::vector<std::vector<ov::test::InputShape>> dynInputShapes = {
    {
        // Inputs for Range
        {{ov::PartialShape::dynamic(0)}, {{}}},
        {{ov::PartialShape::dynamic(0)}, {{}}},
        {{ov::PartialShape::dynamic(0)}, {{}}}
    }
};

const std::vector<std::vector<float>> inputValues = {
    {
        // Inputs for Range
        {2,  23, 3},
        {1,  21, 2},
        {23, 2, -3},
        {4,  0, -1},
    }
};

const std::vector<ElementType> netPrecisions = {
    ElementType::i8,
    ElementType::i32,
    ElementType::i64,
};

const auto testParams_smoke = ::testing::Combine(::testing::ValuesIn(dynInputShapes),
                                                 ::testing::ValuesIn(inputValues),
                                                 ::testing::ValuesIn(netPrecisions),
                                                 ::testing::Values(ov::test::utils::DEVICE_GPU),
                                                 ::testing::Values(emptyAdditionalConfig));

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_range_01, RangeDynamicGPUTest,
                         testParams_smoke, RangeDynamicGPUTest::getTestCaseName);


const std::vector<std::vector<float>> inputFloatValues = {
    {
        // Inputs for Range
        {1.0f,  2.5f,  0.5f},
        {23.0f, 5.0f, -2.0f},
    }
};

const std::vector<ElementType> netFloatPrecisions = {
    ElementType::f16,
    ElementType::f32,
};

const auto testFloatParams_smoke = ::testing::Combine(::testing::ValuesIn(dynInputShapes),
                                                      ::testing::ValuesIn(inputFloatValues),
                                                      ::testing::ValuesIn(netFloatPrecisions),
                                                      ::testing::Values(ov::test::utils::DEVICE_GPU),
                                                      ::testing::Values(emptyAdditionalConfig));

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_range_02, RangeDynamicGPUTest,
                         testFloatParams_smoke, RangeDynamicGPUTest::getTestCaseName);

const std::vector<std::vector<float>> inputMixedValues = {
    {
        // Inputs for Range
        {4.5f, 12.0f, 1.0f},
        {2.5f, 19.0f, 1.1f},
    }
};

const std::vector<ElementType> netMixedPrecisions = {
    // Mixed type test(start/step:fp32, end:i32)
    ElementType::undefined
};


const auto testMixedParams_smoke = ::testing::Combine(::testing::ValuesIn(dynInputShapes),
                                                      ::testing::ValuesIn(inputMixedValues),
                                                      ::testing::ValuesIn(netMixedPrecisions),
                                                      ::testing::Values(ov::test::utils::DEVICE_GPU),
                                                      ::testing::Values(emptyAdditionalConfig));

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_diff_types, RangeDynamicGPUTest,
                         testMixedParams_smoke, RangeDynamicGPUTest::getTestCaseName);
} // namespace
} // namespace GPULayerTestsDefinitions
