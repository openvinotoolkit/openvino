// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/single_layer/shape_of.hpp"
#include "shared_test_classes/single_layer/strided_slice.hpp"
#include <shared_test_classes/single_layer/eltwise.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>

using namespace ngraph;
using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

typedef std::tuple<
        std::vector<InputShape>,    // input shapes
        std::vector<float>,        // input values
        ElementType,                // Network precision
        TargetDevice,               // Device name
        std::map<std::string, std::string> // Additional network configuration
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
            result << CommonTestUtils::partialShape2str({shape.first}) << "_";
            for (const auto& actual_shape : shape.second) {
                result << CommonTestUtils::partialShape2str({actual_shape}) << "_";
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
    std::shared_ptr<ov::Node> inline generate_constant(ElementType netType, ov::PartialShape& pshape, const float value) {
        std::vector<T> data_vec = {static_cast<T>(value)};
        return builder::makeConstant(netType, pshape.to_shape(), data_vec);
    }

    std::shared_ptr<ngraph::opset8::Range> generate_range_op(ElementType netType, std::vector<ov::PartialShape>& pshapes, std::vector<float>& values) {
        const size_t num_inputs = 3;

        std::vector<std::shared_ptr<ov::Node>> input_vec;
        for (size_t idx = 0; idx < num_inputs; idx++) {
#define CASE(X) case X: input_vec.push_back(generate_constant<element_type_traits<X>::value_type>(netType, inputDynamicShapes[idx], values[idx])); break;
            switch (netType) {
                CASE(ov::element::Type_t::boolean)
                CASE(ov::element::Type_t::i8)
                CASE(ov::element::Type_t::i16)
                CASE(ov::element::Type_t::i32)
                CASE(ov::element::Type_t::i64)
                CASE(ov::element::Type_t::u8)
                CASE(ov::element::Type_t::u16)
                CASE(ov::element::Type_t::u32)
                CASE(ov::element::Type_t::u64)
                CASE(ov::element::Type_t::bf16)
                CASE(ov::element::Type_t::f16)
                CASE(ov::element::Type_t::f32)
                CASE(ov::element::Type_t::f64)
                case ov::element::Type_t::u1:
                case ov::element::Type_t::i4:
                case ov::element::Type_t::u4:
                    input_vec.push_back(generate_constant<uint8_t>(netType, inputDynamicShapes[idx], values[idx])); break;
                default: OPENVINO_UNREACHABLE("Unsupported element type: ", netType);
            }
#undef CASE
        }

        return std::make_shared<ngraph::opset8::Range>(input_vec[0], input_vec[1], input_vec[2], netType);
    }

    void SetUp() override {
        RangeDynamicGPUTestParamsSet basicParamsSet = this->GetParam();
        std::vector<InputShape> inputShapes;
        std::vector<float> inputValues;
        ElementType netType;
        std::map<std::string, std::string> additionalConfig;
        inputValues.clear();
        std::tie(inputShapes, inputValues, netType, targetDevice, additionalConfig) = basicParamsSet;
        auto params = builder::makeDynamicParams(netType, {});

        init_input_shapes(inputShapes);

        const auto range = generate_range_op(netType, inputDynamicShapes, inputValues);
        ngraph::ResultVector results = {std::make_shared<ngraph::opset1::Result>(range)};
        function = std::make_shared<ngraph::Function>(results, params, "shapeof_out");
    }
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
        {23, 2,  -3},
        {4,  0,  -1},
    }
};

const std::vector<ElementType> netPrecisions = {
    ElementType::i8,
    ElementType::i32,
    ElementType::i64,
};


const auto testParams_smoke = ::testing::Combine(::testing::ValuesIn(dynInputShapes),
                                                   ::testing::ValuesIn(inputValues),
                                                   ::testing::ValuesIn(netPrecisions), // netprec
                                                   ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                                   ::testing::Values(emptyAdditionalConfig));

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_range_01, RangeDynamicGPUTest,
                         testParams_smoke, RangeDynamicGPUTest::getTestCaseName);


const std::vector<std::vector<float>> inputFloatValues = {
    {
        // Inputs for Range
        {1.0f,  2.5f,   0.5f},
        {23.0f, 5.0f,   -2.0f},
    }
};

const std::vector<ElementType> netFloatPrecisions = {
    ElementType::f16,
    ElementType::f32,
};


const auto testFloatParams_smoke = ::testing::Combine(::testing::ValuesIn(dynInputShapes),
                                                   ::testing::ValuesIn(inputFloatValues),
                                                   ::testing::ValuesIn(netFloatPrecisions), // netprec
                                                   ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                                   ::testing::Values(emptyAdditionalConfig));

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_range_02, RangeDynamicGPUTest,
                         testFloatParams_smoke, RangeDynamicGPUTest::getTestCaseName);

} // namespace
} // namespace GPULayerTestsDefinitions
