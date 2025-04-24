// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "openvino/core/type/element_type_traits.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/range.hpp"

namespace {
using ov::test::InputShape;

typedef std::tuple<
        std::vector<InputShape>,            // input shapes
        std::vector<float>,                 // input values
        ov::element::Type,                  // Model type
        std::string                        // Device name
> RangeDynamicGPUTestParamsSet;

class RangeDynamicGPUTest : public testing::WithParamInterface<RangeDynamicGPUTestParamsSet>,
                            virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RangeDynamicGPUTestParamsSet>& obj) {
        RangeDynamicGPUTestParamsSet basicParamsSet = obj.param;
        std::ostringstream result;
        std::vector<InputShape> inputShapes;
        std::vector<float> inputValues;
        ov::element::Type model_type;
        std::string targetDevice;
        std::tie(inputShapes, inputValues, model_type, targetDevice) = basicParamsSet;

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
        result << "model_type=" << model_type << "_";
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
                auto *dataPtr = tensor.data<ov::element_type_traits<X>::value_type>();        \
                dataPtr[0] = static_cast<ov::element_type_traits<X>::value_type>(scalar);     \
                break;                                                                    \
            }

        switch (tensor.get_element_type()) {
            CASE(ov::element::boolean)
            CASE(ov::element::i8)
            CASE(ov::element::i16)
            CASE(ov::element::i32)
            CASE(ov::element::i64)
            CASE(ov::element::u8)
            CASE(ov::element::u16)
            CASE(ov::element::u32)
            CASE(ov::element::u64)
            CASE(ov::element::bf16)
            CASE(ov::element::f16)
            CASE(ov::element::f32)
            CASE(ov::element::f64)
            CASE(ov::element::u1)
            CASE(ov::element::i4)
            CASE(ov::element::u4)
            default: OPENVINO_THROW("Unsupported element type: ", tensor.get_element_type());
        }
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        auto generate_input = [&](size_t index, ov::element::Type element_type) {
            ov::Tensor tensor(element_type, targetInputStaticShapes[index]);
            add_scalar_to_tensor<float>(input_values[index], tensor);
            inputs.insert({funcInputs[index].get_node_shared_ptr(), tensor});
        };

        // net_type=undifined means mixed type test
        if (net_type == ov::element::dynamic) {
            generate_input(0, ov::element::f32);
            generate_input(1, ov::element::i32);
            generate_input(2, ov::element::f32);
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
        ov::element::Type model_type;
        ov::ParameterVector params;
        std::tie(inputShapes, inputValues, model_type, targetDevice) = basicParamsSet;

        input_values = inputValues;
        net_type = model_type;

        init_input_shapes(inputShapes);

        if (model_type == ov::element::dynamic) {
            std::vector<ov::element::Type> types = { ov::element::f32, ov::element::i32, ov::element::f32 };
            for (size_t i = 0; i < types.size(); i++) {
                auto paramNode = std::make_shared<ov::op::v0::Parameter>(types[i], inputDynamicShapes[i]);
                params.push_back(paramNode);
            }
            model_type = ov::element::f32;
        } else {
            for (auto&& shape : inputDynamicShapes) {
                params.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));
            }
        }
        const auto range = std::make_shared<ov::op::v4::Range>(params[0], params[1], params[2], model_type);

        ov::ResultVector results = {std::make_shared<ov::op::v0::Result>(range)};
        function = std::make_shared<ov::Model>(results, params, "shapeof_out");
    }

private:
    std::vector<float> input_values;
    ov::element::Type net_type;
};


TEST_P(RangeDynamicGPUTest, Inference) {
    run();
}

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

const std::vector<ov::element::Type> netPrecisions = {
    ov::element::i8,
    ov::element::i32,
    ov::element::i64,
};

const auto testParams_smoke = ::testing::Combine(::testing::ValuesIn(dynInputShapes),
                                                 ::testing::ValuesIn(inputValues),
                                                 ::testing::ValuesIn(netPrecisions),
                                                 ::testing::Values(ov::test::utils::DEVICE_GPU));

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_range_01, RangeDynamicGPUTest,
                         testParams_smoke, RangeDynamicGPUTest::getTestCaseName);


const std::vector<std::vector<float>> inputFloatValues = {
    {
        // Inputs for Range
        {1.0f,  2.5f,  0.5f},
        {23.0f, 5.0f, -2.0f},
    }
};

const std::vector<ov::element::Type> netFloatPrecisions = {
    ov::element::f16,
    ov::element::f32,
};

const auto testFloatParams_smoke = ::testing::Combine(::testing::ValuesIn(dynInputShapes),
                                                      ::testing::ValuesIn(inputFloatValues),
                                                      ::testing::ValuesIn(netFloatPrecisions),
                                                      ::testing::Values(ov::test::utils::DEVICE_GPU));

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_range_02, RangeDynamicGPUTest,
                         testFloatParams_smoke, RangeDynamicGPUTest::getTestCaseName);

const std::vector<std::vector<float>> inputMixedValues = {
    {
        // Inputs for Range
        {4.5f, 12.0f, 1.0f},
        {2.5f, 19.0f, 1.1f},
    }
};

const std::vector<ov::element::Type> netMixedPrecisions = {
    // Mixed type test(start/step:fp32, end:i32)
    ov::element::dynamic};

const auto testMixedParams_smoke = ::testing::Combine(::testing::ValuesIn(dynInputShapes),
                                                      ::testing::ValuesIn(inputMixedValues),
                                                      ::testing::ValuesIn(netMixedPrecisions),
                                                      ::testing::Values(ov::test::utils::DEVICE_GPU));

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_diff_types, RangeDynamicGPUTest,
                         testMixedParams_smoke, RangeDynamicGPUTest::getTestCaseName);
} // namespace
