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
        std::vector<InputShape>,            // Input shapes
        std::pair<double, double>,          // Min value, Max value
        std::pair<uint64_t, uint64_t>,      // Global seed, operation seed
        ElementType,                        // Network precision
        TargetDevice,                       // Device name
        std::map<std::string, std::string>  // Additional network configuration
> RandomUnifromDynamicGPUTestParamsSet;

class RandomUnifromDynamicGPUTest : public testing::WithParamInterface<RandomUnifromDynamicGPUTestParamsSet>,
                            virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RandomUnifromDynamicGPUTestParamsSet>& obj) {
        RandomUnifromDynamicGPUTestParamsSet basicParamsSet = obj.param;
        std::ostringstream result;
        std::vector<InputShape> input_shapes;
        std::pair<double, double> min_max_values;
        std::pair<uint64_t, uint64_t> seeds;
        ElementType precision;
        TargetDevice target_device;
        std::map<std::string, std::string> additionalConfig;
        std::tie(input_shapes, min_max_values, seeds, precision, target_device, additionalConfig) = basicParamsSet;

        result << "shape=";
        for (const auto& shape : input_shapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
            for (const auto& actual_shape : shape.second) {
                result << ov::test::utils::partialShape2str({actual_shape}) << "_";
            }
        }
        result << "precision=" << precision << "_";
        result << "min_max_values=" << min_max_values.first << "_" << min_max_values.second << "_";
        result << "seeds=" << seeds.first << "_" << seeds.second << "_";
        result << "target_device=" << target_device;
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
    void set_tensor_value(T scalar, ov::Tensor& tensor) {
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
            if (index != 0) {
                auto scalar_val = index == 1 ? min_max_values.first : min_max_values.second;
                set_tensor_value(scalar_val, tensor);
            }
            inputs.insert({funcInputs[index].get_node_shared_ptr(), tensor});
        };

        for (size_t i = 0; i < targetInputStaticShapes.size(); ++i)
            generate_input(i, funcInputs[i].get_element_type());
    }

    void SetUp() override {
        RandomUnifromDynamicGPUTestParamsSet basicParamsSet = this->GetParam();
        std::vector<InputShape> shapes;
        ElementType netType;
        std::map<std::string, std::string> additionalConfig;
        std::pair<uint64_t, uint64_t> seeds;

        std::tie(shapes, min_max_values, seeds, netType, targetDevice, additionalConfig) = basicParamsSet;

        init_input_shapes(shapes);

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(netType, shape));
        }
        const auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(params[0]);
        const auto random_uniform = std::make_shared<ov::op::v8::RandomUniform>(shape_of, params[1], params[2], netType, seeds.first, seeds.second);

        ov::ResultVector results = {std::make_shared<ov::op::v0::Result>(random_uniform)};
        function = std::make_shared<ov::Model>(results, params, "random_uniform_test");
    }

private:
    std::pair<double, double> min_max_values;
};


TEST_P(RandomUnifromDynamicGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

namespace {
std::map<std::string, std::string> emptyAdditionalConfig;
const std::vector<std::vector<ov::test::InputShape>> dynInputShapes = {
    {
        {{ov::PartialShape::dynamic(4)}, {{1, 2, 3, 4}, {1, 1, 5, 5}, {2, 3, 4, 5}}},
        {{1}, {{1}}},
        {{1}, {{1}}}
    },
    {
        {{ov::PartialShape::dynamic(3)}, {{1, 2, 3}, {1, 1, 5}, {2, 3, 4}}},
        {{1}, {{1}}},
        {{1}, {{1}}}
    },
    {
        {{ov::PartialShape::dynamic(2)}, {{1, 2}, {1, 1}, {2, 3}}},
        {{1}, {{1}}},
        {{1}, {{1}}}
    },
    {
        {{ov::PartialShape::dynamic(1)}, {{1}, {2}, {3}}},
        {{1}, {{1}}},
        {{1}, {{1}}}
    },
};

const std::vector<std::pair<double, double>> min_max_values = {
    {10, 30},
};

const std::vector<std::pair<uint64_t, uint64_t>> seeds = {
    {100, 10},
};

const std::vector<ElementType> netPrecisions = {
    ElementType::i32,
    ElementType::f32,
    ElementType::f16,
};

const auto testParams_smoke = ::testing::Combine(::testing::ValuesIn(dynInputShapes),
                                                 ::testing::ValuesIn(min_max_values),
                                                 ::testing::ValuesIn(seeds),
                                                 ::testing::ValuesIn(netPrecisions),
                                                 ::testing::Values(ov::test::utils::DEVICE_GPU),
                                                 ::testing::Values(emptyAdditionalConfig));

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_random_uniform, RandomUnifromDynamicGPUTest,
                         testParams_smoke, RandomUnifromDynamicGPUTest::getTestCaseName);

} // namespace
} // namespace GPULayerTestsDefinitions
