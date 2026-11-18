// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "openvino/core/type/element_type_traits.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/random_uniform.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/reference/random_uniform.hpp"

namespace {
using ov::test::InputShape;

typedef std::tuple<
        std::vector<InputShape>,            // Input shapes
        std::pair<double, double>,          // Min value, Max value
        std::pair<uint64_t, uint64_t>,      // Global seed, operation seed
        ov::element::Type,                  // Network precision
        std::string                        // Device name
> RandomUniformDynamicGPUTestParamsSet;

class RandomUniformDynamicGPUTest : public testing::WithParamInterface<RandomUniformDynamicGPUTestParamsSet>,
                            virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RandomUniformDynamicGPUTestParamsSet>& obj) {
        RandomUniformDynamicGPUTestParamsSet basicParamsSet = obj.param;
        std::ostringstream result;

        const auto& [input_shapes, min_max_values, seeds, precision, target_device] = basicParamsSet;

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
        RandomUniformDynamicGPUTestParamsSet basicParamsSet = this->GetParam();

        const auto& [shapes, _min_max_values, seeds, netType, _targetDevice] = basicParamsSet;
        min_max_values = _min_max_values;
        targetDevice = _targetDevice;
        global_seed = seeds.first;
        op_seed = seeds.second;
        out_type = netType;
        philox_state = {0, 0};

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

    // All target shapes run on one compiled model, and RandomUniform advances its Philox state after every
    // inference (n += 256 * <output elements>, same as the CPU plugin). The template plugin reference starts from
    // the initial state on every call, so compute the reference here with the same state progression.
    std::vector<ov::Tensor> calculate_refs() override {
        const auto& params = function->get_parameters();
        const auto out_shape = inputs.at(params[0]).get_shape();
        std::vector<uint64_t> out_shape_u64(out_shape.begin(), out_shape.end());
        ov::Tensor expected(out_type, out_shape);
        ov::reference::random_uniform(out_shape_u64.data(),
                                      static_cast<const char*>(inputs.at(params[1]).data()),
                                      static_cast<const char*>(inputs.at(params[2]).data()),
                                      static_cast<char*>(expected.data()),
                                      ov::Shape{out_shape.size()},
                                      out_type,
                                      global_seed,
                                      op_seed,
                                      philox_state);
        const uint64_t skip_count = ov::shape_size(out_shape) * 256;
        philox_state.first += skip_count;
        if (philox_state.first < skip_count) {
            philox_state.second++;
        }
        return {expected};
    }

private:
    std::pair<double, double> min_max_values;
    uint64_t global_seed = 0;
    uint64_t op_seed = 0;
    ov::element::Type out_type;
    std::pair<uint64_t, uint64_t> philox_state{0, 0};
};

TEST_P(RandomUniformDynamicGPUTest, Inference) {
    run();
}

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

const std::vector<ov::element::Type> netPrecisions = {
    ov::element::i32,
    ov::element::f32,
    ov::element::f16,
};

const auto testParams_smoke = ::testing::Combine(::testing::ValuesIn(dynInputShapes),
                                                 ::testing::ValuesIn(min_max_values),
                                                 ::testing::ValuesIn(seeds),
                                                 ::testing::ValuesIn(netPrecisions),
                                                 ::testing::Values(ov::test::utils::DEVICE_GPU));

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_random_uniform, RandomUniformDynamicGPUTest,
                         testParams_smoke, RandomUniformDynamicGPUTest::getTestCaseName);
} // namespace
