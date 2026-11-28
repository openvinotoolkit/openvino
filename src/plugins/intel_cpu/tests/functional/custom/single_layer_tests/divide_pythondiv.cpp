// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_max.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov::test {

class DividePythonDivCPUTest : public testing::WithParamInterface<bool>, public SubgraphBaseTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        const bool pythondiv = GetParam();
        init_input_shapes({InputShape{{8}, {{8}}}, InputShape{{8}, {{8}}}});

        ov::ParameterVector parameters;
        for (const auto& shape : inputDynamicShapes) {
            parameters.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::i32, shape));
        }
        auto divide = std::make_shared<ov::op::v1::Divide>(parameters[0], parameters[1], pythondiv);
        function = std::make_shared<ov::Model>(ov::OutputVector{divide}, parameters, "DividePythonDiv");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        static constexpr std::array<std::array<int32_t, 8>, 2> values = {
            {{-7, -6, -5, 5, 6, -6, 12, -12}, {3, 3, 3, 3, 3, 3, 3, 3}}};
        const auto& modelInputs = function->inputs();
        for (size_t i = 0; i < modelInputs.size(); ++i) {
            ov::Tensor tensor(ov::element::i32, targetInputStaticShapes[i]);
            std::copy(values[i].begin(), values[i].end(), tensor.data<int32_t>());
            inputs.insert({modelInputs[i].get_node_shared_ptr(), tensor});
        }
    }
};

class DividePythonDivFusedCPUTest : public testing::WithParamInterface<bool>, public SubgraphBaseTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        const bool pythondiv = GetParam();
        init_input_shapes({InputShape{{8}, {{8}}}, InputShape{{8}, {{8}}}});

        ov::ParameterVector parameters;
        for (const auto& shape : inputDynamicShapes) {
            parameters.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::i32, shape));
        }

        auto divide = std::make_shared<ov::op::v1::Divide>(parameters[0], parameters[1], pythondiv);
        auto multiply = std::make_shared<ov::op::v1::Multiply>(
            divide,
            ov::op::v0::Constant::create(ov::element::i32, {1}, {2}));
        auto add = std::make_shared<ov::op::v1::Add>(
            multiply,
            ov::op::v0::Constant::create(ov::element::i32, {1}, {5}));
        function = std::make_shared<ov::Model>(ov::OutputVector{add}, parameters, "DividePythonDivFused");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        static constexpr std::array<std::array<int32_t, 8>, 2> values = {
            {{-7, -6, -5, 5, 6, -6, 12, -12}, {3, 3, 3, 3, 3, 3, 3, 3}}};
        const auto& modelInputs = function->inputs();
        for (size_t i = 0; i < modelInputs.size(); ++i) {
            ov::Tensor tensor(ov::element::i32, targetInputStaticShapes[i]);
            std::copy(values[i].begin(), values[i].end(), tensor.data<int32_t>());
            inputs.insert({modelInputs[i].get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(DividePythonDivCPUTest, CompareWithRefs) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_IntegerDivide,
                         DividePythonDivCPUTest,
                         testing::Values(false, true),
                         [](const testing::TestParamInfo<bool>& info) {
                             return info.param ? "PythonDiv" : "TruncateDiv";
                         });

TEST_P(DividePythonDivFusedCPUTest, CompareWithRefs) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_IntegerDivide,
                         DividePythonDivFusedCPUTest,
                         testing::Values(false, true),
                         [](const testing::TestParamInfo<bool>& info) {
                             return info.param ? "PythonDiv" : "TruncateDiv";
                         });

TEST(smoke_IntegerDivide, DynamicU8) {
    const ov::PartialShape shape{1, -1, -1};
    const auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, shape);
    const auto y = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, shape);
    const auto divide = std::make_shared<ov::op::v1::Divide>(x, y);
    const auto model = std::make_shared<ov::Model>(divide->outputs(), ov::ParameterVector{x, y});
    auto request = utils::PluginCache::get().core()->compile_model(model, utils::DEVICE_CPU).create_infer_request();

    const auto check = [&](const ov::Shape& x_shape,
                           const std::vector<uint8_t>& numerators,
                           const ov::Shape& y_shape,
                           const std::vector<uint8_t>& divisors,
                           const std::vector<uint8_t>& expected) {
        request.set_input_tensor(0, utils::create_tensor(ov::element::u8, x_shape, numerators));
        request.set_input_tensor(1, utils::create_tensor(ov::element::u8, y_shape, divisors));
        request.infer();
        const auto result = request.get_output_tensor();
        ASSERT_EQ(result.get_shape(), x_shape);
        const auto* data = result.data<const uint8_t>();
        EXPECT_EQ(std::vector<uint8_t>(data, data + result.get_size()), expected);
    };

    check({1, 8, 1}, {1, 3, 5, 7, 9, 11, 13, 15}, {1, 1, 1}, {2}, {0, 1, 2, 3, 4, 5, 6, 7});
    check({1, 2, 4},
          {0, 1, 127, 128, 254, 255, 254, 255},
          {1, 2, 4},
          {3, 5, 7, 3, 5, 7, 255, 255},
          {0, 0, 18, 42, 50, 36, 0, 1});
}

class DivideI8PostOpsCPUTest : public testing::WithParamInterface<bool>, public SubgraphBaseTest {
protected:
    void SetUp() override {
        targetDevice = utils::DEVICE_CPU;
        abs_threshold = rel_threshold = 0;
        const InputShape shape{{1, -1, -1}, {{1, 4, 2}}};
        init_input_shapes({shape});
        const auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::i8, inputDynamicShapes[0]);
        const auto axis = ov::op::v0::Constant::create(ov::element::i32, {1}, {2});
        const auto max = std::make_shared<ov::op::v1::ReduceMax>(x, axis, true);
        const auto two = ov::op::v0::Constant::create(ov::element::i8, {}, {2});
        const auto divide = std::make_shared<ov::op::v1::Divide>(max, two, GetParam());
        const auto multiply = std::make_shared<ov::op::v1::Multiply>(divide, two);
        function = std::make_shared<ov::Model>(multiply->outputs(), ov::ParameterVector{x});
    }

    void generate_inputs(const std::vector<ov::Shape>& shapes) override {
        inputs[function->get_parameters()[0]] =
            utils::create_tensor<int8_t>(ov::element::i8, shapes[0], {-8, -7, -4, -3, 2, 3, 6, 7});
    }
};

TEST_P(DivideI8PostOpsCPUTest, CompareWithRefs) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_IntegerDivide, DivideI8PostOpsCPUTest, testing::Bool());

}  // namespace ov::test
