// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/add.hpp"

namespace {

using ov::test::InputShape;

using StaticEltwiseDynamicFusionsParams = std::tuple<std::vector<InputShape>,   // input shapes
                                                     ov::element::Type>;        // input precision

class StaticEltwiseDynamicFusions : public testing::WithParamInterface<StaticEltwiseDynamicFusionsParams>,
                     virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<StaticEltwiseDynamicFusionsParams> obj) {
        const auto& [input_shapes, input_precision] = obj.param;

        std::ostringstream result;
        result << "IS=(";
        for (const auto& shape : input_shapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=";
        for (const auto& shape : input_shapes) {
            result << "(";
            if (!shape.second.empty()) {
                auto itr = shape.second.begin();
                do {
                    result << ov::test::utils::vec2str(*itr);
                } while (++itr != shape.second.end() && result << "_");
            }
            result << ")_";
        }
        result << "input_precision=" << input_precision;
        return result.str();
    }

protected:
    std::shared_ptr<ov::Model> init_subgraph(std::vector<ov::PartialShape>& input_shapes,
                                             const ov::element::Type input_precision) {
        auto input0 = std::make_shared<ov::op::v0::Parameter>(input_precision, input_shapes[0] /* static input */);
        auto input1 = std::make_shared<ov::op::v0::Parameter>(input_precision, input_shapes[1] /* dynamic input */);

        auto mul_const = ov::op::v0::Constant::create(input_precision, ov::Shape{}, {0.5f});
        auto mul = std::make_shared<ov::op::v1::Multiply>(input0, mul_const);
        auto add = std::make_shared<ov::op::v1::Add>(mul, input1);
        auto mul2 = std::make_shared<ov::op::v1::Multiply>(add, mul_const);

        add->set_friendly_name("Add1");
        mul->set_friendly_name("Mul1");
        mul2->set_friendly_name("Mul2");

        return std::make_shared<ov::Model>(ov::OutputVector{mul2}, ov::ParameterVector{input0, input1}, "StaticEltwiseDynamicFusions");
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const auto& [input_shapes, input_precision] = GetParam();

        init_input_shapes(input_shapes);

        inType = outType = input_precision;
        function = init_subgraph(inputDynamicShapes, input_precision);
    }
};

TEST_P(StaticEltwiseDynamicFusions, Inference) {
    run();
}

const std::vector<ov::element::Type> input_precisions2 = {ov::element::f32, ov::element::f16};

const std::vector<std::vector<InputShape>> input_shapes_dyn2 = {
    {{{2, 2, 32}, {{2, 2, 32}}}, {{-1, -1, 32}, {{1, 1, 32}}}},
};

INSTANTIATE_TEST_SUITE_P(StaticEltwiseDynamicFusions_basic,
                         StaticEltwiseDynamicFusions,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_dyn2),
                                            ::testing::ValuesIn(input_precisions2)),
                         StaticEltwiseDynamicFusions::getTestCaseName);
} // namespace
