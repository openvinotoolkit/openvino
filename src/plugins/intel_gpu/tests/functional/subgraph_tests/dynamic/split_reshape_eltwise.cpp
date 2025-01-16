// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/variadic_split.hpp"

namespace {
using ov::test::InputShape;

typedef std::tuple<
        std::vector<InputShape>,           // input shapes
        size_t,                            // split axis
        ov::element::Type,                 // Model type
        std::string                        // Device name
> SplitReshapeEltwiseTestParams;

const std::vector<ov::element::Type> model_precisions = {
    ov::element::f16
};

class SplitReshapeEltwiseTest : public testing::WithParamInterface<SplitReshapeEltwiseTestParams>,
                                virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SplitReshapeEltwiseTestParams>& obj) {
        SplitReshapeEltwiseTestParams test_params = obj.param;
        std::ostringstream result;
        std::vector<InputShape> input_shapes;
        size_t axis;
        ov::element::Type precision;
        std::string target_device;

        std::tie(input_shapes, axis, precision, target_device) = test_params;
        result << "IS=";
        for (const auto& shape : input_shapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
            for (const auto& actual_shape : shape.second) {
                result << ov::test::utils::partialShape2str({actual_shape}) << "_";
            }
        }
        result << "axis=" << axis << "_";
        result << "Precision=" << precision << "_";
        result << "target_device=" << target_device;
        return result.str();
    }

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];

            ov::Tensor tensor;
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = 0;
            in_data.range = 80;
            in_data.resolution = 8;
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

    void SetUp() override {
        SplitReshapeEltwiseTestParams test_params = this->GetParam();
        std::vector<InputShape> input_shapes;
        size_t axis;
        ov::element::Type model_type;
        std::tie(input_shapes, axis, model_type, targetDevice) = test_params;

        init_input_shapes(input_shapes);

        ov::ParameterVector params = {
            std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0]),
            std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[1]),
            std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[2]),
        };

        auto axis_op = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        axis_op->set_friendly_name("axis");

        auto split_sizes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {5, 5});
        split_sizes->set_friendly_name("split_sizes");

        auto split = std::make_shared<ov::op::v1::VariadicSplit>(params[0], axis_op, split_sizes);
        split->set_friendly_name("split");

        auto add_not_reshaped = std::make_shared<ov::op::v1::Add>(split->output(1), params[1]);
        add_not_reshaped->set_friendly_name("add_not_reshaped");

        std::vector<int64_t> target_shape;
        for (auto& d : inputDynamicShapes[2]) {
            target_shape.push_back(d.is_dynamic() ? -1 : d.get_length());
        }
        auto target_shape_node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{target_shape.size()}, target_shape);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(split->output(0), target_shape_node, false);

        auto add_reshaped = std::make_shared<ov::op::v1::Add>(params[2], reshape);
        add_reshaped->set_friendly_name("add_reshaped");

        auto convert1 = std::make_shared<ov::op::v0::Convert>(add_not_reshaped, ov::element::f32);
        auto convert2 = std::make_shared<ov::op::v0::Convert>(add_reshaped, ov::element::f32);

        ov::ResultVector results = {std::make_shared<ov::op::v0::Result>(convert1), std::make_shared<ov::op::v0::Result>(convert2)};
        function = std::make_shared<ov::Model>(results, params, "eltwise_add_out");
    }
};

TEST_P(SplitReshapeEltwiseTest, Inference) {
    run();
}

const std::vector<std::vector<ov::test::InputShape>> input_shapes = {
    {
        {{-1, 10}, {{2, 10}, {1, 10}}},          // split in shape
        {{-1, 5}, {{2, 5}, {1, 5}}},             // not reshaped add input shape
        {{-1, 1, 5}, {{2, 1, 5}, {1, 1, 5}}}     // reshaped add input shape
    },
};


const auto testParams_smoke = ::testing::Combine(::testing::ValuesIn(input_shapes),
                                                 ::testing::Values(1), // axis
                                                 ::testing::ValuesIn(model_precisions),
                                                 ::testing::Values(ov::test::utils::DEVICE_GPU));

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_model, SplitReshapeEltwiseTest,
                         testParams_smoke, SplitReshapeEltwiseTest::getTestCaseName);
} // namespace
