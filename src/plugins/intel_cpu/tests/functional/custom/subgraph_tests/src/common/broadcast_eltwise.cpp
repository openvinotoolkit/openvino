// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;
namespace ov {
namespace test {

using BroadcastEltwiseParams = std::tuple<
    ElementType, // input precision
    InputShape,  // input shape
    ov::Shape    // target broadcast shape
>;

class BroadcastEltwise : virtual public SubgraphBaseTest,
                      public CPUTestsBase,
                      public testing::WithParamInterface<BroadcastEltwiseParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<BroadcastEltwiseParams>& obj) {
        ElementType input_precision;
        InputShape input_shape;
        ov::Shape target_shape;
        std::tie(input_precision, input_shape, target_shape) = obj.param;

        std::ostringstream result;
        result << "precision=" << input_precision << "IS=(" << ov::test::utils::partialShape2str({input_shape.first}) << ")_TS=(";
        for (const auto& item : input_shape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
        result << ")_target_shape=" << ov::test::utils::vec2str(target_shape);
        return result.str();
    }

protected:
    void SetUp() override {
        ElementType input_precision;
        InputShape input_shape;
        std::tie(input_precision, input_shape, target_shape) = GetParam();
        targetDevice = ov::test::utils::DEVICE_CPU;

        std::vector<InputShape> input_shapes{input_shape, {{}, {{target_shape.size()}}}};
        init_input_shapes(input_shapes);

        ov::element::TypeVector input_precisions{input_precision, ov::element::i64};
        ov::ParameterVector params;
        for (size_t i = 0; i < input_precisions.size(); i++) {
            auto param_node = std::make_shared<ov::op::v0::Parameter>(input_precisions[i], inputDynamicShapes[i]);
            params.push_back(param_node);
        }
        const auto bcast_data = ov::opset10::Constant::create(input_precision, {}, {1.f});
        const auto bcast = std::make_shared<ov::opset10::Broadcast>(bcast_data, params[1]);
        const auto add = std::make_shared<ov::opset10::Add>(params[0], bcast);
        function = std::make_shared<ov::Model>(add, params);
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        auto data_tensor = ov::test::utils::create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0]);
        inputs.insert({funcInputs[0].get_node_shared_ptr(), data_tensor});

        auto shape_tensor = ov::Tensor{ov::element::i64, targetInputStaticShapes[1]};
        auto data = shape_tensor.data<ov::element_type_traits<ov::element::i64>::value_type>();
        for (size_t i = 0; i < target_shape.size(); i++) {
            data[i] = target_shape[i];
        }
        inputs.insert({funcInputs[1].get_node_shared_ptr(), shape_tensor});
    }

    void CheckLastNode(const ov::CompiledModel& execNet) {
        const auto model = execNet.get_runtime_model();
        const auto last_node = model->get_result()->get_input_node_shared_ptr(0);
        const auto& rt_info = last_node->get_rt_info();
        const auto layer_type = rt_info.find("layerType")->second.as<std::string>();
        EXPECT_EQ(layer_type, "Broadcast");

        const bool data_shape_exceeds_target = [&]() {
            const auto& in_data_shape = model->get_parameters()[0]->get_output_shape(0);
            if (in_data_shape.size() < target_shape.size()) {
                return false;
            }
            auto bcasted_target = target_shape;
            bcasted_target.insert(bcasted_target.begin(), in_data_shape.size() - bcasted_target.size(), 1);
            for (size_t i = 0; i < in_data_shape.size(); ++i) {
                if (in_data_shape[i] < bcasted_target[i]) {
                    return false;
                }
            }
            return true;
        }();

        // If data shape exceeds original target shape, Broadcast must have equal input and output shapes after transition
        if (data_shape_exceeds_target) {
            EXPECT_EQ(last_node->get_input_shape(0), last_node->get_output_shape(0));
        }
    }

private:
    ov::Shape target_shape;
};

TEST_P(BroadcastEltwise, smoke_CompareWithRefs) {
    run();
    CheckLastNode(compiledModel);
}

namespace {
const std::vector<InputShape> input_shapes = {
    {{-1, -1, -1, -1}, {{1, 3, 16, 16}}},
    {{-1, -1}, {{16, 16}}},
};

const std::vector<ov::Shape> target_shapes = {
    {1, 3, 16, 16},
    {1, 3, 16, 1},
    {16, 16},
};

INSTANTIATE_TEST_SUITE_P(smoke_BroadcastEltwise,
                         BroadcastEltwise,
                         ::testing::Combine(::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(target_shapes)),
                         BroadcastEltwise::getTestCaseName);
} // namespace
}  // namespace test
}  // namespace ov