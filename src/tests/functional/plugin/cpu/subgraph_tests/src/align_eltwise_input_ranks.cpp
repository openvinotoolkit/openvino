// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include <ngraph/opsets/opset8.hpp>

using namespace ngraph;

namespace SubgraphTestsDefinitions {

enum class OpType {
    SQR_DIFF,
    OR,
    ADD,
    MUL,
    EQUAL,
    LESS,
};

static std::ostream& operator<<(std::ostream& os, const OpType op_type) {
    switch (op_type) {
        case OpType::SQR_DIFF:
            os << "SquaredDifference";
            break;
        case OpType::OR:
            os << "Or";
            break;
        case OpType::ADD:
            os << "Add";
            break;
        case OpType::MUL:
            os << "Multiply";
            break;
        case OpType::EQUAL:
            os << "Equal";
            break;
        case OpType::LESS:
            os << "Less";
            break;
    }
    return os;
}

using AlignEltwiseInputRanksTestParams = std::tuple<OpType, std::pair<Shape, Shape>>;

class AlignEltwiseInputRanksTest : public testing::WithParamInterface<AlignEltwiseInputRanksTestParams>,
                      virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<AlignEltwiseInputRanksTestParams> obj) {
        OpType op_type;
        std::pair<Shape, Shape> shapes;
        std::tie(op_type, shapes) = obj.param;

        std::ostringstream result;
        result << "op_type=" << op_type <<
            "_first_input_shape=" << shapes.first <<
            "_second_input_shape=" << shapes.second;

        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;
        OpType op_type;
        std::pair<Shape, Shape> shapes;
        std::tie(op_type, shapes) = GetParam();
        auto type = element::f32;
        if (op_type == OpType::OR)
            type = element::boolean;

        auto first_input = std::make_shared<opset8::Parameter>(type, shapes.first);
        auto second_input = builder::makeConstant(type, shapes.second, std::vector<float>{}, true);

        std::shared_ptr<Node> op;
        switch (op_type) {
            case OpType::SQR_DIFF: {
                op = std::make_shared<opset8::SquaredDifference>(first_input, second_input);
                break;
            }
            case OpType::OR: {
                op = std::make_shared<opset8::LogicalOr>(first_input, second_input);
                break;
            }
            case OpType::ADD: {
                op = std::make_shared<opset8::Add>(first_input, second_input);
                break;
            }
            case OpType::MUL: {
                op = std::make_shared<opset8::Multiply>(first_input, second_input);
                break;
            }
            case OpType::EQUAL: {
                op = std::make_shared<opset8::Equal>(first_input, second_input);
                break;
            }
            case OpType::LESS: {
                op = std::make_shared<opset8::Less>(first_input, second_input);
                break;
            }
        }
        function = std::make_shared<Function>(op, ParameterVector{first_input});
    }

    void TearDown() override {
        auto runtime_function = executableNetwork.GetExecGraphInfo().getFunction();
        int nodes_found = 0;
        for (const auto& n : runtime_function->get_ordered_ops()) {
            auto layer_type = n->get_rt_info().at(ExecGraphInfoSerialization::LAYER_TYPE).as<std::string>();
            if (layer_type == "Select" || layer_type == "Eltwise") {
                nodes_found++;
                const auto inputs = n->input_values();
                auto rank = inputs[0].get_partial_shape().rank().get_length();
                for (auto it = inputs.begin() + 1; it != inputs.end(); it++) {
                    ASSERT_EQ(rank, it->get_partial_shape().rank().get_length());
                }
            }
        }
        ASSERT_GT(nodes_found, 0);
    }
};

TEST_P(AlignEltwiseInputRanksTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
}

namespace {

const std::vector<OpType> operators = {
    OpType::SQR_DIFF,
    OpType::OR,
    OpType::ADD,
    OpType::MUL,
    OpType::EQUAL,
    OpType::LESS,
};

const std::vector<std::pair<Shape, Shape>> shapes = {
    {Shape{1, 2, 100}, Shape{100}},
    {Shape{1, 2, 100}, Shape{1, 100}},
    {Shape{1, 3, 100, 100}, Shape{100}},
    {Shape{1, 3, 100, 100}, Shape{1, 100}},
    {Shape{1, 3, 100, 100}, Shape{1, 1, 100}},
    {Shape{1, 3, 100, 100}, Shape{3, 1, 1}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Check, AlignEltwiseInputRanksTest,
                         ::testing::Combine(::testing::ValuesIn(operators),
                                            ::testing::ValuesIn(shapes)),
                         AlignEltwiseInputRanksTest::getTestCaseName);

} // namespace


class SubgraphWithBlockedFormat : virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;

        auto type = element::f32;
        auto param = std::make_shared<opset8::Parameter>(type, Shape{1, 32, 64, 32});
        auto weights = builder::makeConstant(type, Shape{32, 32, 1, 1}, std::vector<float>{}, true);
        auto conv = std::make_shared<opset8::Convolution>(param, weights, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});
        auto mean = std::make_shared<opset8::ReduceMean>(conv, opset8::Constant::create(element::i32, Shape{2}, {2, 3}), true);
        auto reshape_before = std::make_shared<opset8::Reshape>(mean, opset8::Constant::create(element::i32, Shape{3}, {0, 16, -1}), true);
        auto mvn = std::make_shared<opset8::MVN>(reshape_before, opset8::Constant::create(element::i32, Shape{1}, {2}),
                false, 0.1, op::MVNEpsMode::INSIDE_SQRT);
        auto reshape_after = std::make_shared<opset8::Reshape>(mvn, std::make_shared<opset8::ShapeOf>(mean), false);
        auto mul = std::make_shared<opset8::Multiply>(reshape_after, builder::makeConstant(type, Shape{32, 1, 1}, std::vector<float>{}, true));
        auto add = std::make_shared<opset8::Add>(mul, builder::makeConstant(type, Shape{32, 1, 1}, std::vector<float>{}, true));
        auto sigmoid = std::make_shared<opset8::Sigmoid>(add);
        auto mul2 = std::make_shared<opset8::Multiply>(conv, sigmoid);

        function = std::make_shared<Function>(mul2, ParameterVector{param});
    }

    void TearDown() override {
        auto runtime_function = executableNetwork.GetExecGraphInfo().getFunction();
        int nodes_found = 0;
        for (const auto& n : runtime_function->get_ordered_ops()) {
            auto layer_type = n->get_rt_info().at(ExecGraphInfoSerialization::LAYER_TYPE).as<std::string>();
            if (layer_type == "Subgraph") {
                nodes_found++;
                auto output_layout = n->get_rt_info().at(ExecGraphInfoSerialization::OUTPUT_LAYOUTS).as<std::string>();
                ASSERT_EQ("aBcd8b", output_layout);
            }
        }
        ASSERT_GT(nodes_found, 0);
    }
};

TEST_F(SubgraphWithBlockedFormat, smoke_CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
}
} // namespace SubgraphTestsDefinitions
