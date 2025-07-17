// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

// Pattern:
/* ******************************************
 *    Input    bias1
 *      |      /   \
 *    Reshape /   Reshape
 *      |   \/     /
 *      |   /\    /
 *      Add    Add
 *      |       |
 *  ShapeOf  ShapeOf
 *       \    /
 *       Result
 ********************************************/
namespace {

using ov::test::InputShape;

using EltwiseMergeParams = std::tuple<std::vector<InputShape>,  // input shapes
                                      ov::element::Type>;       // input precision

class EltwiseMerge : public testing::WithParamInterface<EltwiseMergeParams>, virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<EltwiseMergeParams> obj) {
        std::vector<InputShape> input_shapes;
        ov::element::Type input_precision;

        std::tie(input_shapes, input_precision) = obj.param;

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
    std::shared_ptr<ov::Model> init_subgraph(std::vector<ov::PartialShape>& input_shapes, const ov::element::Type input_precision) {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(input_precision, input_shapes[0]);

        auto bias1 = ov::op::v0::Constant::create(input_precision, ov::Shape{2, 1}, {0.5f, 0.02f});
        bias1->set_friendly_name("Bias1");

        auto target_shape1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {-1, 1, 1});
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(input1, target_shape1, true);
        reshape1->set_friendly_name("Reshape1");

        auto target_shape2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto bias2 = std::make_shared<ov::op::v1::Reshape>(bias1, target_shape2, true);
        bias2->set_friendly_name("Reshape2");

        auto add1 = std::make_shared<ov::op::v1::Add>(reshape1, bias1);
        add1->set_friendly_name("Add1");
        auto add2 = std::make_shared<ov::op::v1::Add>(reshape1, bias2);
        add2->set_friendly_name("Add2");

        auto out_shape1 = std::make_shared<ov::op::v3::ShapeOf>(add1);
        auto out_shape2 = std::make_shared<ov::op::v3::ShapeOf>(add2);

        return std::make_shared<ov::Model>(ov::OutputVector{out_shape1, out_shape2}, ov::ParameterVector{input1}, "EltwiseMerge");
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        std::vector<InputShape> input_shapes;
        ov::element::Type input_precision;

        std::tie(input_shapes, input_precision) = GetParam();

        init_input_shapes(input_shapes);

        inType = outType = input_precision;
        function = init_subgraph(inputDynamicShapes, input_precision);
    }
};

TEST_P(EltwiseMerge, Inference) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
}

const std::vector<ov::element::Type> input_precisions = {ov::element::f32, ov::element::f16};

const std::vector<std::vector<InputShape>> input_shapes_2 = {
    {{{-1}, {{5}, {3}}}},
};

INSTANTIATE_TEST_SUITE_P(EltwiseMerge_basic,
                         EltwiseMerge,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_2), ::testing::ValuesIn(input_precisions)),
                         EltwiseMerge::getTestCaseName);
}  // namespace
