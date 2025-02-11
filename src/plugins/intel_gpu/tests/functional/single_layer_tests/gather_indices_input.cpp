// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"

namespace {

using ov::test::InputShape;

using GatherIndicesInputParams = std::tuple<
                                    ov::Shape,  // Input shapes
                                    ov::Shape,                // Indices shape
                                    std::tuple<int, int>,     // Gather axis and batch
                                    ov::element::Type>;         // input precision

class GatherIndicesInputTest : public testing::WithParamInterface<GatherIndicesInputParams>,
                     virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GatherIndicesInputParams> obj) {
        ov::Shape input_shape;
        ov::Shape indices_shape;
        std::tuple<int, int> axis_batch_idx;
        ov::element::Type input_precision;

        std::tie(input_shape, indices_shape, axis_batch_idx, input_precision) = obj.param;

        std::ostringstream result;
        result << "IS=(";
        result << ov::test::utils::vec2str(input_shape) << "_";
        result << "axis=" << std::get<0>(axis_batch_idx) << "_";
        result << "batch_idx=" << std::get<1>(axis_batch_idx) << "_";
        result << "indices_shape=" << ov::test::utils::vec2str(indices_shape) << "_";
        result << "input_precision=" << input_precision;
        return result.str();
    }

protected:
    ov::Shape input_shape, indices_shape;
    std::tuple<int, int> axis_batch_idx;

    void SetUp() override {
        ov::element::Type input_precision;
        targetDevice = ov::test::utils::DEVICE_GPU;
        std::tie(input_shape, indices_shape, axis_batch_idx, input_precision) = GetParam();

        int axis = std::get<0>(axis_batch_idx);
        int batch_idx = std::get<1>(axis_batch_idx);

        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = -10;
        in_data.range = 10;
        auto input_tensor = ov::test::utils::create_and_fill_tensor(input_precision, input_shape, in_data);
        auto input_node = std::make_shared<ov::op::v0::Constant>(input_tensor);
        auto indices_node = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, indices_shape);
        auto axis_node = ov::op::v0::Constant::create(ov::element::i64, indices_shape, {axis});

        auto gather = std::make_shared<ov::op::v8::Gather>(input_node, indices_node, axis_node, batch_idx);

        auto result = std::make_shared<ov::op::v0::Result>(gather);
        function = std::make_shared<ov::Model>(result, ov::ParameterVector{indices_node}, "gather");
    }

void generate_inputs(const std::vector<ov::Shape>& target_shapes) override {
    inputs.clear();
    const auto& func_inputs = function->inputs();
    auto& data_input = func_inputs[0];
    ov::Tensor tensor;
    ov::test::utils::InputGenerateData in_data;
    in_data.start_from = 0;
    in_data.range = input_shape[std::get<0>(axis_batch_idx)];
    tensor = ov::test::utils::create_and_fill_tensor(ov::element::i64, indices_shape, in_data);
    inputs.insert({data_input.get_node_shared_ptr(), tensor});
}
};

TEST_P(GatherIndicesInputTest, Inference) {
    run();
}

const std::vector<ov::element::Type> input_precisions = {ov::element::f16, ov::element::f32};

const std::vector<ov::Shape> input_shapes = {
    {{271, 300}},
};

const ov::Shape indices_shape = {ov::Shape{}};

const std::tuple<int, int> axis_batch_idx = {0, 0};

// Gather input is constant [271,300]
// Gather indices is parameter [] (scalar)
// Gather axis is constant [] (scalar)
// This graph should not to be selected crop primitive in CreateGatherOpBase()
INSTANTIATE_TEST_SUITE_P(Smoke_GatherIndicesInputTest,
                         GatherIndicesInputTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::Values(indices_shape),
                                            ::testing::Values(axis_batch_idx),
                                            ::testing::ValuesIn(input_precisions)),
                         GatherIndicesInputTest::getTestCaseName);
} // namespace
