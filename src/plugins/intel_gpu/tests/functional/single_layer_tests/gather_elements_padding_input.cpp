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
#include "openvino/op/variadic_split.hpp"
#include "openvino/op/gather_elements.hpp"

namespace {

using ov::test::InputShape;

using GatherElementsPaddingInputParams = std::tuple<std::tuple<
                                                        std::vector<ov::Shape>, // input shapes for VadiadicSplit and GatherElements
                                                        std::vector<int64_t>,   // axis for VadiadicSplit and GatherElements
                                                        std::vector<size_t>>,   // numSplits for VariadicSplit
                                                    ov::element::Type>;         // input precision

class GatherElementsPaddingInputTest : public testing::WithParamInterface<GatherElementsPaddingInputParams>,
                     virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GatherElementsPaddingInputParams> obj) {
        std::tuple<std::vector<ov::Shape>, std::vector<int64_t>, std::vector<size_t>> input_params;
        std::vector<ov::Shape> input_shape;
        std::vector<int64_t> axis;
        std::vector<size_t> numSplits;
        ov::element::Type input_precision;

        std::tie(input_params, input_precision) = obj.param;
        std::tie(input_shape, axis, numSplits) = input_params;

        std::ostringstream result;
        result << "IS=(";
        result << ov::test::utils::vec2str(input_shape) << "_";
        result << "input_precision=" << input_precision;
        return result.str();
    }

protected:
    std::shared_ptr<ov::Model> init_subgraph(std::tuple<std::vector<ov::Shape>, std::vector<int64_t>, std::vector<size_t>>& input_params,
                                             const ov::element::Type input_precision) {
        std::vector<ov::Shape> input_shape;
        std::vector<int64_t> axis;
        std::vector<size_t> numSplits;
        std::tie(input_shape, axis, numSplits) = input_params;

        int64_t axis_variadicSplit = axis[0];
        int64_t axis_gatherElements = axis[1];
        std::vector<size_t> connectIndexes = {0, 1};
        ov::ParameterVector input{std::make_shared<ov::op::v0::Parameter>(input_precision, ov::Shape(input_shape[0])),
                                  std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape(input_shape[1])),
                                  std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape(input_shape[2]))};

        // Use VariadicSplit to make padding input for GatherElements. Padding is generated from buffer fusing pass
        auto split_axis_op = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, axis_variadicSplit);
        auto num_split = std::make_shared<ov::op::v0::Constant>(ov::element::u64, ov::Shape{numSplits.size()}, numSplits);
        auto split = std::make_shared<ov::op::v1::VariadicSplit>(input[0], split_axis_op, num_split);

        ov::ResultVector results;
        for (size_t i : connectIndexes) {
            auto gather_elements = std::make_shared<ov::op::v6::GatherElements>(split->output(i), input[i + 1], axis_gatherElements);

            results.push_back(std::make_shared<ov::op::v0::Result>(gather_elements));
        }
        return std::make_shared<ov::Model>(results, input, "gather_elements_pad");
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        std::tuple<std::vector<ov::Shape>, std::vector<int64_t>, std::vector<size_t>> input_params;
        ov::element::Type input_precision;

        std::tie(input_params, input_precision) = GetParam();

        inType = outType = input_precision;
        function = init_subgraph(input_params, input_precision);
    }
};

TEST_P(GatherElementsPaddingInputTest, Inference) {
    run();
}

const std::vector<ov::element::Type> input_precisions = {ov::element::f16, ov::element::f32};

const std::vector<std::tuple<std::vector<ov::Shape>, std::vector<int64_t>, std::vector<size_t>>> input_shapes = {
    {{{1, 8400, 84, 1}, {1, 300, 4, 1}, {1, 300, 80, 1}}, {2, 1}, {4, 80}},
    {{{10, 840, 100, 1}, {10, 400, 30, 1}, {10, 440, 30, 1}}, {1, 2}, {400, 440}},
    {{{100, 20, 10, 100}, {10, 20, 10, 40}, {10, 20, 10, 60}}, {3, 0}, {40, 60}},
    {{{1, 20, 30}, {1, 10, 5}, {1, 10, 5}}, {1, 2}, {10, 10}},
    {{{1, 20, 30, 5, 20}, {1, 20, 15, 5, 5}, {1, 20, 15, 5, 5}}, {2, 4}, {15, 15}},
    {{{1, 20, 30, 5, 5, 20}, {1, 20, 15, 5, 5, 5}, {1, 20, 15, 5, 5, 5}}, {2, 5}, {15, 15}},
};

INSTANTIATE_TEST_SUITE_P(Smoke_GatherElementsPaddingInput,
                         GatherElementsPaddingInputTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(input_precisions)),
                         GatherElementsPaddingInputTest::getTestCaseName);
} // namespace
