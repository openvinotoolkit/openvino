// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <algorithm>

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
        const auto& [input_params, input_precision] = obj.param;
        const auto& [input_shape, axis, numSplits] = input_params;

        std::ostringstream result;
        result << "IS=(";
        result << ov::test::utils::vec2str(input_shape) << "_";
        result << "input_precision=" << input_precision;
        return result.str();
    }

protected:
    int64_t axis_gatherElements;
    std::shared_ptr<ov::Model> init_subgraph(const std::tuple<std::vector<ov::Shape>, std::vector<int64_t>, std::vector<size_t>>& input_params,
                                             const ov::element::Type input_precision) {
        const auto& [input_shape, axis, numSplits] = input_params;

        int64_t axis_variadicSplit = axis[0];
        axis_gatherElements = axis[1];
        std::vector<size_t> connectIndexes = {0, 1};
        ov::ParameterVector input{std::make_shared<ov::op::v0::Parameter>(input_precision, ov::Shape(input_shape[0])),
                                  std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape(input_shape[1])),
                                  std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape(input_shape[2]))};
        input[0]->set_friendly_name("input_data");
        input[1]->set_friendly_name("gather0");
        input[2]->set_friendly_name("gather1");
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

        const auto& [input_params, input_precision] = GetParam();

        inType = outType = input_precision;
        function = init_subgraph(input_params, input_precision);
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        ov::Shape inputDataShape;
        for (auto op : function->get_ordered_ops()) {
            if (ov::is_type<ov::op::v6::GatherElements>(op)) {
                inputDataShape = op->get_input_shape(0);
                break;
            }
        }
        const auto& funcInputs = function->inputs();

        for (size_t i = 0lu; i < funcInputs.size(); i++) {
            const auto& funcInput = funcInputs[i];
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = 0;
            in_data.resolution = 1;
            in_data.range = 4096u;

            if (funcInput.get_node()->get_friendly_name() == "gather0" || funcInput.get_node()->get_friendly_name() == "gather1") {
                // to not go beyond range due to uint32 to half conversion error - it can cause go beyond range
                in_data.range = std::min(static_cast<unsigned int>(inputDataShape[axis_gatherElements]), in_data.range);
            }

            ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
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

INSTANTIATE_TEST_SUITE_P(smoke_GatherElementsPaddingInput,
                         GatherElementsPaddingInputTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(input_precisions)),
                         GatherElementsPaddingInputTest::getTestCaseName);
} // namespace
