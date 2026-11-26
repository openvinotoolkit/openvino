// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/variadic_split.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {
using ov::test::InputShape;

// An identity Gather on a non-first VariadicSplit output must read that output's slice, not the parent buffer start.
class GatherAfterInPlaceSplitTest : public testing::WithParamInterface<std::vector<InputShape>>, virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::vector<InputShape>>& obj) {
        std::ostringstream result;
        result << "IS=";
        for (const auto& shape : obj.param) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
            for (const auto& actual_shape : shape.second)
                result << ov::test::utils::partialShape2str({actual_shape}) << "_";
        }
        return result.str();
    }

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto data = function->input(0);
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = -10;
        in_data.range = 20;
        in_data.resolution = 16;
        inputs.insert({data.get_node_shared_ptr(), ov::test::utils::create_and_fill_tensor(data.get_element_type(), targetInputStaticShapes[0], in_data)});
        const auto indices = function->input(1);
        ov::Tensor identity(ov::element::i64, targetInputStaticShapes[1]);
        for (size_t i = 0; i < identity.get_size(); ++i)
            identity.data<int64_t>()[i] = static_cast<int64_t>(i);
        inputs.insert({indices.get_node_shared_ptr(), identity});
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        init_input_shapes(GetParam());
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0]);
        auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputDynamicShapes[1]);
        auto split = std::make_shared<ov::op::v1::VariadicSplit>(data,
                                                                 ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3}),
                                                                 ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {64, 64, 64, 64}));
        auto gather = std::make_shared<ov::op::v8::Gather>(split->output(2), indices, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
        auto relu = std::make_shared<ov::op::v0::Relu>(split->output(0));
        function = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(gather), std::make_shared<ov::op::v0::Result>(relu)},
                                               ov::ParameterVector{data, indices},
                                               "gather_after_in_place_split");
    }
};

TEST_P(GatherAfterInPlaceSplitTest, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_model,
                         GatherAfterInPlaceSplitTest,
                         ::testing::Values(std::vector<InputShape>{{{-1, 1, 1, 256}, {{5, 1, 1, 256}, {1, 1, 1, 256}, {3, 1, 1, 256}}},
                                                                   {{-1}, {{5}, {1}, {3}}}}),
                         GatherAfterInPlaceSplitTest::getTestCaseName);
}  // namespace
