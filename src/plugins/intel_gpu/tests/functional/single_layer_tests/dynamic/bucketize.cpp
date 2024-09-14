// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <random>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/bucketize.hpp"

namespace {
using ov::test::InputShape;

typedef std::tuple<InputShape,        // input shape
                   InputShape,        // boundary shape
                   bool,              // with_right_bound
                   ov::element::Type, // input type
                   ov::element::Type, // boundary type
                   ov::element::Type, // output type
                   std::string
> BucketizeLayerParamsSet;

class BucketizeLayerDynamicTest : public testing::WithParamInterface<BucketizeLayerParamsSet>,
                              virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<BucketizeLayerParamsSet> obj) {
        InputShape shapes;
        InputShape boundary;
        bool with_right_boundary;
        ov::element::Type input_type;
        ov::element::Type boundary_type;
        ov::element::Type output_type;
        std::string deviceName;
        std::tie(shapes, boundary, with_right_boundary, input_type, boundary_type, output_type, deviceName) = obj.param;

        std::ostringstream results;
        results << "IS=" << ov::test::utils::partialShape2str({shapes.first}) << "_";
        results << "TS=";
        for (const auto& item : shapes.second) {
            results << ov::test::utils::vec2str(item) << "_";
        }
        for (const auto& item : boundary.second) {
            results << ov::test::utils::vec2str(item) << "_";
        }
        results << "withRightBoundary=" << with_right_boundary << "_";
        results << "netPRC=" << input_type << "_";
        results << "boundPRC=" << boundary_type << "_";
        results << "outPRC=" << output_type << "_";

        return results.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& func_inputs = function->inputs();
        ov::Tensor tensor;
        for (size_t i = 0; i < func_inputs.size(); ++i) {
            const auto& func_input = func_inputs[i];
            tensor = ov::test::utils::create_and_fill_tensor(func_input.get_element_type(), targetInputStaticShapes[i]);
            inputs.insert({func_input.get_node_shared_ptr(), tensor});
        }
        inferRequestNum++;
    }

protected:
    size_t inferRequestNum = 0;

    void SetUp() override {
        InputShape shapes;
        InputShape boundary;
        bool with_right_boundary;
        ov::element::Type input_type;
        ov::element::Type boundary_type;
        ov::element::Type output_type;
        std::string deviceName;
        std::tie(shapes, boundary, with_right_boundary, input_type, boundary_type, output_type, deviceName) = this->GetParam();

        targetDevice = ov::test::utils::DEVICE_GPU;

        std::vector<InputShape> input_shapes;
        input_shapes.push_back(shapes);
        input_shapes.push_back(boundary);

        init_input_shapes(input_shapes);

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(input_type, inputDynamicShapes.front()),
                                   std::make_shared<ov::op::v0::Parameter>(boundary_type, inputDynamicShapes.back())};

        auto bucketize = std::make_shared<ov::op::v3::Bucketize>(params[0], params[1], output_type, with_right_boundary);

        ov::ResultVector results;
        for (size_t i = 0; i < bucketize->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::op::v0::Result>(bucketize->output(i)));
        }

        function = std::make_shared<ov::Model>(results, params, "BucketizeFuncTest");
    }
};

TEST_P(BucketizeLayerDynamicTest, inference) {
    run();
}

const std::vector<InputShape> input_shapes = {
        {{-1, -1}, {{2, 3}, {3, 4}}}
};

const std::vector<InputShape> boundary_shapes = {
        {{-1}, {{1}, {0}}}
};

const std::vector<bool> with_right_boundary = {
    true,
    false
};

const std::vector<ov::element::Type> input_precisions = {
    ov::element::f16,
    ov::element::f32,
    ov::element::i64,
    ov::element::i32,
    ov::element::i8,
    ov::element::u8
};

const std::vector<ov::element::Type> boundary_precisions = {
    ov::element::f16,
    ov::element::f32,
    ov::element::i64,
    ov::element::i32,
    ov::element::i8,
    ov::element::u8
};

const std::vector<ov::element::Type> output_precisions = {
    ov::element::i64,
    ov::element::i32
};

INSTANTIATE_TEST_SUITE_P(smoke_bucketize_dynamic,
    BucketizeLayerDynamicTest,
    ::testing::Combine(
        ::testing::ValuesIn(input_shapes),
        ::testing::ValuesIn(boundary_shapes),
        ::testing::ValuesIn(with_right_boundary),
        ::testing::ValuesIn(input_precisions),
        ::testing::ValuesIn(boundary_precisions),
        ::testing::ValuesIn(output_precisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    BucketizeLayerDynamicTest::getTestCaseName);

} // namespace