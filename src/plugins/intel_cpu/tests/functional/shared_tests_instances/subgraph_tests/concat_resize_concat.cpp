// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"

#include <tuple>
#include <string>
#include <vector>

namespace ov {
namespace test {

typedef std::tuple<int,  // channels count
                   int   // batch count
                   >
    ConcResizeConcParams;

class ConcatResizeConcatTest : public testing::WithParamInterface<ConcResizeConcParams>,
                               public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConcResizeConcParams>& obj) {
        int channels_count;
        int batch_count;
        std::tie(channels_count, batch_count) = obj.param;
        std::ostringstream result;
        result << "Batches=" << batch_count << "_";
        result << "Channels=" << channels_count << "_";
        result << obj.index;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        int channels_count;
        int batch_count;
        std::tie(channels_count, batch_count) = this->GetParam();

        std::vector<int> dims1({batch_count, channels_count, 2, 2});
        std::vector<int> dims2({batch_count, channels_count, 3, 3});

        std::vector<size_t> shape1({size_t(dims1[0]), size_t(dims1[1]), size_t(dims1[2]), size_t(dims1[3])});
        std::vector<size_t> shape2({size_t(dims2[0]), size_t(dims2[1]), size_t(dims2[2]), size_t(dims2[3])});
        auto inputNode1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(shape1));
        auto inputNode2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(shape1));
        auto inputNode3 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(shape2));
        // concat layer
        ov::OutputVector concatNodes1;
        concatNodes1.push_back(inputNode1);
        concatNodes1.push_back(inputNode2);
        std::shared_ptr<ov::Node> inputNode = std::make_shared<ov::op::v0::Concat>(concatNodes1, 1);

        // preresize layer
        ov::op::util::InterpolateBase::InterpolateAttrs attrs;
        attrs.mode = ov::op::util::InterpolateBase::InterpolateMode::LINEAR_ONNX;
        attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SIZES;
        attrs.coordinate_transformation_mode = ov::op::util::InterpolateBase::CoordinateTransformMode::ASYMMETRIC;
        attrs.nearest_mode = ov::op::util::InterpolateBase::NearestMode::CEIL;
        std::vector<int64_t> shape = {3, 3};

        std::vector<float> scales = {1.5, 1.5};
        auto outputShape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, shape.data());
        auto scalesShape = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{2}, scales.data());
        auto axes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{2, 3});
        std::shared_ptr<ov::Node> preresizeNode =
            std::make_shared<ov::op::v4::Interpolate>(inputNode, outputShape, scalesShape, axes, attrs);

        // concat layer
        ov::OutputVector concatNodes2;
        concatNodes2.push_back(preresizeNode);
        concatNodes2.push_back(inputNode3);
        std::shared_ptr<ov::Node> outputNode = std::make_shared<ov::op::v0::Concat>(concatNodes2, 1);

        // Run shape inference on the nodes
        ov::NodeVector nodes;
        nodes.push_back(inputNode1);
        nodes.push_back(inputNode2);
        nodes.push_back(inputNode3);
        nodes.push_back(inputNode);
        nodes.push_back(preresizeNode);
        nodes.push_back(outputNode);

        // Create graph
        ov::ParameterVector inputs;
        inputs.push_back(inputNode1);
        inputs.push_back(inputNode2);
        inputs.push_back(inputNode3);
        ov::ResultVector outputs;
        outputs.push_back(std::make_shared<ov::op::v0::Result>(outputNode));
        function = std::make_shared<ov::Model>(outputs, inputs);
    }
};

TEST_P(ConcatResizeConcatTest, CompareWithRefs) {
    run();
}

namespace {

const std::vector<int> batch_count = {1, 2};

const std::vector<int> channel_count = {1, 2};

INSTANTIATE_TEST_SUITE_P(smoke_ConcResizeConc,
                         ConcatResizeConcatTest,
                         ::testing::Combine(::testing::ValuesIn(channel_count), ::testing::ValuesIn(batch_count)),
                         ConcatResizeConcatTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
