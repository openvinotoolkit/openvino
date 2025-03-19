// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

namespace ov {
namespace test {
typedef std::tuple<Shape,          // Input shape
                   element::Type,  // Input precision
                   int,            // Axis
                   size_t,         // num_splits
                   size_t,         // scale_factor
                   std::string     // Device name
                   >
    FuseSplitConcatPairToInterpolateTuple;

class FuseSplitConcatPairToInterpolateTest : public testing::WithParamInterface<FuseSplitConcatPairToInterpolateTuple>,
                                             virtual public SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FuseSplitConcatPairToInterpolateTuple>& obj) {
        Shape inputShape;
        element::Type inputPrecision;
        int axis;
        size_t num_splits;
        size_t scale_factor;
        std::string targetName;
        std::tie(inputShape, inputPrecision, axis, num_splits, scale_factor, targetName) = obj.param;
        std::ostringstream results;

        results << "IS=" << inputShape << "_InPRC=" << inputPrecision << "_Axis=" << axis
                << "_Num_splits=" << num_splits << "_Scale_factor=" << scale_factor;
        results << "_targetDevice=" << targetName;

        return results.str();
    }

protected:
    void SetUp() override {
        Shape inputShape;
        element::Type inputPrecision;
        int axis;
        size_t num_splits;
        size_t scale_factor;
        std::tie(inputShape, inputPrecision, axis, num_splits, scale_factor, targetDevice) = this->GetParam();

        size_t num_of_concat_inputs = num_splits * scale_factor;

        const auto param = std::make_shared<ov::op::v0::Parameter>(inputPrecision, inputShape);
        auto split_axis_op =
            std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{}, std::vector<int64_t>{axis});
        auto split = std::make_shared<ov::op::v1::Split>(param, split_axis_op, num_splits);

        ov::OutputVector concat_inputs_vec(num_of_concat_inputs);
        for (size_t split_output_port = 0; split_output_port < num_splits; ++split_output_port) {
            for (size_t j = 0; j < scale_factor; ++j) {
                concat_inputs_vec[split_output_port * scale_factor + j] = split->output(split_output_port);
            }
        }

        const auto concat = std::make_shared<ov::op::v0::Concat>(concat_inputs_vec, axis);

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(concat)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "FuseSplitConcatPairToInterpolate");
    }
};

TEST_P(FuseSplitConcatPairToInterpolateTest, CompareWithRefs) {
    run();
}

namespace {
std::vector<Shape> inputShapes4D{{1, 2, 6, 6}};

std::vector<size_t> num_of_outputs_of_split{2, 3, 6};

std::vector<size_t> scale_factors{2, 3, 4};

std::vector<int> axes4D{2, 3};

std::vector<Shape> inputShapes5D{{1, 3, 10, 6, 6}};

std::vector<int> axes5D{3, 4};

INSTANTIATE_TEST_SUITE_P(smoke_FuseSplitConcatPairToInterpolate4D,
                         FuseSplitConcatPairToInterpolateTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes4D),
                                            ::testing::Values(element::f32),
                                            ::testing::ValuesIn(axes4D),
                                            ::testing::ValuesIn(num_of_outputs_of_split),
                                            ::testing::ValuesIn(scale_factors),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         FuseSplitConcatPairToInterpolateTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FuseSplitConcatPairToInterpolate5D,
                         FuseSplitConcatPairToInterpolateTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes5D),
                                            ::testing::Values(element::f32),
                                            ::testing::ValuesIn(axes5D),
                                            ::testing::ValuesIn(num_of_outputs_of_split),
                                            ::testing::ValuesIn(scale_factors),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         FuseSplitConcatPairToInterpolateTest::getTestCaseName);
}  // namespace
}  // namespace test
}  // namespace ov
