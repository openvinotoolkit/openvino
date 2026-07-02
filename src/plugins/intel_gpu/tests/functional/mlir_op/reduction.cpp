// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {
using ReduceParams = std::tuple<ov::Shape,             // Input shape
                                ov::element::Type,     // Input precision
                                std::vector<int64_t>,  // Reduce axes
                                bool>;                 // Keep dims

template <typename ReduceOp>
class ReduceTest : public testing::WithParamInterface<ReduceParams>, virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReduceParams>& obj) {
        const auto& [input_shape, precision, axes, keep_dims] = obj.param;
        std::ostringstream result;
        result << "IS=" << ov::test::utils::vec2str(input_shape) << "_";
        result << "axes=" << ov::test::utils::vec2str(axes) << "_";
        result << "keep_dims=" << keep_dims << "_";
        result << "precision=" << precision;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto& [input_shape, precision, axes, keep_dims] = GetParam();
        auto input = std::make_shared<ov::op::v0::Parameter>(precision, input_shape);
        auto axes_node = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
        auto reduce = std::make_shared<ReduceOp>(input, axes_node, keep_dims);
        auto result = std::make_shared<ov::op::v0::Result>(reduce);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "Reduce");
    }
};

using ReduceMeanTest = ReduceTest<ov::op::v1::ReduceMean>;
using ReduceMaxTest = ReduceTest<ov::op::v1::ReduceMax>;
using ReduceMinTest = ReduceTest<ov::op::v1::ReduceMin>;
using ReduceProdTest = ReduceTest<ov::op::v1::ReduceProd>;
using ReduceSumTest = ReduceTest<ov::op::v1::ReduceSum>;

TEST_P(ReduceMeanTest, Inference) {
    run();
}
TEST_P(ReduceMaxTest, Inference) {
    run();
}
TEST_P(ReduceMinTest, Inference) {
    run();
}
TEST_P(ReduceProdTest, Inference) {
    run();
}
TEST_P(ReduceSumTest, Inference) {
    run();
}

const auto reduce_test_params = ::testing::Combine(::testing::Values(ov::Shape{1, 24, 1024, 64}),
                                                   ::testing::Values(ov::element::f32),
                                                   ::testing::Values(std::vector<int64_t>{3}),
                                                   ::testing::Values(true));
// ::testing::Values(true, false));

#define INSTANTIATE_TS(Name) INSTANTIATE_TEST_SUITE_P(mlir_Reduce##Name##Test, Reduce##Name##Test, reduce_test_params, Reduce##Name##Test::getTestCaseName)
INSTANTIATE_TS(Mean);
INSTANTIATE_TS(Max);
INSTANTIATE_TS(Min);
INSTANTIATE_TS(Prod);
INSTANTIATE_TS(Sum);

}  // namespace
