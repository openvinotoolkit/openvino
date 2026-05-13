// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/transpose.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {

// Params: input shape, order, precision
using TransposeParams = std::tuple<ov::Shape, std::vector<int64_t>, ov::element::Type>;

class TransposeTest : public testing::WithParamInterface<TransposeParams>, virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TransposeParams>& obj) {
        const auto& [shape, order, precision] = obj.param;
        std::ostringstream result;
        result << "Input=" << ov::test::utils::vec2str(shape) << "_";
        result << "Order=" << ov::test::utils::vec2str(order) << "_";
        result << "precision=" << precision;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto& [shape, order, precision] = GetParam();
        auto input = std::make_shared<ov::op::v0::Parameter>(precision, shape);
        auto order_const = ov::op::v0::Constant::create(ov::element::i64, {order.size()}, order);
        auto op = std::make_shared<ov::op::v1::Transpose>(input, order_const);
        auto result = std::make_shared<ov::op::v0::Result>(op);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
    }
};

TEST_P(TransposeTest, Inference) {
    run();
}

const auto test_params = ::testing::Combine(::testing::Values(ov::Shape{1, 1024, 24, 64}, ov::Shape{1, 128, 24, 64}),
                                            ::testing::Values(std::vector<int64_t>{0, 2, 1, 3}),
                                            ::testing::Values(ov::element::f16));

INSTANTIATE_TEST_SUITE_P(smoke_Transpose, TransposeTest, test_params, TransposeTest::getTestCaseName);

}  // namespace
