// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/tanh.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {

using UnaryElementwiseParams = std::tuple<ov::Shape, ov::element::Type>;

template <typename Op>
class UnaryElementwiseTest : public testing::WithParamInterface<UnaryElementwiseParams>, virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<UnaryElementwiseParams>& obj) {
        const auto& [shape, precision] = obj.param;
        std::ostringstream result;
        result << "Input=" << ov::test::utils::vec2str(shape) << "_";
        result << "precision=" << precision;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto& [shape, precision] = GetParam();
        auto input = std::make_shared<ov::op::v0::Parameter>(precision, shape);
        auto op = std::make_shared<Op>(input);
        auto result = std::make_shared<ov::op::v0::Result>(op);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
    }
};

using AbsTest = UnaryElementwiseTest<ov::op::v0::Abs>;
using CeilingTest = UnaryElementwiseTest<ov::op::v0::Ceiling>;
using ExpTest = UnaryElementwiseTest<ov::op::v0::Exp>;
using FloorTest = UnaryElementwiseTest<ov::op::v0::Floor>;
using LogTest = UnaryElementwiseTest<ov::op::v0::Log>;
using NegativeTest = UnaryElementwiseTest<ov::op::v0::Negative>;
using ReluTest = UnaryElementwiseTest<ov::op::v0::Relu>;
using SqrtTest = UnaryElementwiseTest<ov::op::v0::Sqrt>;
using TanhTest = UnaryElementwiseTest<ov::op::v0::Tanh>;

#define DEFINE_TEST(Name)     \
    TEST_P(Name, Inference) { \
        run();                \
    }

DEFINE_TEST(AbsTest)
DEFINE_TEST(CeilingTest)
DEFINE_TEST(ExpTest)
DEFINE_TEST(FloorTest)
DEFINE_TEST(LogTest)
DEFINE_TEST(NegativeTest)
DEFINE_TEST(ReluTest)
DEFINE_TEST(SqrtTest)
DEFINE_TEST(TanhTest)

const auto test_params = ::testing::Combine(::testing::Values(ov::Shape{1, 24, 1024, 1}, ov::Shape{1, 24, 128, 1}), ::testing::Values(ov::element::f16));

#define INSTANTIATE_TS(Name) INSTANTIATE_TEST_SUITE_P(smoke_UnaryElementwise##Name, Name, test_params, Name::getTestCaseName)

INSTANTIATE_TS(AbsTest);
INSTANTIATE_TS(CeilingTest);
INSTANTIATE_TS(ExpTest);
INSTANTIATE_TS(FloorTest);
INSTANTIATE_TS(LogTest);
INSTANTIATE_TS(NegativeTest);
INSTANTIATE_TS(ReluTest);
INSTANTIATE_TS(SqrtTest);
INSTANTIATE_TS(TanhTest);

}  // namespace
