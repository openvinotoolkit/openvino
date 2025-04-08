// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <regex>

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "common_test_utils/node_builders/reshape.hpp"
#include "openvino/openvino.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

enum class FQInterval { U8, I8 };
inline std::ostream& operator<<(std::ostream& os, FQInterval interval) {
    switch (interval) {
    case FQInterval::U8:
        os << "U8";
        break;
    case FQInterval::I8:
        os << "I8";
        break;
    default:
        OPENVINO_THROW("Unknown FQInterval");
    }
    return os;
}

typedef std::tuple<InputShape, InputShape, FQInterval, FQInterval> QuantizedMatMulsWithSharedWeightsParans;

/* This test verifies the correctness of the hash function computation for the shared weights.
   Specifically, it checks that when one op requires compensations computation and second one does not,
   the resulting hashes are not identical, and the weights are repacked for each op separately
*/
class QuantizedMatMulsWithSharedWeightsTest
    : public testing::WithParamInterface<QuantizedMatMulsWithSharedWeightsParans>,
      virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<QuantizedMatMulsWithSharedWeightsParans>& obj) {
        InputShape shape1;
        InputShape shape2;
        FQInterval interval1;
        FQInterval interval2;
        std::tie(shape1, shape2, interval1, interval2) = obj.param;
        std::ostringstream result;
        result << "IS1=" << shape1 << "IS2=" << shape2 << "FQInterval1=" << interval1 << "FQInterval2=" << interval2;
        return result.str();
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        abs_threshold = 1e-4;

        InputShape shape1;
        InputShape shape2;
        FQInterval interval1;
        FQInterval interval2;
        std::tie(shape1, shape2, interval1, interval2) = this->GetParam();
        init_input_shapes({shape1, shape2});

        const auto weights = ov::test::utils::make_constant(ov::element::i8, {16, 16});
        const auto convert = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32);
        const auto scale = ov::test::utils::make_constant(ov::element::f32, {16, 1}, ov::test::utils::InputGenerateData(0, 1, 5));
        const auto mul = std::make_shared<ov::op::v1::Multiply>(convert, scale);

        auto build_fq = [](const ov::Output<ov::Node>& parent, FQInterval interval_type) {
            const auto low = interval_type == FQInterval::I8 ? std::vector<float>{-12.8f} : std::vector<float>{0.f};
            const auto high = interval_type == FQInterval::I8 ? std::vector<float>{12.7f} : std::vector<float>{25.5f};
            return ov::test::utils::make_fake_quantize(parent, ov::element::f32, 256, {1, 1, 1, 1}, low, high, low, high);
        };

        const auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0]);
        const auto fq1 = build_fq(param1, interval1);
        const auto mm1 = std::make_shared<ov::op::v0::MatMul>(fq1, mul, false, true);

        const auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[1]);
        const auto fq2 = build_fq(param2, interval2);
        const auto mm2 = std::make_shared<ov::op::v0::MatMul>(fq2, mul, false, true);

        function = std::make_shared<ov::Model>(ov::OutputVector{mm1, mm2}, ov::ParameterVector{param1, param2});
    }
};

TEST_P(QuantizedMatMulsWithSharedWeightsTest, CompareWithRefs) {
    run();
}

namespace {

std::vector<InputShape> shapes1{{{-1, -1, -1, 16}, {{1, 1, 15, 16}, {1, 1, 12, 16}, {1, 1, 15, 16}}}};
std::vector<InputShape> shapes2{{{-1, -1, -1, 16}, {{1, 1, 12, 16}, {1, 1, 15, 16}, {1, 1, 12, 16}}}};
INSTANTIATE_TEST_SUITE_P(smoke_CustomTest, QuantizedMatMulsWithSharedWeightsTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(shapes1),
                                 ::testing::ValuesIn(shapes2),
                                 ::testing::Values(FQInterval::U8, FQInterval::I8),
                                 ::testing::Values(FQInterval::U8, FQInterval::I8)),
                         QuantizedMatMulsWithSharedWeightsTest::getTestCaseName);
}  // namespace
}  // namespace test
}  // namespace ov
