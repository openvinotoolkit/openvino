// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

class FQScaleshiftWithConstantShiftTest : public testing::WithParamInterface<ov::element::Type>,
                                          public CPUTestsBase,
                                          virtual public SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::element::Type> obj) {
        ov::element::Type netPrecision;
        netPrecision = obj.param;
        std::ostringstream result;
        result << "netPRC=" << netPrecision.get_type_name() << "_";
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        ov::element::Type netPrecision;
        netPrecision = this->GetParam();

        ov::Shape mmShape{25, 14, 14, 768};
        ov::Shape mmShape2{768, 2304};
        ov::Shape sumShape{1, 1, 1, 2304};

        // avoid eliminations
        std::vector<int> mmInData(768 * 2304);
        std::fill(mmInData.begin(), mmInData.end(), 2);
        mmInData[0] = 1;
        std::vector<int> sumConstData(1 * 1 * 1 * 2304);
        std::iota(sumConstData.begin(), sumConstData.end(), 0);

        auto constShift = ov::op::v0::Constant::create(ov::element::f32, sumShape, sumConstData);
        auto mmConst = ov::op::v0::Constant::create(ov::element::f32, mmShape2, mmInData);
        ov::ParameterVector mmParams{std::make_shared<ov::op::v0::Parameter>(netPrecision, mmShape)};

        const auto mm = std::make_shared<ov::op::v0::MatMul>(mmParams[0], mmConst, false, false);
        auto sum = ov::test::utils::make_eltwise(constShift, mm, ov::test::utils::EltwiseTypes::ADD);
        auto fq = ov::test::utils::make_fake_quantize(sum, ov::element::f32, 256, {}, {-8.0f}, {7.0f}, {-8.0f}, {7.0f});

        ov::ParameterVector inputParams = {mmParams[0]};
        function = makeNgraphFunction(netPrecision, inputParams, fq, "FQScaleshiftWithConstantShift");
    }
};

/* Network with SS subgraph and FQ node. Shift in SS is constant-folded.
 * Test that FQ-SS fusing works correctly while comparing SS and FQ channel dims.
     Input         Const
          \       /
           \     /
            \   /
            MatMul      Const
               \         /
                \       /
                 \     /
                   Add
                    |
                    |
                   FQ
                    |
                    |
                 Output
*/

TEST_P(FQScaleshiftWithConstantShiftTest, CompareWithRefs) {
    run();
}

namespace {
INSTANTIATE_TEST_SUITE_P(smoke_Check, FQScaleshiftWithConstantShiftTest,
                         ::testing::Values(ov::element::f32),
                         FQScaleshiftWithConstantShiftTest::getTestCaseName);
}  // namespace
}  // namespace test
}  // namespace ov
