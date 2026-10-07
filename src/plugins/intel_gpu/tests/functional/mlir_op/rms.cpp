// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/rms.hpp"

#include "mlir_test_env.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace {

using RMSParams = std::tuple<ov::Shape,          // input shape
                             ov::element::Type,  // precision
                             bool>;              // with gamma

class RMSTest : public testing::WithParamInterface<RMSParams>, virtual public ov::test::MlirSubgraphStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RMSParams>& obj) {
        const auto& [shape, precision, with_gamma] = obj.param;
        std::ostringstream result;
        result << "Input=" << ov::test::utils::vec2str(shape) << "_";
        result << "precision=" << precision << "_";
        result << "gamma=" << with_gamma;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto& [shape, precision, with_gamma] = GetParam();
        abs_threshold = 0.01;

        auto input = std::make_shared<ov::op::v0::Parameter>(precision, shape);
        std::shared_ptr<ov::Node> rms;
        if (with_gamma) {
            const size_t last = shape.back();
            std::vector<float> gamma_val(last);
            for (size_t i = 0; i < last; ++i) {
                gamma_val[i] = 0.5F + 0.01F * static_cast<float>(i % 10);
            }
            auto gamma = ov::op::v0::Constant::create(precision, {last}, gamma_val);
            rms = std::make_shared<ov::op::internal::RMS>(input, gamma, 1e-5, precision);
        } else {
            rms = std::make_shared<ov::op::internal::RMS>(input, 1e-5, precision);
        }
        auto result = std::make_shared<ov::op::v0::Result>(rms);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "RMS");
    }
};

TEST_P(RMSTest, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(mlir_RMS,
                         RMSTest,
                         ::testing::Combine(::testing::Values(ov::Shape{1, 128, 64}, ov::Shape{1, 24, 128, 64}),
                                            ::testing::Values(ov::element::f16, ov::element::f32),
                                            ::testing::Bool()),
                         RMSTest::getTestCaseName);

}  // namespace
