// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sign.hpp"

#include <limits>

namespace {

using SignNaNTestParams = std::tuple<ov::element::Type, ov::Shape>;

class SignNaNTest : public testing::WithParamInterface<SignNaNTestParams>,
                    virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SignNaNTestParams>& obj) {
        const auto& [prec, shape] = obj.param;
        std::ostringstream result;
        result << "netPRC=" << prec << "_shape=" << ov::test::utils::vec2str(shape);
        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [prec, shape] = GetParam();
        targetDevice = ov::test::utils::DEVICE_GPU;

        auto param = std::make_shared<ov::op::v0::Parameter>(prec, shape);
        param->set_friendly_name("input");
        auto sign = std::make_shared<ov::op::v0::Sign>(param);
        auto result = std::make_shared<ov::op::v0::Result>(sign);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "SignNaN");
    }

    void generate_inputs(const std::vector<ov::Shape>& target_shapes) override {
        inputs.clear();
        const auto& param = function->get_parameters()[0];
        const auto prec = param->get_element_type();
        ov::Tensor tensor(prec, target_shapes[0]);

        const std::vector<float> data{std::numeric_limits<float>::quiet_NaN(),
                                      std::numeric_limits<float>::infinity(),
                                      -std::numeric_limits<float>::infinity(),
                                      +0.0f,
                                      -0.0f,
                                      2.0f,
                                      -1.5f};
        OPENVINO_ASSERT(data.size() <= tensor.get_size(),
                        "Input tensor of shape ",
                        target_shapes[0],
                        " is too small to hold the Sign NaN test vector");

        for (size_t i = 0; i < data.size(); ++i) {
            switch (prec) {
            case ov::element::f32:
                static_cast<float*>(tensor.data())[i] = data[i];
                break;
            case ov::element::f16:
                static_cast<ov::float16*>(tensor.data())[i] = static_cast<ov::float16>(data[i]);
                break;
            default:
                OPENVINO_THROW("Unsupported precision ", prec, " in SignNaNTest::generate_inputs");
            }
        }
        inputs.insert({param, tensor});
    }
};

TEST_P(SignNaNTest, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_SignNaN,
                         SignNaNTest,
                         ::testing::Combine(::testing::Values(ov::element::f32, ov::element::f16),
                                            ::testing::Values(ov::Shape{1, 8})),
                         SignNaNTest::getTestCaseName);

}  // namespace
