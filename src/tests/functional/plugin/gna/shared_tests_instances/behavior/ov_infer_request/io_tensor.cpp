// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/io_tensor.hpp"

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> configs = {
    {{"GNA_DEVICE_MODE", "GNA_SW_FP32"},
     {"GNA_DEVICE_MODE", "GNA_SW_EXACT"}}
};

std::vector<ov::element::Type> prcs = {
    ov::element::boolean,
    ov::element::bf16,
    ov::element::f16,
    ov::element::f32,
    ov::element::f64,
    ov::element::i4,
    ov::element::i8,
    ov::element::i16,
    ov::element::i32,
    ov::element::i64,
    ov::element::u1,
    ov::element::u4,
    ov::element::u8,
    ov::element::u16,
    ov::element::u32,
    ov::element::u64,
};

std::vector<ov::element::Type> supported_input_prcs = {
    ov::element::f32,
    ov::element::i16,
    ov::element::u8
};

class OVInferRequestCheckTensorPrecisionGNA : public OVInferRequestCheckTensorPrecision {
public:
    void SetUp() override {
        try {
            OVInferRequestCheckTensorPrecision::SetUp();
            if (std::count(supported_input_prcs.begin(), supported_input_prcs.end(), element_type) == 0) {
                FAIL() << "Precision " << element_type.c_type_string() << " is marked as unsupported but the network was loaded successfully";
            }
        }
        catch (std::runtime_error& e) {
            const std::string errorMsg = e.what();
            const auto expectedMsg = exp_error_str_;
            ASSERT_STR_CONTAINS(errorMsg, expectedMsg);
            EXPECT_TRUE(errorMsg.find(expectedMsg) != std::string::npos)
            << "Wrong error message, actual error message: " << errorMsg
            << ", expected: " << expectedMsg;
            if (std::count(supported_input_prcs.begin(), supported_input_prcs.end(), element_type) == 0) {
                GTEST_SKIP_(expectedMsg.c_str());
            } else {
                FAIL() << "Precision " << element_type.c_type_string() << " is marked as supported but the network was not loaded";
            }
        }
    }

private:
    std::string exp_error_str_ = "The plugin does not support input precision";
};

TEST_P(OVInferRequestCheckTensorPrecisionGNA, CheckInputsOutputs) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestIOTensorTest,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                ::testing::ValuesIn(configs)),
                        OVInferRequestIOTensorTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestIOTensorSetPrecisionTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(prcs),
                                 ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                 ::testing::ValuesIn(configs)),
                         OVInferRequestIOTensorSetPrecisionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestCheckTensorPrecisionGNA,
                         ::testing::Combine(
                                 ::testing::ValuesIn(prcs),
                                 ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                 ::testing::ValuesIn(configs)),
                         OVInferRequestCheckTensorPrecisionGNA::getTestCaseName);

}  // namespace