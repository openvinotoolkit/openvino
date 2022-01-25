// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

namespace LayerTestsDefinitions {

#define GNA_UNSUPPPORTED_LAYER_NEG_TEST(test_class, expected_error_substring, test_configuration)       \
class Negative##test_class : public test_class {                                                        \
public:                                                                                                 \
        void Run() override {                                                                           \
                try {                                                                                   \
                        test_class::LoadNetwork();                                                      \
                        FAIL() << "GNA's unsupported layers were not detected during LoadNetwork()";    \
                }                                                                                       \
                catch (std::runtime_error& e) {                                                         \
                        const std::string errorMsg = e.what();                                          \
                        const auto expectedMsg = expected_error_substring;                              \
                        ASSERT_STR_CONTAINS(errorMsg, expectedMsg);                                     \
                        EXPECT_TRUE(errorMsg.find(expectedMsg) != std::string::npos)                    \
                        << "Wrong error message, actual error message: " << errorMsg                    \
                        << ", expected: " << expectedMsg;                                               \
                }                                                                                       \
        }                                                                                               \
};                                                                                                      \
                                                                                                        \
TEST_P(Negative##test_class, ThrowAsNotSupported) {                                                     \
    Run();                                                                                              \
}                                                                                                       \
                                                                                                        \
INSTANTIATE_TEST_SUITE_P(smoke_NegativeTestsUnsupportedLayer, Negative##test_class,                     \
                         test_configuration, Negative##test_class::getTestCaseName);

}  // namespace LayerTestsDefinitions
