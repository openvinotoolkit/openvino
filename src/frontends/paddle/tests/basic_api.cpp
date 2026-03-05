// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "basic_api.hpp"

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "openvino/frontend/paddle/frontend.hpp"
#include "openvino/util/file_util.hpp"
#include "paddle_utils.hpp"

using namespace ov::frontend;

using PaddleBasicTest = FrontEndBasicTest;

static const std::vector<std::string> models{
    std::string("conv2d/conv2d.pdmodel"),
    std::string("conv2d_relu/conv2d_relu.pdmodel"),
    std::string("2in_2out/2in_2out.pdmodel"),
    std::string("multi_tensor_split/multi_tensor_split.pdmodel"),
    std::string("2in_2out_dynbatch/2in_2out_dynbatch.pdmodel"),
};

INSTANTIATE_TEST_SUITE_P(PaddleBasicTest,
                         FrontEndBasicTest,
                         ::testing::Combine(::testing::Values(PADDLE_FE),
                                            ::testing::Values(std::string(TEST_PADDLE_MODELS_DIRNAME)),
                                            ::testing::ValuesIn(models)),
                         FrontEndBasicTest::getTestCaseName);

TEST(PaddleBasicTest, check_supported_legacy_model) {
    const auto test_dir = std::filesystem::path(ov::test::utils::generateTestFilePrefix());
    const auto legacy_model_path = test_dir / "paddle_legacy.model";

    {
        ov::util::create_directory_recursive(legacy_model_path);
        std::ofstream model(legacy_model_path / "__model__");
        model << "Fake model data which are not important for this test";
    }

    paddle::FrontEnd fe;

    EXPECT_TRUE(fe.supported({{legacy_model_path.native()}, {false}}));
    EXPECT_FALSE(fe.supported({{legacy_model_path}, {false}})) << "Should be true if std path supported";

    std::filesystem::remove_all(test_dir);
}

TEST(PaddleBasicTest, check_supported_legacy_model_not_exists) {
    const auto test_dir = std::filesystem::path(ov::test::utils::generateTestFilePrefix());
    const auto legacy_model_path = test_dir / "dir.with_dot" / "paddle_legacy_model";

    paddle::FrontEnd fe;

    OV_EXPECT_THROW(fe.supported({{legacy_model_path.native()}, {false}}), ov::Exception, testing::_);
    EXPECT_FALSE(fe.supported({{legacy_model_path}, {false}})) << "Should throw if std path supported";
}

TEST(PaddleBasicTest, check_supported_incorect_extension) {
    const auto test_dir = std::filesystem::path(ov::test::utils::generateTestFilePrefix());
    const auto model_path = test_dir / "model.onnx";

    {
        ov::util::create_directory_recursive(test_dir);
        std::ofstream model(model_path);
        model << "Fake model data which are not important for this test";
    }

    paddle::FrontEnd fe;

    EXPECT_TRUE(fe.supported({{model_path.native()}, {false}}));
    EXPECT_FALSE(fe.supported({{model_path}, {false}}));

    std::filesystem::remove_all(test_dir);
}
