// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/file_utils.hpp"
#include "default_opset.hpp"
#include "engines_util/test_case.hpp"
#include "engines_util/test_engines.hpp"
#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "ngraph/type/element_type.hpp"
#include "onnx_import/onnx.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"
#include "util/type_prop.hpp"

using namespace ngraph;
using namespace ngraph::onnx_import;
using namespace ngraph::test;

OPENVINO_SUPPRESS_DEPRECATED_START

static std::string s_manifest = "${MANIFEST}";
static std::string s_device = test::backend_name_to_device("${BACKEND_NAME}");

NGRAPH_TEST(${BACKEND_NAME}, onnx_external_data) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/external_data/external_data.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_external_data_from_stream) {
    std::string path = file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                            SERIALIZED_ZOO,
                                            "onnx/external_data/external_data.onnx");
    std::ifstream stream{path, std::ios::in | std::ios::binary};
    ASSERT_TRUE(stream.is_open());
    const auto function = onnx_import::import_onnx_model(stream, path);

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();

    stream.close();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_external_data_optional_fields) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/external_data/external_data_optional_fields.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_external_data_in_different_paths) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/external_data/external_data_different_paths.onnx"));

    auto test_case = test::TestCase(function, s_device);
    // first input: {3.f}, second: {1.f, 2.f, 5.f} read from external files
    test_case.add_input<float>({2.f, 7.f, 7.f});

    test_case.add_expected_output<float>({2.f, 4.f, 5.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_external_two_tensors_data_in_the_same_file) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                             SERIALIZED_ZOO,
                             "onnx/external_data/external_data_two_tensors_data_in_the_same_file.onnx"));

    auto test_case = test::TestCase(function, s_device);
    // first input: {3, 2, 1}, second: {1, 2, 3} read from external file
    test_case.add_input<int32_t>({2, 3, 1});

    test_case.add_expected_output<int32_t>({3, 3, 3});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_external_invalid_external_data_exception) {
    try {
        auto function = onnx_import::import_onnx_model(
            file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                 SERIALIZED_ZOO,
                                 "onnx/external_data/external_data_file_not_found.onnx"));
        FAIL() << "Incorrect path to external data not detected";
    } catch (const ngraph_error& error) {
        EXPECT_PRED_FORMAT2(testing::IsSubstring,
                            std::string("not_existed_file.data, offset: 4096, data_length: 16)"),
                            error.what());
    } catch (...) {
        FAIL() << "Importing onnx model failed for unexpected reason";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_external_invalid_up_dir_path) {
    try {
        auto function = onnx_import::import_onnx_model(
            file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                 SERIALIZED_ZOO,
                                 "onnx/external_data/inner_scope/external_data_file_in_up_dir.onnx"));
        FAIL() << "Incorrect path to external data not detected";
    } catch (const ngraph_error& error) {
        EXPECT_PRED_FORMAT2(testing::IsSubstring,
                            std::string("tensor.data, offset: 4096, "
                                        "data_length: 16)"),
                            error.what());
    } catch (...) {
        FAIL() << "Importing onnx model failed for unexpected reason";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_external_invalid_data_length) {
    try {
        auto function = onnx_import::import_onnx_model(
            file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                 SERIALIZED_ZOO,
                                 "onnx/external_data/external_data_invalid_data_length.onnx"));
        FAIL() << "Incorrect path to external data not detected";
    } catch (const ngraph_error& error) {
        EXPECT_PRED_FORMAT2(testing::IsSubstring,
                            std::string("tensor.data, offset: 0, "
                                        "data_length: 30000000000)"),
                            error.what());
    } catch (...) {
        FAIL() << "Importing onnx model failed for unexpected reason";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_external_data_sanitize_path) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/external_data/external_data_sanitize_test.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_external_data_in_constant_node) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/external_data/external_data_in_constant_node.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({3.f, 5.f, 8.f, 13.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {4.f, 7.f, 11.f, 17.f});

    test_case.run();
}
