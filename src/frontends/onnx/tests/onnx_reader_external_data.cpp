// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <set>
#include <streambuf>
#include <string>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_case.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "onnx_utils.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/op/constant.hpp"

using namespace std;
using namespace ov;
using namespace frontend;
using namespace frontend::onnx::tests;

// API 2.0 tests
class OnnxFeMmapFixture : public ::testing::TestWithParam<bool> {};

TEST_P(OnnxFeMmapFixture, onnx_external_data) {
    const auto path =
        test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) + "external_data/external_data.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>({2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_from_stream) {
    const auto path =
        test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) + "external_data/external_data.onnx");
    ifstream stream{path, ios::in | ios::binary};
    istream* is = &stream;
    ASSERT_TRUE(stream.is_open());

    auto fem = FrontEndManager();
    auto frontend = fem.load_by_framework("onnx");
    const bool enable_mmap = GetParam();
    const auto in_model = frontend->load(is, path, enable_mmap);
    const auto model = frontend->convert(in_model);

    auto test_case = test::TestCase(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();

    stream.close();
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_incorrect_size_exception) {
    try {
        const auto path = test::utils::getModelFromTestModelZoo(
            string(TEST_ONNX_MODELS_DIRNAME) + "external_data/external_data_incorrect_data_shape.onnx");
        Core core;
        core.set_property(enable_mmap(GetParam()));
        const auto model = core.read_model(path);
        FAIL() << "Incorrect size of external data not detected";
    } catch (const Exception& ex) {
        EXPECT_PRED_FORMAT2(
            testing::IsSubstring,
            string(
                "The size of the external data file does not match the byte size of an initializer 'A' in the model"),
            ex.what());
    } catch (...) {
        FAIL() << "Importing onnx model failed for unexpected reason";
    }
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_optional_fields) {
    const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                            "external_data/external_data_optional_fields.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_offset_not_aligned_with_page_size) {
    const auto path = test::utils::getModelFromTestModelZoo(
        string(TEST_ONNX_MODELS_DIRNAME) + "external_data/external_data_optional_fields_offset_not_aligned.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_offset_not_aligned_with_page_and_less_than_page_size_with_length_provided) {
    const auto path = test::utils::getModelFromTestModelZoo(
        string(TEST_ONNX_MODELS_DIRNAME) +
        "external_data/external_data_offset_not_aligned_with_page_and_less_than_page_size_with_length_provided.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_offset_not_aligned_with_page_and_greater_than_page_size_with_length_provided) {
    const auto path = test::utils::getModelFromTestModelZoo(
        string(TEST_ONNX_MODELS_DIRNAME) +
        "external_data/"
        "external_data_offset_not_aligned_with_page_and_greater_than_page_size_with_length_provided.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_offset_not_aligned_with_page_in_two_pages_scope) {
    const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                            "external_data/"
                                                            "offset_not_aligned_with_page_in_two_pages_scope.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_in_different_paths) {
    const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                            "external_data/external_data_different_paths.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    // first input: {3.f}, second: {1.f, 2.f, 5.f} read from external files
    test_case.add_input<float>({2.f, 7.f, 7.f});

    test_case.add_expected_output<float>({2.f, 4.f, 5.f});
    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_two_tensors_data_in_the_same_file) {
    const auto path = test::utils::getModelFromTestModelZoo(
        string(TEST_ONNX_MODELS_DIRNAME) + "external_data/external_data_two_tensors_data_in_the_same_file.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    // first input: {3, 2, 1}, second: {1, 2, 3} read from external file
    test_case.add_input<int32_t>({2, 3, 1});

    test_case.add_expected_output<int32_t>({3, 3, 3});
    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_invalid_external_data_exception) {
    try {
        const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                                "external_data/external_data_file_not_found.onnx");
        Core core;
        core.set_property(enable_mmap(GetParam()));
        core.read_model(path);
        FAIL() << "Incorrect path to external data not detected";
    } catch (const Exception& ex) {
        EXPECT_PRED_FORMAT2(testing::IsSubstring,
                            string("not_existed_file.data, offset: 4096, data_length: 16)"),
                            ex.what());
    } catch (...) {
        FAIL() << "Importing onnx model failed for unexpected reason";
    }
}

TEST_P(OnnxFeMmapFixture, onnx_external_invalid_up_dir_path) {
    try {
        const auto path = test::utils::getModelFromTestModelZoo(
            string(TEST_ONNX_MODELS_DIRNAME) + "external_data/inner_scope/external_data_file_in_up_dir.onnx");
        Core core;
        core.set_property(enable_mmap(GetParam()));
        core.read_model(path);
        FAIL() << "Incorrect path to external data not detected";
    } catch (const Exception& ex) {
        EXPECT_PRED_FORMAT2(testing::IsSubstring,
                            string("tensor.data, offset: 4096, "
                                   "data_length: 16)"),
                            ex.what());
    } catch (...) {
        FAIL() << "Importing onnx model failed for unexpected reason";
    }
}

TEST_P(OnnxFeMmapFixture, onnx_external_invalid_data_length) {
    try {
        const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                                "external_data/external_data_invalid_data_length.onnx");
        Core core;
        core.set_property(enable_mmap(GetParam()));
        core.read_model(path);
        FAIL() << "Incorrect path to external data not detected";
    } catch (const Exception& ex) {
        EXPECT_PRED_FORMAT2(testing::IsSubstring,
                            string("tensor.data, offset: 0, "
                                   "data_length: 30000000000)"),
                            ex.what());
    } catch (...) {
        FAIL() << "Importing onnx model failed for unexpected reason";
    }
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_sanitize_path) {
    const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                            "external_data/external_data_sanitize_test.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_in_constant_node) {
    const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                            "external_data/external_data_in_constant_node.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_input<float>({3.f, 5.f, 8.f, 13.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {4.f, 7.f, 11.f, 17.f});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_int16) {
    const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                            "external_data/external_data_int16.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_input<int16_t>({-100});
    test_case.add_expected_output<int16_t>(Shape{2, 2}, {-100, 16156, -100, 16284});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_uint16) {
    const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                            "external_data/external_data_uint16.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_input<uint16_t>({100});
    test_case.add_expected_output<uint16_t>(Shape{2, 2}, {100, 16356, 100, 16484});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_int8) {
    const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                            "external_data/external_data_int8.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_input<int8_t>({-100});
    test_case.add_expected_output<int8_t>(Shape{2, 2}, {-100, 106, -100, -37});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_uint8) {
    const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                            "external_data/external_data_uint8.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_input<uint8_t>({100});
    test_case.add_expected_output<uint8_t>(Shape{2, 2}, {100, 100, 228, 163});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_int4) {
    const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                            "external_data/external_data_int4.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_expected_output<int8_t>(Shape{2, 2}, {static_cast<int8_t>(0x80), 0x3f});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_uint4) {
    const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                            "external_data/external_data_uint4.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_expected_output<uint8_t>(Shape{2, 2}, {0x80, 0x3f});

    test_case.run();
}

INSTANTIATE_TEST_SUITE_P(OnnxFeMMapReadModel, OnnxFeMmapFixture, ::testing::Bool());
