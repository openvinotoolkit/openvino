// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <file_utils.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <set>
#include <streambuf>
#include <string>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_case.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "ie_blob.h"
#include "ie_common.h"
#include "ie_core.hpp"
#include "ngraph/ngraph.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/openvino.hpp"

// API 1.0 tests
TEST(ONNX_Reader_Tests, ImportModelWithExternalDataFromFile) {
    InferenceEngine::Core ie;
    auto cnnNetwork = ie.ReadNetwork(
        ov::test::utils::getModelFromTestModelZoo(std::string(ONNX_TEST_MODELS) + "onnx_external_data.onnx"),
        "");
    auto function = cnnNetwork.getFunction();

    int count_additions = 0;
    int count_constants = 0;
    int count_parameters = 0;

    std::shared_ptr<ngraph::Node> external_data_node;
    for (auto op : function->get_ops()) {
        const auto op_type = std::string(op->get_type_name());
        count_additions += (op_type == "Add" ? 1 : 0);
        count_parameters += (op_type == "Parameter" ? 1 : 0);
        if (op_type == "Constant") {
            count_constants += 1;
            external_data_node = op;
        }
    }

    ASSERT_EQ(function->get_output_size(), 1);
    ASSERT_EQ(std::string(function->get_output_op(0)->get_type_name()), "Result");
    ASSERT_EQ(function->get_output_element_type(0), ngraph::element::f32);
    ASSERT_EQ(function->get_output_shape(0), ngraph::Shape({2, 2}));
    ASSERT_EQ(count_additions, 2);
    ASSERT_EQ(count_constants, 1);
    ASSERT_EQ(count_parameters, 2);

    const auto external_data_node_const = ngraph::as_type_ptr<ngraph::op::Constant>(external_data_node);
    ASSERT_TRUE(external_data_node_const->get_vector<float>() == (std::vector<float>{1, 2, 3, 4}));
}

TEST(ONNX_Reader_Tests, ImportModelWithExternalDataFromStringException) {
    InferenceEngine::Core ie;
    const auto path =
        ov::test::utils::getModelFromTestModelZoo(std::string(ONNX_TEST_MODELS) + "onnx_external_data.onnx");
    InferenceEngine::Blob::CPtr weights;  // not used
    std::ifstream stream(path, std::ios::binary);
    std::string modelAsString((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    stream.close();
    try {
        auto cnnNetwork = ie.ReadNetwork(modelAsString, weights);
    } catch (const InferenceEngine::Exception& e) {
        EXPECT_PRED_FORMAT2(testing::IsSubstring, std::string("invalid external data:"), e.what());

        EXPECT_PRED_FORMAT2(testing::IsSubstring,
                            std::string("data/tensor.data, offset: 0, data_length: 0)"),
                            e.what());
    } catch (...) {
        FAIL() << "Reading network failed for unexpected reason";
    }
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
TEST(ONNX_Reader_Tests, ImportModelWithExternalDataFromWstringNamedFile) {
    InferenceEngine::Core ie;
    std::string win_dir_path = ov::test::utils::getModelFromTestModelZoo(ONNX_TEST_MODELS "onnx_external_data.onnx");
    std::wstring wmodel =
        ov::test::utils::addUnicodePostfixToPath(win_dir_path, ov::test::utils::test_unicode_postfix_vector[0]);
    bool is_copy_successfully = ov::test::utils::copyFile(win_dir_path, wmodel);
    if (!is_copy_successfully) {
        FAIL() << "Unable to copy from '" << win_dir_path << "' to '" << ov::util::wstring_to_string(wmodel) << "'";
    }

    auto cnnNetwork = ie.ReadNetwork(wmodel, L"");
    ov::test::utils::removeFile(wmodel);
    auto function = cnnNetwork.getFunction();

    int count_add = 0;
    int count_constants = 0;
    int count_parameters = 0;

    std::shared_ptr<ngraph::Node> external_data_node;
    for (auto op : function->get_ops()) {
        const auto op_type = std::string(op->get_type_name());
        count_add += (op_type == "Add" ? 1 : 0);
        count_parameters += (op_type == "Parameter" ? 1 : 0);
        if (op_type == "Constant") {
            count_constants += 1;
            external_data_node = op;
        }
    }

    ASSERT_EQ(function->get_output_size(), 1);
    ASSERT_EQ(std::string(function->get_output_op(0)->get_type_name()), "Result");
    ASSERT_EQ(function->get_output_element_type(0), ngraph::element::f32);
    ASSERT_EQ(function->get_output_shape(0), ngraph::Shape({2, 2}));
    ASSERT_EQ(count_add, 2);
    ASSERT_EQ(count_constants, 1);
    ASSERT_EQ(count_parameters, 2);

    const auto external_data_node_const = ngraph::as_type_ptr<ngraph::op::Constant>(external_data_node);
    ASSERT_TRUE(external_data_node_const->get_vector<float>() == (std::vector<float>{1, 2, 3, 4}));
}
#endif

// API 2.0 tests

class OnnxFeMmapFixture : public ::testing::TestWithParam<bool> {};

TEST_P(OnnxFeMmapFixture, onnx_external_data) {
    const auto path =
        ov::test::utils::getModelFromTestModelZoo(std::string(ONNX_TEST_MODELS) + "external_data/external_data.onnx");
    ov::Core core;
    core.set_property(ov::enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = ov::test::TestCase(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>({2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_from_stream) {
    const auto path =
        ov::test::utils::getModelFromTestModelZoo(std::string(ONNX_TEST_MODELS) + "external_data/external_data.onnx");
    std::ifstream stream{path, std::ios::in | std::ios::binary};
    std::istream* is = &stream;
    ASSERT_TRUE(stream.is_open());

    auto fem = ov::frontend::FrontEndManager();
    auto frontend = fem.load_by_framework("onnx");
    const bool enable_mmap = GetParam();
    const auto in_model = frontend->load(is, path, enable_mmap);
    const auto model = frontend->convert(in_model);

    auto test_case = ov::test::TestCase(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(ov::Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();

    stream.close();
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_incorrect_size_exception) {
    try {
        const auto path = ov::test::utils::getModelFromTestModelZoo(
            std::string(ONNX_TEST_MODELS) + "external_data/external_data_incorrect_data_shape.onnx");
        ov::Core core;
        core.set_property(ov::enable_mmap(GetParam()));
        const auto model = core.read_model(path);
        FAIL() << "Incorrect size of external data not detected";
    } catch (const ov::Exception& ex) {
        EXPECT_PRED_FORMAT2(
            testing::IsSubstring,
            std::string(
                "The size of the external data file does not match the byte size of an initializer 'A' in the model"),
            ex.what());
    } catch (...) {
        FAIL() << "Importing onnx model failed for unexpected reason";
    }
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_optional_fields) {
    const auto path = ov::test::utils::getModelFromTestModelZoo(std::string(ONNX_TEST_MODELS) +
                                                                "external_data/external_data_optional_fields.onnx");
    ov::Core core;
    core.set_property(ov::enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = ov::test::TestCase(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(ov::Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_offset_not_aligned_with_page_size) {
    const auto path = ov::test::utils::getModelFromTestModelZoo(
        std::string(ONNX_TEST_MODELS) + "external_data/external_data_optional_fields_offset_not_aligned.onnx");
    ov::Core core;
    core.set_property(ov::enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = ov::test::TestCase(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(ov::Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_offset_not_aligned_with_page_and_less_than_page_size_with_length_provided) {
    const auto path = ov::test::utils::getModelFromTestModelZoo(
        std::string(ONNX_TEST_MODELS) +
        "external_data/external_data_offset_not_aligned_with_page_and_less_than_page_size_with_length_provided.onnx");
    ov::Core core;
    core.set_property(ov::enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = ov::test::TestCase(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(ov::Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_offset_not_aligned_with_page_and_greater_than_page_size_with_length_provided) {
    const auto path = ov::test::utils::getModelFromTestModelZoo(
        std::string(ONNX_TEST_MODELS) +
        "external_data/"
        "external_data_offset_not_aligned_with_page_and_greater_than_page_size_with_length_provided.onnx");
    ov::Core core;
    core.set_property(ov::enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = ov::test::TestCase(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(ov::Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_offset_not_aligned_with_page_in_two_pages_scope) {
    const auto path = ov::test::utils::getModelFromTestModelZoo(std::string(ONNX_TEST_MODELS) +
                                                                "external_data/"
                                                                "offset_not_aligned_with_page_in_two_pages_scope.onnx");
    ov::Core core;
    core.set_property(ov::enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = ov::test::TestCase(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(ov::Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_in_different_paths) {
    const auto path = ov::test::utils::getModelFromTestModelZoo(std::string(ONNX_TEST_MODELS) +
                                                                "external_data/external_data_different_paths.onnx");
    ov::Core core;
    core.set_property(ov::enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = ov::test::TestCase(model);
    // first input: {3.f}, second: {1.f, 2.f, 5.f} read from external files
    test_case.add_input<float>({2.f, 7.f, 7.f});

    test_case.add_expected_output<float>({2.f, 4.f, 5.f});
    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_two_tensors_data_in_the_same_file) {
    const auto path = ov::test::utils::getModelFromTestModelZoo(
        std::string(ONNX_TEST_MODELS) + "external_data/external_data_two_tensors_data_in_the_same_file.onnx");
    ov::Core core;
    core.set_property(ov::enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = ov::test::TestCase(model);
    // first input: {3, 2, 1}, second: {1, 2, 3} read from external file
    test_case.add_input<int32_t>({2, 3, 1});

    test_case.add_expected_output<int32_t>({3, 3, 3});
    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_invalid_external_data_exception) {
    try {
        const auto path = ov::test::utils::getModelFromTestModelZoo(std::string(ONNX_TEST_MODELS) +
                                                                    "external_data/external_data_file_not_found.onnx");
        ov::Core core;
        core.set_property(ov::enable_mmap(GetParam()));
        core.read_model(path);
        FAIL() << "Incorrect path to external data not detected";
    } catch (const ov::Exception& ex) {
        EXPECT_PRED_FORMAT2(testing::IsSubstring,
                            std::string("not_existed_file.data, offset: 4096, data_length: 16)"),
                            ex.what());
    } catch (...) {
        FAIL() << "Importing onnx model failed for unexpected reason";
    }
}

TEST_P(OnnxFeMmapFixture, onnx_external_invalid_up_dir_path) {
    try {
        const auto path = ov::test::utils::getModelFromTestModelZoo(
            std::string(ONNX_TEST_MODELS) + "external_data/inner_scope/external_data_file_in_up_dir.onnx");
        ov::Core core;
        core.set_property(ov::enable_mmap(GetParam()));
        core.read_model(path);
        FAIL() << "Incorrect path to external data not detected";
    } catch (const ov::Exception& ex) {
        EXPECT_PRED_FORMAT2(testing::IsSubstring,
                            std::string("tensor.data, offset: 4096, "
                                        "data_length: 16)"),
                            ex.what());
    } catch (...) {
        FAIL() << "Importing onnx model failed for unexpected reason";
    }
}

TEST_P(OnnxFeMmapFixture, onnx_external_invalid_data_length) {
    try {
        const auto path = ov::test::utils::getModelFromTestModelZoo(
            std::string(ONNX_TEST_MODELS) + "external_data/external_data_invalid_data_length.onnx");
        ov::Core core;
        core.set_property(ov::enable_mmap(GetParam()));
        core.read_model(path);
        FAIL() << "Incorrect path to external data not detected";
    } catch (const ov::Exception& ex) {
        EXPECT_PRED_FORMAT2(testing::IsSubstring,
                            std::string("tensor.data, offset: 0, "
                                        "data_length: 30000000000)"),
                            ex.what());
    } catch (...) {
        FAIL() << "Importing onnx model failed for unexpected reason";
    }
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_sanitize_path) {
    const auto path = ov::test::utils::getModelFromTestModelZoo(std::string(ONNX_TEST_MODELS) +
                                                                "external_data/external_data_sanitize_test.onnx");
    ov::Core core;
    core.set_property(ov::enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = ov::test::TestCase(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(ov::Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_in_constant_node) {
    const auto path = ov::test::utils::getModelFromTestModelZoo(std::string(ONNX_TEST_MODELS) +
                                                                "external_data/external_data_in_constant_node.onnx");
    ov::Core core;
    core.set_property(ov::enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = ov::test::TestCase(model);
    test_case.add_input<float>({3.f, 5.f, 8.f, 13.f});
    test_case.add_expected_output<float>(ov::Shape{2, 2}, {4.f, 7.f, 11.f, 17.f});

    test_case.run();
}

INSTANTIATE_TEST_SUITE_P(OnnxFeMMapReadModel, OnnxFeMmapFixture, ::testing::Bool());
