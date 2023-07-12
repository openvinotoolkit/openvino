// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>
#include <set>
#include <string>

#include "common_test_utils/file_utils.hpp"
#include "onnx_import/onnx.hpp"
#include "openvino/openvino.hpp"
#include "openvino/util/file_util.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START

TEST(ONNX_Importer_Tests, ImportBasicModel) {
    auto model_file_path =
        ov::test::utils::getModelFromTestModelZoo(ov::util::path_join({ONNX_MODELS_DIR, "add_abc_initializers.onnx"}));
    auto function = ngraph::onnx_import::import_onnx_model(model_file_path);

    int count_additions = 0;
    int count_constants = 0;
    int count_parameters = 0;

    for (auto op : function->get_ops()) {
        const auto op_type = std::string(op->get_type_name());
        count_additions += (op_type == "Add" ? 1 : 0);
        count_constants += (op_type == "Constant" ? 1 : 0);
        count_parameters += (op_type == "Parameter" ? 1 : 0);
    }

    ASSERT_EQ(function->get_output_size(), 1);
    ASSERT_EQ(std::string(function->get_output_op(0)->get_type_name()), "Result");
    ASSERT_EQ(function->get_output_element_type(0), ngraph::element::f32);
    ASSERT_EQ(function->get_output_shape(0), ngraph::Shape({2, 2}));
    ASSERT_EQ(count_additions, 2);
    ASSERT_EQ(count_constants, 2);
    ASSERT_EQ(count_parameters, 1);
}

TEST(ONNX_Importer_Tests, ImportModelWithFusedOp) {
    auto model_file_path =
        ov::test::utils::getModelFromTestModelZoo(ov::util::path_join({ONNX_MODELS_DIR, "selu.onnx"}));
    auto function = ngraph::onnx_import::import_onnx_model(model_file_path);

    int count_selu = 0;
    int count_constants = 0;
    int count_parameters = 0;

    for (auto op : function->get_ops()) {
        const auto op_type = std::string(op->get_type_name());
        count_selu += (op_type == "Selu" ? 1 : 0);
        count_constants += (op_type == "Constant" ? 1 : 0);
        count_parameters += (op_type == "Parameter" ? 1 : 0);
    }

    ASSERT_EQ(function->get_output_size(), 1);
    ASSERT_EQ(std::string(function->get_output_op(0)->get_type_name()), "Result");
    ASSERT_EQ(function->get_output_element_type(0), ngraph::element::f32);
    ASSERT_EQ(function->get_output_shape(0), ngraph::Shape({3, 4, 5}));
    ASSERT_EQ(count_selu, 1);
    ASSERT_EQ(count_constants, 2);
    ASSERT_EQ(count_parameters, 1);
}

TEST(ONNX_Importer_Tests, ImportModelWithMultiOutput) {
    auto model_file_path =
        ov::test::utils::getModelFromTestModelZoo(ov::util::path_join({ONNX_MODELS_DIR, "topk.onnx"}));
    auto function = ngraph::onnx_import::import_onnx_model(model_file_path);

    int count_topk = 0;
    int count_constants = 0;
    int count_parameters = 0;

    for (auto op : function->get_ops()) {
        const auto op_type = std::string(op->get_type_name());
        count_topk += (op_type == "TopK" ? 1 : 0);
        count_constants += (op_type == "Constant" ? 1 : 0);
        count_parameters += (op_type == "Parameter" ? 1 : 0);
    }

    ASSERT_EQ(function->get_output_size(), 2);
    ASSERT_EQ(std::string(function->get_output_op(0)->get_type_name()), "Result");
    ASSERT_EQ(std::string(function->get_output_op(1)->get_type_name()), "Result");
    ASSERT_EQ(function->get_output_element_type(0), ngraph::element::f32);
    ASSERT_EQ(function->get_output_element_type(1), ngraph::element::i64);
    ASSERT_EQ(function->get_output_shape(0), ngraph::Shape({3, 3}));
    ASSERT_EQ(function->get_output_shape(1), ngraph::Shape({3, 3}));
    ASSERT_EQ(count_topk, 1);
    ASSERT_EQ(count_constants, 1);
    ASSERT_EQ(count_parameters, 1);
}

TEST(ONNX_Importer_Tests, ImportModelWithNotSupportedOp) {
    auto model_file_path =
        ov::test::utils::getModelFromTestModelZoo(ov::util::path_join({ONNX_MODELS_DIR, "not_supported.onnx"}));
    try {
        auto function = ngraph::onnx_import::import_onnx_model(model_file_path);
        FAIL() << "Any expection was thrown despite the ONNX model is not supported";
    } catch (const ngraph::ngraph_error& error) {
        EXPECT_PRED_FORMAT2(testing::IsSubstring,
                            std::string("OpenVINO does not support the following ONNX operations: NotSupported"),
                            error.what());
    } catch (...) {
        FAIL() << "Expected 'ngraph::ngraph_error' exception was not thrown despite the ONNX model is not supported";
    }
}

TEST(ONNX_Importer_Tests, ImportModelWhenFileDoesNotExist) {
    auto model_file_path =
        ov::test::utils::getModelFromTestModelZoo(ov::util::path_join({ONNX_MODELS_DIR, "not_exist_file.onnx"}));
    try {
        auto function = ngraph::onnx_import::import_onnx_model(model_file_path);
        FAIL() << "Any expection was thrown despite the ONNX model file does not exist";
    } catch (const ngraph::ngraph_error& error) {
        EXPECT_PRED_FORMAT2(testing::IsSubstring,
                            std::string("Error during import of ONNX model expected to be in file:"),
                            error.what());
    } catch (...) {
        FAIL() << "Expected 'ngraph::ngraph_error' exception was not thrown despite the ONNX model file does not exist";
    }
}

// TODO: CVS-61224
TEST(ONNX_Importer_Tests, DISABLED_ImportModelFromStream) {
    auto model_file_path =
        ov::test::utils::getModelFromTestModelZoo(ov::util::path_join({ONNX_MODELS_DIR, "addmul_abc.onnx"}));
    std::ifstream model_file_stream(model_file_path);
    ASSERT_TRUE(model_file_stream.is_open());
    int count_adds = 0;
    int count_multiplies = 0;
    int count_parameters = 0;

    auto function = ngraph::onnx_import::import_onnx_model(model_file_stream);
    for (auto op : function->get_ops()) {
        const auto op_type = std::string(op->get_type_name());
        count_adds += (op_type == "Add" ? 1 : 0);
        count_multiplies += (op_type == "Multiply" ? 1 : 0);
        count_parameters += (op_type == "Parameter" ? 1 : 0);
    }
    ASSERT_EQ(count_adds, 1);
    ASSERT_EQ(count_multiplies, 1);
    ASSERT_EQ(count_parameters, 3);
}

TEST(ONNX_Importer_Tests, GetSupportedOperators) {
    const std::int64_t version = 1;
    const std::string domain = "ai.onnx";
    const std::set<std::string> supported_ops = ngraph::onnx_import::get_supported_operators(version, domain);

    ASSERT_GT(supported_ops.size(), 1);
    ASSERT_TRUE(supported_ops.find("Add") != supported_ops.end());
}

TEST(ONNX_Importer_Tests, IsOperatorSupported) {
    const std::string op_name = "Abs";
    const std::int64_t version = 12;
    const std::string domain = "ai.onnx";
    const bool is_abs_op_supported = ngraph::onnx_import::is_operator_supported(op_name, version, domain);

    ASSERT_TRUE(is_abs_op_supported);
}

TEST(ONNX_Importer_Tests, ImportModelWithoutMetadata) {
    ov::Core core;
    auto model = core.read_model(
        ov::test::utils::getModelFromTestModelZoo(ov::util::path_join({ONNX_MODELS_DIR, "priorbox_clustered.onnx"})));
    ASSERT_FALSE(model->has_rt_info("framework"));
}

TEST(ONNX_Importer_Tests, ImportModelWithMetadata) {
    ov::Core core;
    auto model = core.read_model(
        ov::test::utils::getModelFromTestModelZoo(ov::util::path_join({ONNX_MODELS_DIR, "model_with_metadata.onnx"})));
    ASSERT_TRUE(model->has_rt_info("framework"));

    const auto rtinfo = model->get_rt_info();
    auto metadata = rtinfo.at("framework").as<ov::AnyMap>();

    ASSERT_EQ(metadata.size(), 2);
    ASSERT_EQ(metadata["meta_key1"].as<std::string>(), "meta_value1");
    ASSERT_EQ(metadata["meta_key2"].as<std::string>(), "meta_value2");
}
