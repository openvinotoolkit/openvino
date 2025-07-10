// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>
#include <set>
#include <string>

#include "common_test_utils/file_utils.hpp"
#include "onnx_utils.hpp"
#include "openvino/openvino.hpp"
#include "openvino/util/file_util.hpp"

using namespace ov;
using namespace ov::frontend::onnx::tests;

TEST(ONNX_Importer_Tests, ImportBasicModel) {
    auto model = convert_model("add_abc_initializers.onnx");

    int count_additions = 0;
    int count_constants = 0;
    int count_parameters = 0;

    for (auto op : model->get_ops()) {
        const auto op_type = std::string(op->get_type_name());
        count_additions += (op_type == "Add" ? 1 : 0);
        count_constants += (op_type == "Constant" ? 1 : 0);
        count_parameters += (op_type == "Parameter" ? 1 : 0);
    }

    ASSERT_EQ(model->get_output_size(), 1);
    ASSERT_EQ(std::string(model->get_output_op(0)->get_type_name()), "Result");
    ASSERT_EQ(model->get_output_element_type(0), ov::element::f32);
    ASSERT_EQ(model->get_output_shape(0), Shape({2, 2}));
    ASSERT_EQ(count_additions, 2);
    ASSERT_EQ(count_constants, 2);
    ASSERT_EQ(count_parameters, 1);
}

TEST(ONNX_Importer_Tests, ImportModelWithFusedOp) {
    auto model = convert_model("selu.onnx");

    int count_selu = 0;
    int count_constants = 0;
    int count_parameters = 0;

    for (auto op : model->get_ops()) {
        const auto op_type = std::string(op->get_type_name());
        count_selu += (op_type == "Selu" ? 1 : 0);
        count_constants += (op_type == "Constant" ? 1 : 0);
        count_parameters += (op_type == "Parameter" ? 1 : 0);
    }

    ASSERT_EQ(model->get_output_size(), 1);
    ASSERT_EQ(std::string(model->get_output_op(0)->get_type_name()), "Result");
    ASSERT_EQ(model->get_output_element_type(0), ov::element::f32);
    ASSERT_EQ(model->get_output_shape(0), Shape({3, 4, 5}));
    ASSERT_EQ(count_selu, 1);
    ASSERT_EQ(count_constants, 2);
    ASSERT_EQ(count_parameters, 1);
}

TEST(ONNX_Importer_Tests, ImportModelWithMultiOutput) {
    auto model = convert_model("topk.onnx");

    int count_topk = 0;
    int count_constants = 0;
    int count_parameters = 0;

    for (auto op : model->get_ops()) {
        const auto op_type = std::string(op->get_type_name());
        count_topk += (op_type == "TopK" ? 1 : 0);
        count_constants += (op_type == "Constant" ? 1 : 0);
        count_parameters += (op_type == "Parameter" ? 1 : 0);
    }

    ASSERT_EQ(model->get_output_size(), 2);
    ASSERT_EQ(std::string(model->get_output_op(0)->get_type_name()), "Result");
    ASSERT_EQ(std::string(model->get_output_op(1)->get_type_name()), "Result");
    ASSERT_EQ(model->get_output_element_type(0), ov::element::f32);
    ASSERT_EQ(model->get_output_element_type(1), ov::element::i64);
    ASSERT_EQ(model->get_output_shape(0), Shape({3, 3}));
    ASSERT_EQ(model->get_output_shape(1), Shape({3, 3}));
    ASSERT_EQ(count_topk, 1);
    ASSERT_EQ(count_constants, 1);
    ASSERT_EQ(count_parameters, 1);
}

TEST(ONNX_Importer_Tests, ImportModelWithNotSupportedOp) {
    try {
        auto model = convert_model("not_supported.onnx");
        FAIL() << "Any expection was thrown despite the ONNX model is not supported";
    } catch (const Exception& error) {
        EXPECT_PRED_FORMAT2(testing::IsSubstring,
                            std::string("OpenVINO does not support the following ONNX operations: NotSupported"),
                            error.what());
    } catch (...) {
        FAIL() << "Expected 'Exception' exception was not thrown despite the ONNX model is not supported";
    }
}

TEST(ONNX_Importer_Tests, ImportModelWhenFileDoesNotExist) {
    try {
        auto model = convert_model("not_exist_file.onnx");
        FAIL() << "Any expection was thrown despite the ONNX model file does not exist";
    } catch (const Exception& error) {
        EXPECT_PRED_FORMAT2(testing::IsSubstring, std::string("Could not open the file"), error.what());
    } catch (...) {
        FAIL() << "Expected 'Exception' exception was not thrown despite the ONNX model file does not exist";
    }
}

TEST(ONNX_Importer_Tests, ImportModelFromStream) {
    auto model_file_path =
        test::utils::getModelFromTestModelZoo(util::path_join({TEST_ONNX_MODELS_DIRNAME, "addmul_abc.onnx"}).string());
    std::ifstream model_file_stream(model_file_path, std::ifstream::binary);
    ASSERT_TRUE(model_file_stream.is_open());
    int count_adds = 0;
    int count_multiplies = 0;
    int count_parameters = 0;

    auto model = convert_model(model_file_stream);
    for (auto op : model->get_ops()) {
        const auto op_type = std::string(op->get_type_name());
        count_adds += (op_type == "Add" ? 1 : 0);
        count_multiplies += (op_type == "Multiply" ? 1 : 0);
        count_parameters += (op_type == "Parameter" ? 1 : 0);
    }
    ASSERT_EQ(count_adds, 1);
    ASSERT_EQ(count_multiplies, 1);
    ASSERT_EQ(count_parameters, 3);
}

TEST(ONNX_Importer_Tests, ImportModelWithoutMetadata) {
    Core core;
    auto model = core.read_model(test::utils::getModelFromTestModelZoo(
        util::path_join({TEST_ONNX_MODELS_DIRNAME, "priorbox_clustered.onnx"}).string()));
    ASSERT_FALSE(model->has_rt_info("framework"));
}

TEST(ONNX_Importer_Tests, ImportModelWithMetadata) {
    Core core;
    auto model = core.read_model(test::utils::getModelFromTestModelZoo(
        util::path_join({TEST_ONNX_MODELS_DIRNAME, "model_with_metadata.onnx"}).string()));
    ASSERT_TRUE(model->has_rt_info("framework"));

    const auto rtinfo = model->get_rt_info();
    auto metadata = rtinfo.at("framework").as<AnyMap>();

    ASSERT_EQ(metadata.size(), 2);
    ASSERT_EQ(metadata["meta_key1"].as<std::string>(), "meta_value1");
    ASSERT_EQ(metadata["meta_key2"].as<std::string>(), "meta_value2");
}
