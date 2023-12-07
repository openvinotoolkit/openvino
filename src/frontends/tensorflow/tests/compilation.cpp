// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <exec_graph_info.hpp>
#include <openvino/frontend/manager.hpp>
#include <openvino/openvino.hpp>

#include "gtest/gtest.h"
#include "tf_utils.hpp"
#include "utils.hpp"

using namespace ov::frontend::tensorflow::tests;

class CompileModelsTests : public ::testing::Test {};

TEST_F(CompileModelsTests, NgramCompilation) {
    ov::Core core;
    auto model = convert_model("model_ngram/model_ngram.pbtxt");
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    const auto runtime_model = compiled_model.get_runtime_model();

    // A convert node will be inserted for CPU plugin API 2.0
    EXPECT_EQ(runtime_model->get_ordered_ops().size(), 5);
    EXPECT_EQ(runtime_model->get_parameters().size(), 2);
    EXPECT_EQ(runtime_model->get_results().size(), 1);
}

#ifdef OPENVINO_ARCH_ARM64
// Ticket: CVS-122396
TEST_F(CompileModelsTests, DISABLED_ModelWithSplitConvConcat)
#else
TEST_F(CompileModelsTests, ModelWithSplitConvConcat)
#endif
{
    {
        auto model = convert_model("split_conv_concat/split_conv_concat.pbtxt");
        ov::Core core;
        ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
        const auto runtime_model = compiled_model.get_runtime_model();
        auto get_layer_type = [](const std::shared_ptr<ov::Node>& node) {
            return node->get_rt_info().at(ExecGraphInfoSerialization::LAYER_TYPE).as<std::string>();
        };
        const auto ops = runtime_model->get_ops();
        EXPECT_EQ(0, std::count_if(ops.begin(), ops.end(), [&](const std::shared_ptr<ov::Node>& node) {
                      return get_layer_type(node) == "Split";
                  }));
        EXPECT_EQ(2, std::count_if(ops.begin(), ops.end(), [&](const std::shared_ptr<ov::Node>& node) {
                      return get_layer_type(node) == "Convolution";
                  }));
        EXPECT_EQ(0, std::count_if(ops.begin(), ops.end(), [&](const std::shared_ptr<ov::Node>& node) {
                      return get_layer_type(node) == "Concat";
                  }));
    }
}

TEST_F(CompileModelsTests, ModelWithShapeOf) {
    auto model = convert_model("shapeof_slice_abs/shapeof_slice_abs.pbtxt");
    ov::Core core;
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    const auto runtime_model = compiled_model.get_runtime_model();
    auto get_layer_type = [](const std::shared_ptr<ov::Node>& node) {
        return node->get_rt_info().at(ExecGraphInfoSerialization::LAYER_TYPE).as<std::string>();
    };
    const auto ops = runtime_model->get_ops();
    // one Input, one Eltwise and one Output
    EXPECT_EQ(3, ops.size());
    // ShapeOf is folded
    EXPECT_EQ(0, std::count_if(ops.begin(), ops.end(), [&](const std::shared_ptr<ov::Node>& node) {
                  return get_layer_type(node) == "ShapeOf";
              }));
    // Slice is eliminated
    EXPECT_EQ(0, std::count_if(ops.begin(), ops.end(), [&](const std::shared_ptr<ov::Node>& node) {
                  return get_layer_type(node) == "StridedSlice";
              }));
}
