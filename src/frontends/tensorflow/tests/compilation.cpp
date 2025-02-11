// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "openvino/frontend/manager.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "tf_utils.hpp"
#include "utils.hpp"

using namespace ov::frontend::tensorflow::tests;

class CompileModelsTests : public ::testing::Test {};

#if defined OPENVINO_ARCH_ARM64 || defined OPENVINO_ARCH_ARM
// Ticket: 122666, 153163
TEST_F(CompileModelsTests, DISABLED_NgramCompilation) {
    ov::Core core;
    auto model = convert_model("model_ngram/model_ngram.pbtxt");
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    const auto runtime_model = compiled_model.get_runtime_model();

    // A convert node will be inserted for CPU plugin API 2.0
    EXPECT_EQ(runtime_model->get_ordered_ops().size(), 5);
    EXPECT_EQ(runtime_model->get_parameters().size(), 2);
    EXPECT_EQ(runtime_model->get_results().size(), 1);
}
#else
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
#endif

#if defined OPENVINO_ARCH_ARM64 || defined OPENVINO_ARCH_ARM
// Ticket: CVS-122396, 153163
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
            return node->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
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
    core.set_property("CPU", ov::hint::inference_precision(ov::element::f32));
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    const auto runtime_model = compiled_model.get_runtime_model();
    auto get_layer_type = [](const std::shared_ptr<ov::Node>& node) {
        return node->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
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
