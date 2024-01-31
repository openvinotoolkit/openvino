// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <sstream>
#include <fstream>

#include "openvino/runtime/core.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/pass/serialize.hpp"

static std::shared_ptr<ov::Model> getModel() {
    auto type = ov::element::f32;
    auto input = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{16, 8});
    auto tensor = ov::test::utils::create_and_fill_tensor(type, ov::Shape{8, 16});
    auto weight = std::make_shared<ov::op::v0::Constant>(tensor);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weight);
    return std::make_shared<ov::Model>(matmul, ov::ParameterVector{input}, "import_model");
}

TEST(CPUFuzzyImportModel, RunWithCompiledModel) {
    auto model = getModel();
    ov::Core core;
    auto compiled_model = core.compile_model(model, "CPU");
    // Import model with compiled model works fine.
    std::stringstream stream;
    compiled_model.export_model(stream);
    ASSERT_NO_THROW(core.import_model(stream, "CPU"));
}

TEST(CPUFuzzyImportModel, ThrowhWithIRModel) {
    auto model = getModel();
    ov::Core core;
    auto compiled_model = core.compile_model(model, "CPU");
    // Import model with IR input throw exception
    ov::serialize(model, "cpu_import_model.xml", "cpu_import_model.bin");
    std::ifstream stream{"cpu_import_model.xml", std::ios::in | std::ios::binary};
    ASSERT_THROW(core.import_model(stream, "CPU"), ov::Exception);
}