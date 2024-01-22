// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <sstream>
#include <fstream>

#include "openvino/openvino.hpp"
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset8.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/pass/serialize.hpp"

TEST(CPUImportModel, Fuzzy) {
    auto type = ov::element::f32;
    auto input = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{16, 8});
    auto tensor = ov::test::utils::create_and_fill_tensor(type, ov::Shape{8, 16});
    auto weight = std::make_shared<ov::op::v0::Constant>(tensor);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weight);
    auto model = std::make_shared<ov::Model>(matmul, ov::ParameterVector{input}, "import_model");
    ov::Core core;
    auto compiled_model = core.compile_model(model, "CPU");
    // Import model with compiled model works fine.
    {
        std::stringstream stream;
        compiled_model.export_model(stream);
        ASSERT_NO_THROW(core.import_model(stream, "CPU"));
    }
    // Import model with IR input throw exception
    {
        ov::serialize(model, "cpu_import_model.xml", "cpu_import_model.bin");
        std::istringstream modelStringStream("cpu_import_model.xml");
        ASSERT_THROW(core.import_model(modelStringStream, "CPU"), ov::Exception);
    }
}