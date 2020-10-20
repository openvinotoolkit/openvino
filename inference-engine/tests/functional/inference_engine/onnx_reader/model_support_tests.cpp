// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <fstream>

#include <ie_core.hpp>

namespace {
    std::string model_path(const char* model) {
        std::string path = ONNX_TEST_MODELS;
        path += "support_test/";
        path += model;
        return path;
    }
}

TEST(ONNXReader_ModelSupported, model1) {
    // this model is a basic ONNX model taken from ngraph's unit test (add_abc.onnx)
    // it contains the minimum number of fields required to accept this file as a valid model
    EXPECT_NO_THROW(InferenceEngine::Core{}.ReadNetwork(model_path("supported/basic.onnx")));
}
