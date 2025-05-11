// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/proxy/properties.hpp"
#include "proxy_tests.hpp"

using namespace ov::proxy::tests;

// IR frontend is needed for import
#ifdef IR_FRONTEND_ENABLED

TEST_F(ProxyTests, import_and_infer_on_device_without_split_on_default_device) {
    std::stringstream model_stream;
    // Model has only add (+ 1) op and reshape
    auto model = create_model_with_reshape();
    {
        auto compiled_model = core.compile_model(model, "MOCK");
        compiled_model.export_model(model_stream);
    }
    auto compiled_model = core.import_model(model_stream, "MOCK", {});
    EXPECT_EQ(1, compiled_model.inputs().size());
    EXPECT_EQ(1, compiled_model.outputs().size());
    auto infer_request = compiled_model.create_infer_request();
    auto input_tensor = create_and_fill_tensor(model->input().get_element_type(), model->input().get_shape());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    auto output_tensor = infer_request.get_output_tensor();
    EXPECT_EQ(input_tensor.get_size(), output_tensor.get_size());
    EXPECT_EQ(input_tensor.get_element_type(), output_tensor.get_element_type());
    EXPECT_NE(memcmp(input_tensor.data(), output_tensor.data(), input_tensor.get_byte_size()), 0);
    // Change input tensor
    {
        auto* data = input_tensor.data<int64_t>();
        for (size_t i = 0; i < input_tensor.get_size(); i++)
            data[i] += 1;
    }
    EXPECT_EQ(memcmp(input_tensor.data(), output_tensor.data(), input_tensor.get_byte_size()), 0);
}

TEST_F(ProxyTests, import_and_infer_on_device_without_split) {
    std::stringstream model_stream;
    auto model = create_model_with_subtract();
    {
        auto compiled_model = core.compile_model(model, "MOCK.3");
        compiled_model.export_model(model_stream);
    }
    auto compiled_model = core.import_model(model_stream, "MOCK.3", {});
    EXPECT_EQ(1, compiled_model.inputs().size());
    EXPECT_EQ(1, compiled_model.outputs().size());
    auto infer_request = compiled_model.create_infer_request();
    auto input_tensor = create_and_fill_tensor(model->input().get_element_type(), model->input().get_shape());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    auto output_tensor = infer_request.get_output_tensor();
    EXPECT_EQ(input_tensor.get_shape(), output_tensor.get_shape());
    EXPECT_EQ(input_tensor.get_element_type(), output_tensor.get_element_type());
    EXPECT_EQ(memcmp(input_tensor.data(), output_tensor.data(), input_tensor.get_byte_size()), 0);
}

#    ifdef HETERO_ENABLED

TEST_F(ProxyTests, import_and_infer_on_support_with_hetero_plugin) {
    std::stringstream model_stream;
    auto model = create_model_with_subtract();
    {
        auto compiled_model = core.compile_model(model, "MOCK.1");
        compiled_model.export_model(model_stream);
    }
    auto compiled_model = core.import_model(model_stream, "MOCK.1", {});
    EXPECT_EQ(1, compiled_model.inputs().size());
    EXPECT_EQ(1, compiled_model.outputs().size());
    auto infer_request = compiled_model.create_infer_request();
    auto input_tensor = create_and_fill_tensor(model->input().get_element_type(), model->input().get_shape());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    auto output_tensor = infer_request.get_output_tensor();
    EXPECT_EQ(input_tensor.get_shape(), output_tensor.get_shape());
    EXPECT_EQ(input_tensor.get_element_type(), output_tensor.get_element_type());
    EXPECT_EQ(memcmp(input_tensor.data(), output_tensor.data(), input_tensor.get_byte_size()), 0);
}
#    endif
#endif
