// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_constants.hpp"
#include "hetero_tests.hpp"

namespace ov {
namespace hetero {
namespace tests {

// IR frontend is needed for import
#ifdef IR_FRONTEND_ENABLED
TEST_F(HeteroTests, import_single_plugins) {
    std::stringstream model_stream;
    auto model = create_model_with_reshape();
    {
        auto compiled_model =
            core.compile_model(model, ov::test::utils::DEVICE_HETERO, ov::device::priorities("MOCK0"));
        compiled_model.export_model(model_stream);
    }
    auto compiled_model = core.import_model(model_stream, ov::test::utils::DEVICE_HETERO, {});
    EXPECT_EQ(1, compiled_model.inputs().size());
    EXPECT_EQ(1, compiled_model.outputs().size());
    auto infer_request = compiled_model.create_infer_request();
    auto input_tensor =
        create_and_fill_tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    auto output_tensor = infer_request.get_output_tensor();
    EXPECT_EQ(input_tensor.get_shape(), model->input().get_shape());
    EXPECT_EQ(input_tensor.get_element_type(), model->input().get_element_type());
}

TEST_F(HeteroTests, import_several_plugins) {
    std::stringstream model_stream;
    auto model = create_model_with_subtract();
    {
        auto compiled_model =
            core.compile_model(model, ov::test::utils::DEVICE_HETERO, ov::device::priorities("MOCK0,MOCK1"));
        compiled_model.export_model(model_stream);
    }
    auto compiled_model = core.import_model(model_stream, ov::test::utils::DEVICE_HETERO, {});
    EXPECT_EQ(1, compiled_model.inputs().size());
    EXPECT_EQ(1, compiled_model.outputs().size());
    auto infer_request = compiled_model.create_infer_request();
    auto input_tensor =
        create_and_fill_tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    auto output_tensor = infer_request.get_output_tensor();
    EXPECT_EQ(input_tensor.get_shape(), output_tensor.get_shape());
    EXPECT_EQ(input_tensor.get_element_type(), output_tensor.get_element_type());
    EXPECT_EQ(memcmp(input_tensor.data(), output_tensor.data(), input_tensor.get_byte_size()), 0);
}
#endif
}  // namespace tests
}  // namespace hetero
}  // namespace ov