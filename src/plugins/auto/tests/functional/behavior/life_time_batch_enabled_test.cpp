// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "auto_func_test.hpp"
#include "common_test_utils/include/common_test_utils/file_utils.hpp"
#include "openvino/runtime/make_tensor.hpp"

using namespace ov::auto_plugin::tests;

TEST_F(AutoFuncTests, tensor_life_time_with_batch_model) {
    auto gpu_compiled_model = core.compile_model(model_can_batch, "MOCK_GPU");
    auto gpu_request = gpu_compiled_model.create_infer_request();
    auto input = gpu_compiled_model.input();
    auto gpu_tensor = gpu_request.get_tensor(input);
    auto gpu_tensor_detail = ov::get_tensor_impl(gpu_tensor);

    auto compiled_model = core.compile_model(
        model_can_batch,
        "AUTO",
        {ov::device::priorities("MOCK_GPU"), ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)});
    auto request = compiled_model.create_infer_request();
    auto tensor = request.get_tensor(input);
    auto tensor_detail = ov::get_tensor_impl(gpu_tensor);
    ASSERT_EQ(tensor_detail._so, gpu_tensor_detail._so);
}

TEST_F(AutoFuncTests, tensor_life_time_with_batch_model_latency_hint) {
    auto gpu_compiled_model = core.compile_model(model_can_batch, "MOCK_GPU");
    auto gpu_request = gpu_compiled_model.create_infer_request();
    auto input = gpu_compiled_model.input();
    auto gpu_tensor = gpu_request.get_tensor(input);
    auto gpu_tensor_detail = ov::get_tensor_impl(gpu_tensor);

    auto compiled_model = core.compile_model(model_can_batch, "AUTO", {ov::device::priorities("MOCK_GPU")});
    auto request = compiled_model.create_infer_request();
    auto tensor = request.get_tensor(input);
    auto tensor_detail = ov::get_tensor_impl(gpu_tensor);
    ASSERT_EQ(tensor_detail._so, gpu_tensor_detail._so);
}

TEST_F(AutoFuncTests, tensor_life_time_with_batch_not_applicable_model) {
    auto gpu_compiled_model = core.compile_model(model_cannot_batch, "MOCK_GPU");
    auto gpu_request = gpu_compiled_model.create_infer_request();
    auto input = gpu_compiled_model.input();
    auto gpu_tensor = gpu_request.get_tensor(input);
    auto gpu_tensor_detail = ov::get_tensor_impl(gpu_tensor);

    auto compiled_model = core.compile_model(
        model_cannot_batch,
        "AUTO",
        {ov::device::priorities("MOCK_GPU"), ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)});
    auto request = compiled_model.create_infer_request();
    auto tensor = request.get_tensor(input);
    auto tensor_detail = ov::get_tensor_impl(gpu_tensor);
    ASSERT_EQ(tensor_detail._so, gpu_tensor_detail._so);
}