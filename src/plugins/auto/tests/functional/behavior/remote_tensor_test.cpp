// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "auto_func_test.hpp"

using namespace ov::auto_plugin::tests;

TEST_F(AutoFuncTests, can_create_remotetensor_then_infer_with_affinity) {
    ov::CompiledModel compiled_model;
    compiled_model = core.compile_model(model_cannot_batch, "MULTI", {ov::device::priorities("MOCK_GPU")});
    auto input = model_cannot_batch->get_parameters().at(0);
    auto output = model_cannot_batch->get_results().at(0);
    auto fake_img_data = ov::Tensor(input->get_element_type(), input->get_shape());
    auto inf_req_regular = compiled_model.create_infer_request();
    inf_req_regular.set_tensor(input, fake_img_data);
    // infer using system memory
    OV_ASSERT_NO_THROW(inf_req_regular.infer());
    auto output_tensor_regular = inf_req_regular.get_tensor(output);
    auto cldnn_context = core.get_default_context("MOCK_GPU");
    auto remote_tensor = cldnn_context.create_tensor(input->get_element_type(), input->get_shape());

    auto infer_req_remote = compiled_model.create_infer_request();
    infer_req_remote.set_tensor(input, remote_tensor);
    // infer using remote tensor
    OV_ASSERT_NO_THROW(infer_req_remote.start_async());
    // no actual inference for remote tensor, due to data not able to mmap
    infer_req_remote.wait();
}

TEST_F(AutoFuncTests, cannot_infer_remote_if_not_initialized_for_device) {
    core.compile_model(model_cannot_batch, "MOCK_CPU");
    core.compile_model(model_cannot_batch, "MOCK_GPU");  // need to initialize the order of plugins in mock_engine
    // simulate 2 hardware devices
    register_plugin_mock_gpu(core, "MOCK_3", {});
    ov::CompiledModel compiled_model;
    auto cldnn_context = core.get_default_context("MOCK_GPU");
    auto input = model_cannot_batch->get_parameters().at(0);
    auto remote_tensor = cldnn_context.create_tensor(input->get_element_type(), input->get_shape());
    OV_ASSERT_NO_THROW(compiled_model =
                           core.compile_model(model_cannot_batch, "MULTI", {ov::device::priorities("MOCK_3")}));
    auto infer_req_remote = compiled_model.create_infer_request();
    infer_req_remote.set_tensor(input, remote_tensor);
    OV_ASSERT_NO_THROW(infer_req_remote.start_async());
    ASSERT_THROW(infer_req_remote.wait(), ov::Exception);
}

TEST_F(AutoFuncTests, can_create_remotetensor_then_infer_with_affinity_2_devices) {
    core.compile_model(model_cannot_batch, "MOCK_CPU");
    core.compile_model(model_cannot_batch, "MOCK_GPU");  // need to initialize the order of plugins in mock_engine
    register_plugin_mock_gpu(core, "MOCK_3", {});
    ov::CompiledModel compiled_model;
    auto input = model_cannot_batch->get_parameters().at(0);
    OV_ASSERT_NO_THROW(
        compiled_model =
            core.compile_model(model_cannot_batch, "MULTI", {ov::device::priorities("MOCK_GPU", "MOCK_3")}));
    std::vector<ov::InferRequest> inf_req_shared = {};
    auto cldnn_context = core.get_default_context("MOCK_GPU");
    auto remote_tensor = cldnn_context.create_tensor(input->get_element_type(), input->get_shape());
    ASSERT_EQ(remote_tensor.get_device_name(), "MOCK_GPU");
    auto cldnn_context_2 = core.get_default_context("MOCK_3");
    auto remote_tensor_2 = cldnn_context_2.create_tensor(input->get_element_type(), input->get_shape());
    ASSERT_EQ(remote_tensor_2.get_device_name(), "MOCK_3");
    auto infer_req_remote = compiled_model.create_infer_request();
    infer_req_remote.set_tensor(input, remote_tensor);
    auto infer_req_remote_2 = compiled_model.create_infer_request();
    infer_req_remote_2.set_tensor(input, remote_tensor_2);
    // infer using remote tensor
    OV_ASSERT_NO_THROW(infer_req_remote.start_async());
    OV_ASSERT_NO_THROW(infer_req_remote_2.start_async());
    OV_ASSERT_NO_THROW(infer_req_remote.wait());
    OV_ASSERT_NO_THROW(infer_req_remote_2.wait());
}

TEST_F(AutoFuncTests, can_create_remotetensor_then_infer_with_affinity_2_devices_device_id) {
    ov::CompiledModel compiled_model;
    auto input = model_cannot_batch->get_parameters().at(0);
    OV_ASSERT_NO_THROW(
        compiled_model =
            core.compile_model(model_cannot_batch, "MULTI", {ov::device::priorities("MOCK_GPU.1", "MOCK_CPU")}));
    auto cldnn_context = core.get_default_context("MOCK_GPU");
    auto remote_tensor = cldnn_context.create_tensor(input->get_element_type(), input->get_shape());
    ASSERT_EQ(remote_tensor.get_device_name(), "MOCK_GPU");
    auto infer_req_remote = compiled_model.create_infer_request();
    infer_req_remote.set_tensor(input, remote_tensor);
    // infer using remote tensor
    OV_ASSERT_NO_THROW(infer_req_remote.start_async());
    ASSERT_THROW_WITH_MESSAGE(infer_req_remote.wait(),
                              ov::Exception,
                              "None of the devices supports a remote tensor created on the device named MOCK_GPU");
}

TEST_F(AutoFuncTests, can_throw_if_oversubsciption_of_inferrequest) {
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(
                           model_cannot_batch,
                           "MULTI",
                           {ov::device::priorities("MOCK_GPU", "MOCK_CPU"), ov::intel_auto::device_bind_buffer(true)}));
    auto optimal_num = compiled_model.get_property(ov::optimal_number_of_infer_requests);
    for (size_t i = 0; i < optimal_num; i++) {
        compiled_model.create_infer_request();
    }
    ASSERT_THROW(compiled_model.create_infer_request(), ov::Exception);
}