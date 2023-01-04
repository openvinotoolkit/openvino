// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#ifndef OV_GPU_USE_OPENCL_HPP
# define OV_GPU_USE_OPENCL_HPP
#endif
#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"
#include <remote_blob_tests/remote_blob_helpers.hpp>

TEST_P(MultiDeviceMultipleGPU_Test, canInferOnDefaultContext) {
    auto ie = ov::Core();
    ov::CompiledModel exec_net;
    try {
        exec_net = ie.compile_model(function, device_names, {ov::hint::allow_auto_batching(false),
            ov::hint::performance_mode(ov::hint::PerformanceMode::UNDEFINED)});
    } catch (...) {
        // device is unavailable (e.g. for the "second GPU" test) or other (e.g. env) issues not related to the test
        return;
    }
    std::vector<ov::InferRequest> inf_req_shared = {};
    auto input = function->get_parameters().at(0);
    auto output = function->get_results().at(0);
    auto fakeImageData = FuncTestUtils::create_and_fill_tensor(input->get_element_type(), input->get_shape());
    auto inf_req_regular = exec_net.create_infer_request();
    inf_req_regular.set_tensor(input, fakeImageData);
    // infer using system memory
    inf_req_regular.infer();
    auto output_tensor_regular = inf_req_regular.get_tensor(output);
    auto imSize = ov::shape_size(input->get_shape());
    std::vector<ov::intel_gpu::ocl::ClContext> contexts = {};
    std::vector<ov::intel_gpu::ocl::ClBufferTensor> cldnn_tensor = {};
    for (auto& iter : device_lists) {
        try {
            auto cldnn_context = ie.get_default_context(iter).as<ov::intel_gpu::ocl::ClContext>();
            contexts.push_back(cldnn_context);
            cl_context ctx = cldnn_context;
            auto ocl_instance = std::make_shared<OpenCL>(ctx);
            cl_int err;
            cl::Buffer shared_buffer(ocl_instance->_context, CL_MEM_READ_WRITE, imSize, NULL, &err);
            {
                void* buffer = fakeImageData.data();
                ocl_instance->_queue.enqueueWriteBuffer(shared_buffer, true, 0, imSize, buffer);
            }
            cldnn_tensor.emplace_back(cldnn_context.create_tensor(input->get_element_type(), input->get_shape(), shared_buffer));
        } catch(...) {
            // device does not support remote context
            continue;
        }
    }
    for (int i = 0; i < cldnn_tensor.size(); i++) {
        auto temprequest =  exec_net.create_infer_request();
        temprequest.set_input_tensor(cldnn_tensor.at(i));
        inf_req_shared.emplace_back(temprequest);
    }
    for (int i = 0; i < inf_req_shared.size(); i++)
        inf_req_shared.at(i).start_async();
    for (int i = 0; i < inf_req_shared.size(); i++)
        inf_req_shared.at(i).wait();

    // compare results
    for (int i = 0; i < inf_req_shared.size(); i++) {
        auto output_tensor_shared = inf_req_shared.at(i).get_tensor(output);

        {
            ASSERT_EQ(output->get_element_type(), ov::element::f32);
            ASSERT_EQ(output_tensor_regular.get_size(), output_tensor_shared.get_size());
            auto thr = FuncTestUtils::GetComparisonThreshold(InferenceEngine::Precision::FP32);
            ASSERT_NO_THROW(output_tensor_regular.data());
            ASSERT_NO_THROW(output_tensor_shared.data());
            FuncTestUtils::compare_tensor(output_tensor_regular, output_tensor_shared, thr);
        }
    }
}

TEST_P(MultiDeviceCreateContextMultipleGPU_Test, canInferOnUserContextWithSystemMemory) {
    auto ie = ov::Core();
    ov::CompiledModel exec_net_regular;
    try {
        exec_net_regular = ie.compile_model(function, device_names);
    } catch (...) {
        // device is unavailable (e.g. for the "second GPU" test) or other (e.g. env) issues not related to the test
        return;
    }
    auto input = function->get_parameters().at(0);
    auto output = function->get_results().at(0);

    // regular inference
    auto inf_req_regular = exec_net_regular.create_infer_request();
    auto fakeImageData = FuncTestUtils::create_and_fill_tensor(input->get_element_type(), input->get_shape());
    inf_req_regular.set_tensor(input, fakeImageData);

    inf_req_regular.infer();
    auto output_tensor_regular = inf_req_regular.get_tensor(exec_net_regular.output());

    // inference using remote tensor
    std::vector<std::shared_ptr<OpenCL>> ocl_instances;
    std::vector<cl::Device> devices;
    for (int i = 0; i < device_lists.size(); i++) {
        auto ocl_instance_tmp = std::make_shared<OpenCL>(i);
        ocl_instances.push_back(ocl_instance_tmp);
        devices.push_back(ocl_instances.back()->_device);
    }
    cl::Context multi_device_ctx(devices);
    auto ocl_instance = std::make_shared<OpenCL>(multi_device_ctx.get());
    std::vector<ov::RemoteContext> remote_contexts;
    for (int i = 0; i < device_lists.size(); i++) {
        auto remote_context = ov::intel_gpu::ocl::ClContext(ie, ocl_instance->_context.get(), i);
        remote_contexts.push_back(remote_context);
    }
    ov::AnyMap context_list;
    for (auto& iter : remote_contexts) {
        context_list.insert({iter.get_device_name(), iter});
    }
    auto multi_context = ie.create_context("MULTI", context_list);
    auto exec_net_shared = ie.compile_model(function, multi_context, config);
    auto inf_req_shared = exec_net_shared.create_infer_request();
    inf_req_shared.set_tensor(input, fakeImageData);

    inf_req_shared.infer();
    auto output_tensor_shared = inf_req_shared.get_tensor(output);

    // compare results
    {
        ASSERT_EQ(output->get_element_type(), ov::element::f32);
        ASSERT_EQ(output_tensor_regular.get_size(), output_tensor_shared.get_size());
        auto thr = FuncTestUtils::GetComparisonThreshold(InferenceEngine::Precision::FP32);
        ASSERT_NO_THROW(output_tensor_regular.data());
        ASSERT_NO_THROW(output_tensor_shared.data());
        FuncTestUtils::compare_tensor(output_tensor_regular, output_tensor_shared, thr);
    }
}

TEST_P(MultiDeviceCreateContextMultipleGPU_Test, canInferOnUserContextWithRemoteMemory) {
    auto ie = ov::Core();
    ov::CompiledModel exec_net_regular;
    try {
        exec_net_regular = ie.compile_model(function, device_names);
    } catch (...) {
        // device is unavailable (e.g. for the "second GPU" test) or other (e.g. env) issues not related to the test
        return;
    }
    auto input = function->get_parameters().at(0);
    auto output = function->get_results().at(0);

    // regular inference
    auto inf_req_regular = exec_net_regular.create_infer_request();
    auto fakeImageData = FuncTestUtils::create_and_fill_tensor(input->get_element_type(), input->get_shape());
    inf_req_regular.set_tensor(input, fakeImageData);

    inf_req_regular.infer();
    auto output_tensor_regular = inf_req_regular.get_tensor(exec_net_regular.output());

    // inference using remote tensor
    // get the input/output size
    auto in_size = ov::shape_size(input->get_output_shape(0)) * input->get_output_element_type(0).size();
    auto out_size = ov::shape_size(output->get_output_shape(0)) * output->get_output_element_type(0).size();

    std::vector<std::shared_ptr<OpenCL>> ocl_instances;
    std::vector<ov::RemoteContext> remote_contexts;
    // construcing seprarate GPU contexts
    for (int i = 0; i < device_lists.size(); i++) {
        auto ocl_instance_tmp = std::make_shared<OpenCL>(i);
        ocl_instances.push_back(ocl_instance_tmp);
        auto remote_context = ov::intel_gpu::ocl::ClContext(ie, ocl_instances[i]->_context.get());
        ASSERT_EQ(remote_context.get_device_name(), device_lists[i]);
        remote_contexts.push_back(remote_context);
    }

    // creating multi remote context
    ov::AnyMap context_list;
    for (auto& iter : remote_contexts) {
        context_list.insert({iter.get_device_name(), iter});
    }

    auto multi_context = ie.create_context("MULTI", context_list);
    // load to MULTI with multi remote context

    auto exec_net_context = ie.compile_model(function, multi_context, config);

    auto num_request = device_lists.size();
    std::vector<ov::InferRequest> inf_req_context(num_request);
    std::generate(inf_req_context.begin(), inf_req_context.end(), [&] {
        return exec_net_context.create_infer_request();
    });

    std::vector<cl::Buffer> share_input_buffers(num_request);
    std::vector<cl::Buffer> share_output_buffers(num_request);
    std::vector<ov::Tensor> gpu_in_tensors(num_request);
    std::vector<ov::Tensor> gpu_out_tensors(num_request);
    std::vector<ov::Tensor> out_tensors(num_request);
    std::generate(out_tensors.begin(), out_tensors.end(), [&] {
        return FuncTestUtils::create_and_fill_tensor(output->get_output_element_type(0), output->get_output_shape(0));
    });
    // Fill input data for ireqs
    cl_int err;
    for (int i = 0; i < num_request; i++) {
        // Allocate shared buffers for input and output data which will be set to infer request
        share_input_buffers[i] = cl::Buffer(ocl_instances[i]->_context, CL_MEM_READ_WRITE, in_size, NULL, &err);
        share_output_buffers[i] = cl::Buffer(ocl_instances[i]->_context, CL_MEM_READ_WRITE, out_size, NULL, &err);
        auto temp_context = remote_contexts[i].as<ov::intel_gpu::ocl::ClContext>();
        gpu_in_tensors[i] = temp_context.create_tensor(input->get_output_element_type(0), input->get_output_shape(0), share_input_buffers[i]);
        gpu_out_tensors[i] = temp_context.create_tensor(output->get_output_element_type(0), output->get_output_shape(0), share_output_buffers[i]);
        inf_req_context[i].set_tensor(input, gpu_in_tensors[i]);
        inf_req_context[i].set_tensor(output, gpu_out_tensors[i]);
        void* buffer = fakeImageData.data();
        ocl_instances[i]->_queue.enqueueWriteBuffer(share_input_buffers[i], false, 0, in_size, buffer);
    }
    for (ov::InferRequest& ireq : inf_req_context) {
        ireq.start_async();
    }
    for (ov::InferRequest& ireq : inf_req_context) {
        ireq.wait();
    }
    // read the output into output buffers for result comparing
    for (int i = 0; i < num_request; i++) {
        ocl_instances[i]->_queue.enqueueReadBuffer(share_output_buffers[i], false, 0, out_size, out_tensors[i].data(), nullptr, nullptr);
        // Wait for infer request and post-processing completion
        ocl_instances[i]->_queue.finish();
    }

    // compare results
    for (int i = 0; i < num_request; i++) {
        {
            ASSERT_EQ(output->get_element_type(), ov::element::f32);
            ASSERT_EQ(output_tensor_regular.get_size(), out_tensors[i].get_size());
            auto thr = FuncTestUtils::GetComparisonThreshold(InferenceEngine::Precision::FP32);
            FuncTestUtils::compare_tensor(output_tensor_regular, out_tensors[i], thr);
        }
    }
}