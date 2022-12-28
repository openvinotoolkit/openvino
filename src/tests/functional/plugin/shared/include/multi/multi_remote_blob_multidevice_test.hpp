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
#include "openvino/core/preprocess/pre_post_process.hpp"
#include <remote_blob_tests/remote_blob_helpers.hpp>

TEST_P(MultiDeviceMultipleGPU_Test, canCreateRemoteTensorThenInferWithAffinity) {
    auto ie = ov::Core();
    using namespace ov::preprocess;
    auto p = PrePostProcessor(fn_ptr);
    p.input().tensor().set_element_type(ov::element::i8);
    p.input().preprocess().convert_element_type(ov::element::f32);

    auto function = p.build();
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