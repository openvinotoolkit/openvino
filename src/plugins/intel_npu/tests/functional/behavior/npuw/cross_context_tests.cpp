// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset8.hpp>

#include "openvino/core/weight_sharing_util.hpp"
#include "cross_context_tests.hpp"
#include "shared_context_buffer.hpp"
#include "openvino/util/mmap_object.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "shared_context_buffer.hpp"

using namespace ov::npuw::tests;
std::shared_ptr<ov::Model> CrossContextTestsNPUW::create_shared_context_model(size_t elem_count) {
    auto s = std::make_shared<ov::opset8::Parameter>(ov::element::u8, ov::Shape{elem_count});
    auto one = ov::opset8::Constant::create(ov::element::u8, ov::Shape{elem_count}, {1});
    auto add = std::make_shared<ov::opset8::Add>(s, one);  // [1,1] = s+1
    auto res = std::make_shared<ov::opset8::Result>(add);
    auto m = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{s}, "producer");
    m->input(0).set_names({"input"});
    m->output(0).set_names({"output"});
    return m;
}

std::pair<ov::CompiledModel, ov::CompiledModel> 
CrossContextTestsNPUW::create_shared_compiled_models(std::shared_ptr<ov::Model> model,
                                                    const ov::SoPtr<ov::IRemoteContext>& remote_context_gpu,
                                                    const ov::SoPtr<ov::IRemoteContext>& remote_context_npu) {
    ov::CompiledModel gpu_compiled_model = core.compile_model(model, remote_context_gpu->get_device_name());
    ov::CompiledModel npu_compiled_model = core.compile_model(model, remote_context_npu->get_device_name());
    return {gpu_compiled_model, npu_compiled_model};
}

using SharedContextBufferPtr = std::shared_ptr<ov::npuw::SharedContextBuffer>;
std::tuple<SharedContextBufferPtr, SharedContextBufferPtr> execute_inference_on_shared_context(ov::CompiledModel& gpu_compiled_model, ov::CompiledModel& npu_compiled_model,
                                         const ov::SoPtr<ov::IRemoteContext>& remote_context_gpu,
                                         const ov::SoPtr<ov::IRemoteContext>& remote_context_npu,
                                         size_t model_elem_count,
                                        size_t iteration_num) {
    std::vector<ov::SoPtr<ov::IRemoteContext>> ctx = {remote_context_gpu, remote_context_npu};
    const size_t kMinRelocateBytes = ov::util::get_system_page_size();
    size_t shared_mem_bytes_count = model_elem_count * sizeof(char);
    auto shared_context_input_output = std::make_shared<ov::npuw::SharedContextBuffer>(shared_mem_bytes_count, ctx, kMinRelocateBytes);
    auto remote_gpu_tensor_input = shared_context_input_output->get_remote_tensors_if_exist(remote_context_gpu);
    assert(remote_gpu_tensor_input != nullptr && "Failed to get input remote tensor for GPU context");
    auto remote_npu_tensor_output = shared_context_input_output->get_remote_tensors_if_exist(remote_context_npu);
    assert(remote_npu_tensor_output != nullptr && "Failed to get output remote tensor for NPU context");
    auto gpu_tensor_input = ov::make_tensor(remote_gpu_tensor_input);
    auto npu_tensor_output = ov::make_tensor(remote_npu_tensor_output);
    assert(gpu_tensor_input.get_byte_size() == npu_tensor_output.get_byte_size() && "GPU input and NPU output tensor byte sizes do not match");

    auto shared_context_output_input = std::make_shared<ov::npuw::SharedContextBuffer>(shared_mem_bytes_count, ctx, kMinRelocateBytes);
    auto remote_gpu_tensor_output = shared_context_output_input->get_remote_tensors_if_exist(remote_context_gpu);
    assert(remote_gpu_tensor_output != nullptr && "Failed to get output remote tensor for GPU context");
    auto remote_npu_tensor_input = shared_context_output_input->get_remote_tensors_if_exist(remote_context_npu);
    assert(remote_npu_tensor_input != nullptr && "Failed to get input remote tensor for NPU context");
    auto npu_tensor_input = ov::make_tensor(remote_npu_tensor_input);
    auto gpu_tensor_output = ov::make_tensor(remote_gpu_tensor_output);
    assert(gpu_tensor_output.get_byte_size() == npu_tensor_input.get_byte_size() && "GPU output and NPU input tensor byte sizes do not match");

    auto gireq = gpu_compiled_model.create_infer_request();
    auto nireq = npu_compiled_model.create_infer_request();
    gireq.set_input_tensor(gpu_tensor_input);   // the same as NPU output
    gireq.set_output_tensor(gpu_tensor_output);
    nireq.set_input_tensor(npu_tensor_input);
    nireq.set_output_tensor(npu_tensor_output); // the same as GPU input

    // execute inference on GPU and NPU for a number of iterations
    for (size_t i = 0; i < iteration_num; ++i) {
        gireq.infer();
        nireq.infer();
    }

    return {std::move(shared_context_input_output), std::move(shared_context_output_input)};
}

TEST_F(CrossContextTestsNPUW, CanMemoryShareBetweenContexts) {
    ASSERT_TRUE(remote_context_gpu != nullptr);
    ASSERT_TRUE(remote_context_npu != nullptr);
    size_t iteration_num = 13;
    auto [shared_context_input_output, shared_context_output_input] = execute_inference_on_shared_context(gpu_compiled_model, npu_compiled_model, remote_context_gpu, remote_context_npu, model_elem_count, iteration_num);
    // NPU and GPU must have the same data pointer as parts of the shared context buffer
    // the data must be valid and equal to the expected value after the iterations.
    // A value of the shared_context_input_output buffer must be greater than a value of the shared_context_output_input buffer by 1,
    // as NPU adds additional 1 to the output which is the input for GPU
    for (size_t i = 0; i <model_elem_count; ++i) {
        ASSERT_EQ(shared_context_input_output->get_ptr<uint8_t>()[i], shared_context_output_input->get_ptr<uint8_t>()[i] + 1);
        ASSERT_EQ(shared_context_input_output->get_ptr<uint8_t>()[i], iteration_num * 2);
        ASSERT_EQ(shared_context_output_input->get_ptr<uint8_t>()[i], iteration_num * 2 - 1);
    }
}

std::shared_ptr<ov::op::v0::Constant> make_shared_weight_from_weight(
                        ov::weight_sharing::Context &shared_weight_ctx, 
                        std::vector<ov::SoPtr<ov::IRemoteContext>> remote_contexts,
                        std::shared_ptr<ov::op::v0::Constant> not_shared_weight) {
    const size_t kMinRelocateBytes = ov::util::get_system_page_size();
    const size_t bytes = not_shared_weight->get_byte_size();
    auto buffer = std::make_shared<ov::npuw::SharedContextBuffer>(bytes, remote_contexts, kMinRelocateBytes);
    ov::weight_sharing::set_weight_source(shared_weight_ctx, buffer);
    auto shared_weight =
                std::make_shared<ov::op::v0::Constant>(not_shared_weight->get_element_type(),
                      not_shared_weight->get_shape(), buffer);
    ov::weight_sharing::set_constant(shared_weight_ctx, *shared_weight);
    shared_weight->set_friendly_name(not_shared_weight->get_friendly_name());
    ov::copy_runtime_info(not_shared_weight, shared_weight);
    std::memcpy(buffer->get_ptr(), not_shared_weight->get_data_ptr(), bytes);

    return shared_weight;

}

CrossContextTestsWeightSharingNPUW::CrossContextTestsWeightSharingNPUW() = default;
std::pair<ov::CompiledModel, ov::CompiledModel> 
CrossContextTestsWeightSharingNPUW::create_shared_compiled_models(std::shared_ptr<ov::Model> model,
                                                                  const ov::SoPtr<ov::IRemoteContext>& remote_context_gpu, 
                                                                  const ov::SoPtr<ov::IRemoteContext>& remote_context_npu) {
    if (!m_shared_ctx_ptr) {
        m_shared_ctx_ptr = std::make_unique<ov::weight_sharing::Context>();
    }
    // substitute all weights in the model with shared weights
    for (const auto& op : model->get_ops()) {
        auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(op);
        if (!c) {
            continue;
        }
        auto shared_weight = make_shared_weight_from_weight(*m_shared_ctx_ptr, {remote_context_gpu, remote_context_npu}, c);
        ov::replace_node(c, shared_weight);
    }
    ov::CompiledModel gpu_compiled_model = core.compile_model(model, remote_context_gpu->get_device_name());
    ov::CompiledModel npu_compiled_model = core.compile_model(model, remote_context_npu->get_device_name());
    return {gpu_compiled_model, npu_compiled_model};
}

TEST_F(CrossContextTestsWeightSharingNPUW, CanMemoryShareBetweenContextsAsWeights) {
    ASSERT_TRUE(remote_context_gpu != nullptr);
    ASSERT_TRUE(remote_context_npu != nullptr);
    size_t iteration_num = 17;
    auto [shared_context_input_output, shared_context_output_input] = execute_inference_on_shared_context(gpu_compiled_model, npu_compiled_model, remote_context_gpu, remote_context_npu, model_elem_count, iteration_num);
    // NPU and GPU must have the same data pointer as parts of the shared context buffer
    // the data must be valid and equal to the expected value after the iterations.
    // A value of the shared_context_input_output buffer must be greater than a value of the shared_context_output_input buffer by 1,
    // as NPU adds additional 1 to the output which is the input for GPU
    for (size_t i = 0; i <model_elem_count; ++i) {
        ASSERT_EQ(shared_context_input_output->get_ptr<uint8_t>()[i], shared_context_output_input->get_ptr<uint8_t>()[i] + 1);
        ASSERT_EQ(shared_context_input_output->get_ptr<uint8_t>()[i], iteration_num * 2);
        ASSERT_EQ(shared_context_output_input->get_ptr<uint8_t>()[i], iteration_num * 2 - 1);
    }
}
