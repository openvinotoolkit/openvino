// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef OV_GPU_WITH_ZE_RT

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/runtime/remote_tensor.hpp"

#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

namespace {

ov::RemoteTensor create_usm_remote_tensor(ov::RemoteContext& ctx,
                                          ov::intel_gpu::SharedMemType type,
                                          const ov::element::Type& et,
                                          const ov::Shape& shape) {
    return ctx.create_tensor(et, shape, {ov::intel_gpu::shared_mem_type(type)});
}

void validate_remote_tensor_type(const ov::RemoteTensor& tensor,
                                 ov::intel_gpu::SharedMemType expected_type,
                                 const ov::Shape& expected_shape,
                                 const ov::element::Type& expected_et) {
    auto params = tensor.get_params();
    ASSERT_TRUE(params.count(ov::intel_gpu::shared_mem_type.name()) > 0);
    ASSERT_TRUE(params.count(ov::intel_gpu::mem_handle.name()) > 0);

    ASSERT_EQ(params.at(ov::intel_gpu::shared_mem_type.name()).as<ov::intel_gpu::SharedMemType>(), expected_type);
    ASSERT_NE(params.at(ov::intel_gpu::mem_handle.name()).as<ov::intel_gpu::gpu_handle_param>(), nullptr);
    ASSERT_EQ(tensor.get_shape(), expected_shape);
    ASSERT_EQ(tensor.get_element_type(), expected_et);
}

}  // namespace

TEST(zeRemoteContext, smoke_CorrectContextType) {
    auto core = ov::Core();
    auto remote_context = core.get_default_context(ov::test::utils::DEVICE_GPU);
    ASSERT_EQ(remote_context.get_params().at(ov::intel_gpu::context_type.name()), ov::intel_gpu::ContextType::ZE);
}

TEST(zeRemoteTensor, smoke_CreateUsmHostTensor) {
    auto core = ov::Core();
    auto remote_context = core.get_default_context(ov::test::utils::DEVICE_GPU);
    ASSERT_EQ(remote_context.get_params().at(ov::intel_gpu::context_type.name()), ov::intel_gpu::ContextType::ZE);

    const ov::Shape shape{1, 2, 16, 16};
    const ov::element::Type et = ov::element::f32;

    auto remote_tensor = create_usm_remote_tensor(remote_context, ov::intel_gpu::SharedMemType::USM_HOST_BUFFER, et, shape);
    validate_remote_tensor_type(remote_tensor, ov::intel_gpu::SharedMemType::USM_HOST_BUFFER, shape, et);
}

TEST(zeRemoteTensor, smoke_CreateUsmDeviceTensor) {
    auto core = ov::Core();
    auto remote_context = core.get_default_context(ov::test::utils::DEVICE_GPU);
    ASSERT_EQ(remote_context.get_params().at(ov::intel_gpu::context_type.name()), ov::intel_gpu::ContextType::ZE);

    const ov::Shape shape{1, 2, 16, 16};
    const ov::element::Type et = ov::element::f32;

    auto remote_tensor = create_usm_remote_tensor(remote_context, ov::intel_gpu::SharedMemType::USM_DEVICE_BUFFER, et, shape);
    validate_remote_tensor_type(remote_tensor, ov::intel_gpu::SharedMemType::USM_DEVICE_BUFFER, shape, et);
}

TEST(zeRemoteTensor, smoke_CanInferWithUsmHostInputOutput) {
    auto core = ov::Core();
    auto model = ov::test::behavior::getDefaultNGraphFunctionForTheDevice();

    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU);
    auto remote_context = compiled_model.get_context();
    ASSERT_EQ(remote_context.get_params().at(ov::intel_gpu::context_type.name()), ov::intel_gpu::ContextType::ZE);

    auto input = model->inputs().at(0);
    auto output = model->outputs().at(0);
    auto input_shape = input.get_shape();
    auto output_shape = output.get_shape();

    auto host_input = ov::test::utils::create_and_fill_tensor(input.get_element_type(), input_shape);

    auto remote_input = create_usm_remote_tensor(remote_context,
                                                 ov::intel_gpu::SharedMemType::USM_HOST_BUFFER,
                                                 input.get_element_type(),
                                                 input_shape);
    auto remote_output = create_usm_remote_tensor(remote_context,
                                                  ov::intel_gpu::SharedMemType::USM_HOST_BUFFER,
                                                  output.get_element_type(),
                                                  output_shape);

    remote_input.copy_from(host_input);

    auto infer_request = compiled_model.create_infer_request();
    infer_request.set_tensor(input, remote_input);
    infer_request.set_tensor(output, remote_output);
    infer_request.infer();

    ov::Tensor remote_result(output.get_element_type(), output_shape);
    remote_output.copy_to(remote_result);

    auto ref_request = compiled_model.create_infer_request();
    ref_request.set_tensor(input, host_input);
    ref_request.infer();
    auto ref_result = ref_request.get_tensor(output);

    ov::test::utils::compare(remote_result, ref_result);
}

TEST(zeRemoteTensor, smoke_CanInferWithUsmDeviceInputOutput) {
    auto core = ov::Core();
    auto model = ov::test::behavior::getDefaultNGraphFunctionForTheDevice();

    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU);
    auto remote_context = compiled_model.get_context();
    ASSERT_EQ(remote_context.get_params().at(ov::intel_gpu::context_type.name()), ov::intel_gpu::ContextType::ZE);

    auto input = model->inputs().at(0);
    auto output = model->outputs().at(0);
    auto input_shape = input.get_shape();
    auto output_shape = output.get_shape();

    auto host_input = ov::test::utils::create_and_fill_tensor(input.get_element_type(), input_shape);

    auto remote_input = create_usm_remote_tensor(remote_context,
                                                 ov::intel_gpu::SharedMemType::USM_DEVICE_BUFFER,
                                                 input.get_element_type(),
                                                 input_shape);
    auto remote_output = create_usm_remote_tensor(remote_context,
                                                  ov::intel_gpu::SharedMemType::USM_DEVICE_BUFFER,
                                                  output.get_element_type(),
                                                  output_shape);

    remote_input.copy_from(host_input);

    auto infer_request = compiled_model.create_infer_request();
    infer_request.set_tensor(input, remote_input);
    infer_request.set_tensor(output, remote_output);
    infer_request.infer();

    ov::Tensor remote_result(output.get_element_type(), output_shape);
    remote_output.copy_to(remote_result);

    auto ref_request = compiled_model.create_infer_request();
    ref_request.set_tensor(input, host_input);
    ref_request.infer();
    auto ref_result = ref_request.get_tensor(output);

    ov::test::utils::compare(remote_result, ref_result);
}

#endif  // OV_GPU_WITH_ZE_RT
