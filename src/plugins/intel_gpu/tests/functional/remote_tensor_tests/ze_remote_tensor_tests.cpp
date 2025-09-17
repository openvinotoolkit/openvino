// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef OV_GPU_WITH_ZE_RT

#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/runtime/remote_tensor.hpp"

#include "remote_tensor_tests/helpers.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

TEST(ZeRemoteContext, smoke_CorrectContextType) {
    auto core = ov::Core();
    auto remote_context = core.get_default_context(ov::test::utils::DEVICE_GPU);
    ASSERT_FALSE(remote_context.is<ov::intel_gpu::ocl::ClContext>());
    ASSERT_EQ(remote_context.get_params().at(ov::intel_gpu::context_type.name()), ov::intel_gpu::ContextType::ZE);
}

#endif  // OV_GPU_WITH_ZE_RT
