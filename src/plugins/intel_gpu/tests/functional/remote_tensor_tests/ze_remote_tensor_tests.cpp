// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef OV_GPU_WITH_ZE_RT

#    include "intel_gpu/runtime/internal_properties.hpp"
#    include "openvino/runtime/intel_gpu/remote_properties.hpp"
#    include "openvino/runtime/remote_tensor.hpp"
#    include "shared_test_classes/base/ov_behavior_test_utils.hpp"

// When several GPU libraries are registered under one device name, which runtime serves a device
// is resolved at run time - so assert the context matches the runtime that won, not a fixed one.
TEST(GpuRemoteContext, smoke_ContextTypeMatchesResolvedRuntime) {
    auto core = ov::Core();
    const auto runtime = core.get_property(ov::test::utils::DEVICE_GPU, ov::intel_gpu::runtime_type);
    const auto context_type = core.get_default_context(ov::test::utils::DEVICE_GPU).get_params().at(ov::intel_gpu::context_type.name());

    if (runtime == "OCL") {
        // An OpenCL build always hands out an OpenCL context.
        ASSERT_EQ(context_type, ov::intel_gpu::ContextType::OCL);
    } else if (runtime == "ZE") {
        // A Level Zero build reports ZE, or OCL once the device offers ZE<->OCL interop (LEO):
        // that is what keeps OpenCL-interop applications working on a ZE-served device.
        ASSERT_TRUE(context_type == ov::intel_gpu::ContextType::ZE || context_type == ov::intel_gpu::ContextType::OCL)
            << "a Level Zero device must report a ZE context, or an OCL one when LEO is enabled";
    } else {
        GTEST_SKIP() << "No expected context type mapped for GPU runtime '" << runtime << "'";
    }
}

#endif  // OV_GPU_WITH_ZE_RT
