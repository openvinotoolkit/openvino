// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "intel_npu/config/config.hpp"
#include "intel_npu/config/npuw.hpp"
#include "plugin.hpp"
#include "openvino/openvino.hpp"
#include "compiled_model.hpp"
#include "just_sync_infer_request.hpp"
#include "model_generator/model_generator.hpp"

TEST(JustSyncIReqTest, SetTensorRemoteInput) {
    // ov::Core ov_core;
    // auto devices = ov_core.get_available_devices();

    // if (std::find(devices.begin(), devices.end(), "NPU") == devices.end()) {
    //     GTEST_SKIP() << "No available devices.";
    // }

    // ModelGenerator mg;
    // auto model = mg.get_model_with_repeated_blocks();

    // ov::AnyMap config = {
    //     {"NPU_USE_NPUW", "YES"},
    //     {"NPUW_FUNCALL_FOR_ALL", "YES"},
    //     {"NPUW_WEIGHTS_BANK", "shared"},
    //     {"NPUW_DEVICES", "NPU"},
    //     {"NPUW_FOLD" , "YES"}
    // };
    
    // ov::element::Type element_type = ov::element::i32;
    // auto input_tensor_shape = model->inputs()[0].get_shape();
    // // Calculate total number of elements
    // size_t total_elements = ov::shape_size(input_tensor_shape);

    // // Create input data
    // std::vector<int> data = std::vector<int>(total_elements, 0);
    // std::iota(data.begin(), data.end(), 1);

    // // Create the remote tensor input
    // auto npu_context = ov_core.get_default_context("NPU");
    // auto input = npu_context.create_host_tensor(element_type, input_tensor_shape);

    // // Initialize remote input with non-zero data
    // ov::Tensor values(element_type, input_tensor_shape, data.data());
    // values.copy_to(input);

    // // Compile NPUW
    // ::intel_npu::Plugin plugin;
    // ov::npuw::CompiledModel compiled(model, plugin, config);
    // // Create infer request
    // ov::npuw::JustInferRequest request(compiled);

    // ASSERT_EQ(request.get_input_allocated_size(), 1);

    // // Set remote io
    // request.set_tensor(input);

    // ASSERT_EQ(request.get_input_allocated_size(), 2);

    // ASSERT_NO_THROW(request.infer());
}
