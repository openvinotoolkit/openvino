// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "auto_func_test.hpp"

using namespace ov::auto_plugin::tests;

TEST_F(AutoFuncTests, CanExportImportMultiModel) {
    // Disable auto-batching to ensure we test MULTI export directly and not AutoBatch wrapper
    auto compiled_model = core.compile_model(model_can_batch, "MULTI:MOCK_GPU,MOCK_CPU",
                                             {ov::hint::allow_auto_batching(false)});

    std::stringstream stream;
    compiled_model.export_model(stream);

    auto imported_model = core.import_model(stream, "MULTI");

    auto execution_devices = imported_model.get_property(ov::execution_devices);

    ASSERT_EQ(execution_devices.size(), 2);

    bool found_gpu = false;
    bool found_cpu = false;
    for(const auto& dev : execution_devices) {
        if (dev == "MOCK_GPU") found_gpu = true;
        if (dev == "MOCK_CPU") found_cpu = true;
    }

    EXPECT_TRUE(found_gpu);
    EXPECT_TRUE(found_cpu);

    auto req = imported_model.create_infer_request();
    auto input = create_and_fill_tensor(model_can_batch->input().get_element_type(), model_can_batch->input().get_shape());
    req.set_input_tensor(input);
    req.infer();
    auto output = req.get_output_tensor();

    EXPECT_EQ(output.get_size(), 12);

    auto* data = output.data<int64_t>();
    EXPECT_EQ(data[0], 1);
}
