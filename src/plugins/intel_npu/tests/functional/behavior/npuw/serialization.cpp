// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "npuw/test_engine/models/model_builder.hpp"
#include "openvino/core/parallel.hpp"

using ov::test::npuw::ModelBuilder;

// FIXME: parametrize all the tests below

TEST(SerializationTestNPUW, Stress_ParallelImport) {
    // Only run this test on NPU device
    ov::Core ov_core;
    auto core_devices = ov_core.get_available_devices();
    if (std::find(core_devices.begin(), core_devices.end(), "NPU") == core_devices.end()) {
        GTEST_SKIP() << "No available devices.";
    }

    // Device
    const std::string device = "NPU";

    // Create model
    ModelBuilder mb;
    auto model1 = mb.get_model_with_repeated_blocks();
    auto model2 = mb.get_model_with_repeated_blocks();
    auto model3 = mb.get_model_with_repeated_blocks();
    auto model4 = mb.get_model_with_repeated_blocks();

    // NPUW config
    ov::AnyMap config = {{"NPU_USE_NPUW", "YES"},
                         {"NPUW_FUNCALL_FOR_ALL", "YES"},
                         {"NPUW_DEVICES", "NPU"},
                         {"NPUW_FOLD", "YES"},
                         // FIXME: enable once proper model for weights sharing is available
                         // (go through LLMCompiledModel). Otherwise we hit a case
                         // where bank reads same weights several times, in which
                         // case an assert is triggered.
                         // {"NPUW_WEIGHTS_BANK", "shared"},

                         // FIXME: test weightless mode once proper model with actual weights
                         // is available in tests.
                         {"CACHE_MODE", "OPTIMIZE_SPEED"}};

    // Run stress test to check for data race
    for (size_t i = 0; i < 10; ++i) {
        // Compile NPUW
        auto compiled1 = ov_core.compile_model(model1, device, config);
        auto compiled2 = ov_core.compile_model(model2, device, config);
        auto compiled3 = ov_core.compile_model(model3, device, config);
        auto compiled4 = ov_core.compile_model(model4, device, config);

        // Create infer request and infer
        auto request1 = compiled1.create_infer_request();
        request1.infer();
        auto request2 = compiled2.create_infer_request();
        request2.infer();
        auto request3 = compiled3.create_infer_request();
        request3.infer();
        auto request4 = compiled4.create_infer_request();
        request4.infer();

        std::vector<std::stringstream> ss(4);
        compiled1.export_model(ss[0]);
        compiled2.export_model(ss[1]);
        compiled3.export_model(ss[2]);
        compiled4.export_model(ss[3]);

        std::vector<ov::CompiledModel> imported(4);
        ov::parallel_for(4, [&](size_t idx) {
            imported[idx] = ov_core.import_model(ss[idx], "NPU");
        });

        for (auto& m : imported) {
            auto r = m.create_infer_request();
            r.infer();
        }
    }
}
