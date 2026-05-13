// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <map>

#include "npuw/test_engine/models/model_builder.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"

using ov::test::npuw::LLMConfig;
using ov::test::npuw::ModelBuilder;

namespace {

LLMConfig make_llm_config() {
    LLMConfig cfg;
    cfg.num_layers = 2;
    cfg.hidden_size = 64;
    cfg.num_heads = 4;
    cfg.head_dim = 16;
    cfg.num_kv_heads = 4;
    cfg.vocab_size = 256;
    return cfg;
}

std::shared_ptr<ov::Model> build_chunked_prefill_model() {
    auto cfg = make_llm_config();
    cfg.num_kv_heads = 2;
    cfg.force_gqa_broadcast = true;

    ModelBuilder mb;
    auto model = mb.build_llm(cfg);
    ov::pass::StatefulToStateless().run_on_model(model);
    model = model->clone();

    constexpr std::size_t kSeq = 8;
    constexpr std::size_t kPast = 8;

    std::map<std::string, ov::PartialShape> new_shapes;
    for (const auto& input : model->inputs()) {
        const auto& name = input.get_any_name();
        auto pshape = input.get_partial_shape();
        const auto rank = pshape.rank();

        if (name.find("input_ids") != std::string::npos || name.find("token_type_ids") != std::string::npos) {
            new_shapes[name] = ov::PartialShape{1, kSeq};
        } else if (name.find("inputs_embeds") != std::string::npos && rank.is_static() && rank.get_length() == 3) {
            new_shapes[name] = ov::PartialShape{1, kSeq, pshape[2]};
        } else if (name.find("attention_mask") != std::string::npos) {
            new_shapes[name] = ov::PartialShape{1, kSeq + kPast};
        } else if (name.find("position_ids") != std::string::npos) {
            new_shapes[name] =
                rank.get_length() == 3 ? ov::PartialShape{3, 1, kSeq} : ov::PartialShape{1, kSeq};
        } else if (name.find("beam_idx") != std::string::npos) {
            new_shapes[name] = ov::PartialShape{1};
        } else if (rank.is_static() && rank.get_length() > 2) {
            pshape[0] = 1;
            pshape[2] = kPast;
            new_shapes[name] = pshape;
        } else {
            new_shapes[name] = pshape;
        }
    }

    model->reshape(new_shapes);
    model->validate_nodes_and_infer_types();
    return model;
}

ov::AnyMap make_phase0_decode_npu_opts() {
    return {
        {"NPU_USE_NPUW", "YES"},
        {"NPUW_DEVICES", "NPU"},
        {"NPUW_WEIGHTS_BANK", "shared"},
        {"NPUW_FUNCALL_FOR_ALL", "YES"},
        {"NPUW_CWAI", "YES"},
        {"NPUW_ONLINE_PIPELINE", "NONE"},
        {"NPU_COMPILER_DYNAMIC_QUANTIZATION", "YES"},
    };
}

ov::AnyMap make_phase0_base_config() {
    auto config = make_phase0_decode_npu_opts();
    config["NPUW_ENSURE_COMPATIBILITY"] = "YES";
    return config;
}

ov::AnyMap make_phase0_cpu_subgraph_config() {
    auto config = make_phase0_base_config();
    config["NPUW_DEVICES"] = "NPU,CPU";
    config["NPUW_SUBMODEL_DEVICE"] = "0:CPU";
    return config;
}

void skip_if_no_npu(ov::Core& core) {
    const auto devices = core.get_available_devices();
    if (std::find(devices.begin(), devices.end(), "NPU") == devices.end()) {
        GTEST_SKIP() << "No available devices.";
    }
}

}  // namespace

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

TEST(SerializationTestNPUW, CompiledModelPhase0CompatibilityExportSucceedsWithStaticAttention) {
    ov::Core ov_core;
    skip_if_no_npu(ov_core);

    auto compiled = ov_core.compile_model(build_chunked_prefill_model(), "NPU", make_phase0_base_config());
    std::stringstream blob;
    EXPECT_NO_THROW(compiled.export_model(blob));
}

TEST(SerializationTestNPUW, CompiledModelPhase0CompatibilityRejectsCpuPinnedSubgraphExport) {
    ov::Core ov_core;
    skip_if_no_npu(ov_core);

    auto compiled = ov_core.compile_model(build_chunked_prefill_model(), "NPU", make_phase0_cpu_subgraph_config());
    std::stringstream blob;

    try {
        compiled.export_model(blob);
        FAIL() << "Expected phase-0 compatibility export to reject a CPU-pinned subgraph";
    } catch (const ov::Exception& ex) {
        EXPECT_NE(std::string(ex.what()).find("device \"CPU\""), std::string::npos) << ex.what();
    }
}

