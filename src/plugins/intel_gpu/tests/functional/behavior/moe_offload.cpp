// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstdio>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/node_builders/moe_builders.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {

// Regression test for openvinotoolkit/model_server#4580: a MoE model read from an IR file must compile with
// OFFLOAD_RATIO set, whether or not the file was read with ENABLE_MMAP, and without the caller passing
// WEIGHTS_PATH. The parameter is the ENABLE_MMAP value passed to Core::read_model.
class MoEOffloadFromIRFile : public ::testing::Test, public ::testing::WithParamInterface<bool> {
public:
    static std::string get_test_case_name(::testing::TestParamInfo<bool> obj) {
        return std::string("mmap=") + (obj.param ? "true" : "false");
    }

protected:
    std::shared_ptr<ov::Core> core;
    std::string xml_path;
    std::string bin_path;

    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        core = ov::test::utils::PluginCache::get().core();
        // MoE fusion, and with it the offload path, is gated on supports_immad.
        const auto caps = core->get_property(ov::test::utils::DEVICE_GPU, ov::device::capabilities);
        if (std::find(caps.begin(), caps.end(), ov::intel_gpu::capability::HW_MATMUL) == caps.end()) {
            GTEST_SKIP() << "MoE offload requires a systolic GPU (HW_MATMUL capability)";
        }

        const auto prefix = ov::test::utils::generateTestFilePrefix();
        xml_path = prefix + ".xml";
        bin_path = prefix + ".bin";

        // Same compressed 3-GEMM pattern as smoke_MoE3GemmCompressedFusion in subgraph_tests/dynamic/moe.cpp.
        const ov::test::MoePatternParams moe_params{{-1, -1, 128}, /*topk*/ 2, /*number_of_experts*/ 4, /*intermediate_size*/ 256};
        auto model = ov::test::initMoE3GeMMSubgraph(moe_params,
                                                    ov::element::f32,  // data_precision
                                                    ov::element::u8,   // weights_precision
                                                    true,              // use_weight_decompression
                                                    ov::element::f16,  // decompression_precision
                                                    ov::element::f16,  // scale_precision
                                                    ov::test::utils::DecompressionType::full,
                                                    ov::test::utils::DecompressionType::full,
                                                    true,  // reshape_on_decompression
                                                    128);  // group_size
        // f16 tensors at the boundary, as MoECompressedFusionTest sets through inType/outType; without them the
        // expert block is not fused and OFFLOAD_RATIO has nothing to act on.
        ov::preprocess::PrePostProcessor ppp(model);
        ppp.input(0).tensor().set_element_type(ov::element::f16);
        ppp.output(0).tensor().set_element_type(ov::element::f16);
        model = ppp.build();
        ov::pass::Serialize(xml_path, bin_path).run_on_model(model);
    }

    void TearDown() override {
        std::remove(xml_path.c_str());
        std::remove(bin_path.c_str());
    }

    std::shared_ptr<ov::Model> read_model() const {
        return core->read_model(xml_path, std::string{}, ov::AnyMap{ov::enable_mmap(GetParam())});
    }

    void check_offloaded(const ov::CompiledModel& compiled_model) const {
        ASSERT_TRUE(compiled_model);
        // The model must have taken the fused MoE path, otherwise OFFLOAD_RATIO has nothing to act on.
        ov::test::CheckNumberOfNodesWithType(compiled_model, "moe_3gemm_fused_compressed", 1);
    }
};

TEST_P(MoEOffloadFromIRFile, CompileModelWithOffloadRatio) {
    const auto model = read_model();
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(model, ov::test::utils::DEVICE_GPU, {ov::intel_gpu::offload_ratio(50)}));
    check_offloaded(compiled_model);
}

TEST_P(MoEOffloadFromIRFile, CompileModelWithOffloadRatioAndWeightsPath) {
    const auto model = read_model();
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model =
                           core->compile_model(model, ov::test::utils::DEVICE_GPU, {ov::intel_gpu::offload_ratio(50), ov::weights_path(bin_path)}));
    check_offloaded(compiled_model);
}

INSTANTIATE_TEST_SUITE_P(smoke_MoEOffload, MoEOffloadFromIRFile, ::testing::Bool(), MoEOffloadFromIRFile::get_test_case_name);

}  // namespace
