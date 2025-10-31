// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/moe_pattern.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "intel_gpu/runtime/engine.hpp"

namespace ov::test {

TEST_P(MOETest, Inference) {
    {
        // moe depends on onednn which depends on xmx
        auto capabilities = core->get_property(ov::test::utils::DEVICE_GPU, ov::device::capabilities);
        if (std::find(capabilities.cbegin(), capabilities.cend(), ov::intel_gpu::capability::HW_MATMUL) == capabilities.cend())
            GTEST_SKIP();
    }
    auto ret = core->get_property(ov::test::utils::DEVICE_GPU, ov::internal::supported_properties);
    targetDevice = ov::test::utils::DEVICE_GPU;
    auto actualOutputs = run_test(function);
    check_op("moe", 1);
    check_op("OneHot", 0);
    configuration.insert({"INFERENCE_PRECISION_HINT", "FP32"});
    targetDevice = ov::test::utils::DEVICE_CPU;
    auto expectedOutputs = run_test(functionRefs);

    for (size_t i = 0; i < actualOutputs.size(); i++) {
        ov::test::utils::compare(expectedOutputs[i], actualOutputs[i], abs_threshold, rel_threshold);
    }
}

TEST_P(MOETest, Inference_cached) {
    {
        auto capabilities = core->get_property(ov::test::utils::DEVICE_GPU, ov::device::capabilities);
        if (std::find(capabilities.cbegin(), capabilities.cend(), ov::intel_gpu::capability::HW_MATMUL) == capabilities.cend())
            GTEST_SKIP();
    }
    core->set_property(ov::cache_dir(""));
    auto func_bak = function;
    std::vector<ov::Tensor> actualOutputs, expectedOutputs;
    ElementType inType, weiType;
    std::tie(inType, weiType) = this->GetParam();
    targetDevice = ov::test::utils::DEVICE_CPU;
    expectedOutputs = run_test(functionRefs);

    function = func_bak;

    targetDevice = ov::test::utils::DEVICE_GPU;
    std::stringstream ss;
    ss << "gpu_model_cache_"
       << std::hash<std::string>{}(
              std::string(::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name()) +
              std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()) +
              element::Type(inType).get_type_name());
    std::string cacheDirName = ss.str();
    {
        ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
        ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
        ov::test::utils::removeDir(cacheDirName);
        core->set_property(ov::cache_dir(cacheDirName));
        compile_model();
    }
    {
        actualOutputs = run_test(function);
        check_op("moe", 1);
        ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
        ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
        ov::test::utils::removeDir(cacheDirName);
    }

    for (size_t i = 0; i < actualOutputs.size(); i++) {
        ov::test::utils::compare(expectedOutputs[i], actualOutputs[i], abs_threshold, rel_threshold);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_MOE_basic,
                         MOETest,
                         ::testing::Combine(
                            ::testing::Values(ov::element::f16),
                            ::testing::Values(ov::element::u4)),
                         MOETest::getTestCaseName);

} // namespace ov::test
