// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/moe_pattern.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {
using namespace ov;
using namespace ov::test;

TEST_P(MOEExpertTest, Inference) {
    auto actualOutputs = run_test(function);
    check_op("moe_expert", 1);
    check_op("OneHot", 0);
    configuration.insert({"INFERENCE_PRECISION_HINT", "FP32"});
    auto expectedOutputs = run_test(functionRefs);
    check_op("moe_expert", 0);
    check_op("OneHot", 1);

    for (size_t i = 0; i < actualOutputs.size(); i++) {
        ov::test::utils::compare(expectedOutputs[i], actualOutputs[i], abs_threshold, rel_threshold);
    }
}

// TODO: cache feature
TEST_P(MOEExpertTest, Inference_cached) {
    core->set_property(ov::cache_dir(""));
    auto func_bak = function;
    std::vector<ov::Tensor> actualOutputs, expectedOutputs;
    ElementType inType;
    std::tie(inType) = this->GetParam();
    expectedOutputs = run_test(functionRefs);
    check_op("moe_expert", 0);

    function = func_bak;

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
        check_op("moe_expert", 1);
        ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
        ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
        ov::test::utils::removeDir(cacheDirName);
    }

    for (size_t i = 0; i < actualOutputs.size(); i++) {
        ov::test::utils::compare(expectedOutputs[i], actualOutputs[i], abs_threshold, rel_threshold);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_MOEExpert_basic,
                         MOEExpertTest,
                         ::testing::Combine(::testing::Values(ov::element::f32, ov::element::f16)),
                         MOEExpertTest::getTestCaseName);

} // namespace
