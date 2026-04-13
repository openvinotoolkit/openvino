// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "openvino/pass/manager.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/core.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"

namespace {

size_t count_ops_by_type(const std::shared_ptr<ov::Model>& model, const std::string& type_name) {
    size_t count = 0;
    for (const auto& op : model->get_ordered_ops()) {
        if (op->get_type_name() == type_name) {
            ++count;
        }
    }
    return count;
}

}  // namespace

TEST(PagedCausalConv1DRealModel, SDPAToPAThenCommonOptimizations) {
    const char* env_path = std::getenv("OV_PCC_REAL_MODEL_PATH");
    if (!env_path || std::string(env_path).empty()) {
        GTEST_SKIP() << "OV_PCC_REAL_MODEL_PATH is not set";
    }

    ov::Core core;
    auto model = core.read_model(env_path);

    ov::pass::Manager sdpa_to_pa_pm;
    sdpa_to_pa_pm.register_pass<ov::pass::SDPAToPagedAttention>(false, false, false, false, false, false);
    sdpa_to_pa_pm.register_pass<ov::pass::Serialize>("paged_causal_conv1d_real_model.xml",
                                                     "paged_causal_conv1d_real_model.bin");
    sdpa_to_pa_pm.run_passes(model);

    EXPECT_EQ(count_ops_by_type(model, "ScaledDotProductAttention"), 0u);
    EXPECT_GE(count_ops_by_type(model, "PagedAttentionExtension"), 1u);
    EXPECT_GE(count_ops_by_type(model, "PagedCausalConv1D"), 1u);
    const auto group_conv_count_after_sdpa_to_pa = count_ops_by_type(model, "GroupConvolution");

    ov::pass::Manager common_pm;
    common_pm.register_pass<ov::pass::CommonOptimizations>();
    common_pm.register_pass<ov::pass::Serialize>("paged_causal_conv1d_real_model_after_common_opt.xml",
                                                 "paged_causal_conv1d_real_model_after_common_opt.bin");
    common_pm.run_passes(model);

    EXPECT_GE(count_ops_by_type(model, "PagedCausalConv1D"), 1u);
    EXPECT_LE(count_ops_by_type(model, "GroupConvolution"), group_conv_count_after_sdpa_to_pa);
}

TEST(PagedCausalConv1DRealModel, SDPAToPACreatesSeveralPagedOps) {
    const char* env_path = std::getenv("OV_PCC_REAL_MODEL_PATH");
    if (!env_path || std::string(env_path).empty()) {
        GTEST_SKIP() << "OV_PCC_REAL_MODEL_PATH is not set";
    }

    ov::Core core;
    auto model = core.read_model(env_path);

    ov::pass::Manager sdpa_to_pa_pm;
    sdpa_to_pa_pm.register_pass<ov::pass::SDPAToPagedAttention>(false, false, false, false, false, false);
    sdpa_to_pa_pm.run_passes(model);

    EXPECT_GE(count_ops_by_type(model, "PagedAttentionExtension"), 2u);
    EXPECT_GE(count_ops_by_type(model, "PagedCausalConv1D"), 2u);
}
