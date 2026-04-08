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

    ov::pass::Manager precondition_pm;
    precondition_pm.register_pass<ov::pass::SDPAToPagedAttention>(false, false, false, false, false, false);
    precondition_pm.register_pass<ov::pass::Serialize>("paged_causal_conv1d_real_model.xml", "paged_causal_conv1d_real_model.bin");
    precondition_pm.run_passes(model);

    const auto sdpa_after_precondition = count_ops_by_type(model, "ScaledDotProductAttention");
    const auto pa_after_precondition = count_ops_by_type(model, "PagedAttentionExtension");

    EXPECT_EQ(sdpa_after_precondition, 0u);
    EXPECT_GE(pa_after_precondition, 1u);

    ov::pass::Manager common_pm;
    common_pm.register_pass<ov::pass::CommonOptimizations>();
    common_pm.register_pass<ov::pass::Serialize>("paged_causal_conv1d_real_model_after_common_opt.xml", "paged_causal_conv1d_real_model_after_common_opt.bin");
    common_pm.run_passes(model);

    const auto pcc_count = count_ops_by_type(model, "PagedCausalConv1D");
    const auto group_conv_count = count_ops_by_type(model, "GroupConvolution");

    EXPECT_GE(pcc_count, 1u);
    EXPECT_LT(group_conv_count, 3u);
}
