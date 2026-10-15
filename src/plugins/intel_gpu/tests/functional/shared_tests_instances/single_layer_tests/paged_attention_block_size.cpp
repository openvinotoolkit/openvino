// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "shared_test_classes/single_op/paged_attention.hpp"

namespace ov {
namespace test {
namespace {

// Verifies that the GPU plugin publishes the PagedAttention block size it will
// actually use (16 standard, 256 XAttention) on the runtime model. A swapped or
// missing value here would silently corrupt logical-to-physical block mapping.
class PagedAttentionBlockSizeTest : public PagedAttentionLayerTest {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        // We reuse only the protected make_paged_attn_model() helper and compile
        // standalone models, so skip the parameterized PagedAttentionLayerTest flow.
        targetDevice = ov::test::utils::DEVICE_GPU;
    }

    void check_block_size(ov::element::Type data_type, bool enable_xattn, int64_t block_size, size_t expected) {
        function = make_paged_attn_model(data_type,
                                         enable_xattn,
                                         /*head_size=*/64,
                                         /*head_num=*/8,
                                         /*use_sink_input=*/false,
                                         /*sliding_window=*/0,
                                         /*use_alibi=*/false,
                                         /*max_ctx_len=*/1024,
                                         /*use_rotation=*/false,
                                         block_size);
        ov::Core core;
        auto compiled = core.compile_model(function, ov::test::utils::DEVICE_GPU);
        auto runtime_model = compiled.get_runtime_model();
        ASSERT_NE(runtime_model, nullptr);
        ASSERT_TRUE(runtime_model->has_rt_info(ov::exec_model_info::PAGED_ATTENTION_BLOCK_SIZE));
        EXPECT_EQ(runtime_model->get_rt_info<size_t>(ov::exec_model_info::PAGED_ATTENTION_BLOCK_SIZE), expected);

        bool found_pa_node = false;
        for (const auto& op : runtime_model->get_ops()) {
            const auto& rt_info = op->get_rt_info();
            auto it = rt_info.find("block_size");
            if (it != rt_info.end()) {
                found_pa_node = true;
                EXPECT_EQ(it->second.as<std::string>(), std::to_string(expected));
            }
        }
        EXPECT_TRUE(found_pa_node);
    }
};

TEST_F(PagedAttentionBlockSizeTest, StandardBlockSize) {
    check_block_size(ov::element::f16, /*enable_xattn=*/false, /*block_size=*/16, /*expected=*/16);
}

TEST_F(PagedAttentionBlockSizeTest, XAttentionBlockSize) {
    check_block_size(ov::element::f16, /*enable_xattn=*/true, /*block_size=*/256, /*expected=*/256);
}

}  // namespace
}  // namespace test
}  // namespace ov
