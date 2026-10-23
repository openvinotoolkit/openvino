// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
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
        ov::Core core;
        if (enable_xattn) {
            try {
                const auto caps = core.get_property(ov::test::utils::DEVICE_GPU, ov::device::capabilities);
                if (std::find(caps.begin(), caps.end(), ov::intel_gpu::capability::HW_MATMUL) == caps.end()) {
                    GTEST_SKIP() << "XAttention requires a GPU with HW_MATMUL capability (Xe2/Xe3)";
                }
            } catch (const std::exception& e) {
                GTEST_SKIP() << "Unable to query GPU capabilities for XAttention: " << e.what();
            }
        }

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

        if (enable_xattn) {
            // GPU TransformationsPipeline enables XAttention only when xattention_block_size
            // is a Parameter named "xattention_block_size".
            constexpr size_t xattention_block_size_idx = 18;
            bool replaced = false;
            for (const auto& op : function->get_ops()) {
                if (ov::is_type<ov::op::PagedAttentionExtension>(op)) {
                    auto xattention_block_size = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{});
                    xattention_block_size->set_friendly_name("xattention_block_size");
                    op->input(xattention_block_size_idx).replace_source_output(xattention_block_size->output(0));
                    function->add_parameters({xattention_block_size});
                    function->validate_nodes_and_infer_types();
                    replaced = true;
                    break;
                }
            }
            ASSERT_TRUE(replaced);
        }

        ov::CompiledModel compiled;
        try {
            compiled = core.compile_model(function, ov::test::utils::DEVICE_GPU);
        } catch (const std::exception& e) {
            const std::string message = e.what();
            if (enable_xattn && (message.find("XAttention is not supported") != std::string::npos ||
                                 message.find("IGC version incompatible") != std::string::npos || message.find("CM is not enabled") != std::string::npos)) {
                GTEST_SKIP() << message;
            }
            throw;
        }

        auto runtime_model = compiled.get_runtime_model();
        ASSERT_NE(runtime_model, nullptr);
        ASSERT_TRUE(runtime_model->has_rt_info(ov::exec_model_info::PAGED_ATTENTION_BLOCK_SIZE));
        EXPECT_EQ(runtime_model->get_rt_info<size_t>(ov::exec_model_info::PAGED_ATTENTION_BLOCK_SIZE), expected);

        auto pa_nodes = ov::test::GetNodesWithTypes(runtime_model, {"paged_attention", "PagedAttention"});
        ASSERT_FALSE(pa_nodes.empty());
        for (const auto& op : pa_nodes) {
            const auto& rt_info = op->get_rt_info();
            auto it = rt_info.find("block_size");
            ASSERT_NE(it, rt_info.end());
            EXPECT_EQ(it->second.as<std::string>(), std::to_string(expected));
        }
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
