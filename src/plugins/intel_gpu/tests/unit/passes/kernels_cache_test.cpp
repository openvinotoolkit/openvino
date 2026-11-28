// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/network.hpp"

#include "data_inst.h"
#include "eltwise_inst.h"
#include "reshape_inst.h"
#include "shape_of_inst.h"
#include "fully_connected_inst.h"
#include "permute_inst.h"
#include "reduce_inst.h"
#include "to_string_utils.h"
#include "program_wrapper.h"
#include "pass_manager.h"
#include "kernel_base.h"

#include <memory>
#include <regex>

using namespace cldnn;
using namespace ::tests;

TEST(kernels_cache, reuse_kernel_for_static_model_01) {
    auto& engine = get_test_engine();

    auto input0 = engine.allocate_memory({{1, 1, 4, 5}, data_types::f16, format::bfyx});
    auto input1 = engine.allocate_memory({{1, 1, 4, 5}, data_types::f16, format::bfyx});
    auto input2 = engine.allocate_memory({{1, 1, 4, 5}, data_types::f16, format::bfyx});
    auto input3 = engine.allocate_memory({{1, 1, 4, 5}, data_types::f16, format::bfyx});
    auto input4 = engine.allocate_memory({{1, 1, 4, 5}, data_types::f16, format::bfyx});
    auto input5 = engine.allocate_memory({{1, 1, 4, 5}, data_types::f16, format::bfyx});
    auto weights1 = engine.allocate_memory({{1, 3, 2, 3 }, data_types::f16, format::bfyx});
    auto weights2 = engine.allocate_memory({{1, 3, 2, 3 }, data_types::f16, format::bfyx});

    topology topology(input_layout("input0", input0->get_layout()),
                      input_layout("input1", input1->get_layout()),
                      input_layout("input2", input2->get_layout()),
                      input_layout("input3", input3->get_layout()),
                      input_layout("input4", input4->get_layout()),
                      input_layout("input5", input5->get_layout()),
                      data("weights1", weights1),
                      data("weights2", weights2),
                      concatenation("concat1",
                                    { input_info("input0"), input_info("input1"), input_info("input2") },
                                    1,
                                    data_types::f16),
                      convolution("conv1", input_info("concat1"), "weights1", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false),
                      concatenation("concat2",
                                    { input_info("input3"), input_info("input4"), input_info("input5") },
                                    1,
                                    data_types::f16),
                      convolution("conv2", input_info("concat2"), "weights2", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false),
                      eltwise("sum", {input_info("concat1"), input_info("concat2")}, eltwise_mode::sum),
                      reorder("output", input_info("sum"), {{3, 2}, data_types::f16, format::bfyx}));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, false);
    auto& cache = prog->get_kernels_cache();
    auto& conv1_node = prog->get_node("conv1");
    auto& conv2_node = prog->get_node("conv2");
    auto conv1_kernels = conv1_node.get_selected_impl()->get_kernels();
    cache.add_to_cached_kernels(conv1_kernels);
    auto conv2_kernels = conv2_node.get_selected_impl()->get_kernels();
    cache.add_to_cached_kernels(conv2_kernels);
    ASSERT_EQ(conv1_kernels.size(), conv2_kernels.size());
    for (size_t idx = 0; idx < conv1_kernels.size(); idx++) {
        auto conv1_kern = cache.get_cached_kernel_id(conv1_kernels[idx]);
        auto conv2_kern = cache.get_cached_kernel_id(conv2_kernels[idx]);
        ASSERT_EQ(conv1_kern, conv2_kern);
    }

    auto& concat1_node = prog->get_node("concat1");
    auto& concat2_node = prog->get_node("concat2");
    auto concat1_kernels = concat1_node.get_selected_impl()->get_kernels();
    cache.add_to_cached_kernels(concat1_kernels);
    auto concat2_kernels = concat2_node.get_selected_impl()->get_kernels();
    cache.add_to_cached_kernels(concat2_kernels);
    ASSERT_EQ(concat1_kernels.size(), concat2_kernels.size());
    for (size_t idx = 0; idx < concat1_kernels.size(); idx++) {
        auto concat1_kern = cache.get_cached_kernel_id(concat1_kernels[idx]);
        auto concat2_kern = cache.get_cached_kernel_id(concat2_kernels[idx]);
        ASSERT_EQ(concat1_kern, concat2_kern);
    }
}

TEST(kernels_cache, sub_kernel_ordering_test) {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    ov::threading::IStreamsExecutor::Config task_executor_config("sub_kernel_ordering_test", 2);
    auto executor = std::make_shared<ov::threading::CPUStreamsExecutor>(task_executor_config);
    const size_t num_kernels = 9;
    auto _kernels_cache = std::unique_ptr<kernels_cache>(new kernels_cache(engine, config, 0, executor));
    std::vector<std::string> entry_point_list;
    std::vector<std::shared_ptr<kernel_selector::KernelString>> kernel_code_list;
    for (size_t idx = 0; idx < num_kernels; idx++) {
        std::shared_ptr<kernel_selector::KernelString> kernel_string = std::make_shared<kernel_selector::KernelString>();
        std::string entry_point = "add_kernel_" + std::to_string(idx);
        std::string kernel_code =
            R"__krnl(
                __kernel void $entry_point_name(const __global float* input0, const __global float* input1, __global float* output)
                {
                    const unsigned idx = get_global_id(0);
                    output[idx] = input0[idx] + input1[idx];

                }
            )__krnl";
        kernel_code = std::regex_replace(kernel_code, std::regex("\\$entry_point_name"), entry_point);
        kernel_string->str = kernel_code;
        kernel_string->options = "-cl-mad-enable";
        kernel_string->entry_point = entry_point;
        kernel_string->batch_compilation = true;
        entry_point_list.push_back(entry_point);
        kernel_code_list.push_back(kernel_string);
    }
    kernel_impl_params dummy_params;
    auto dummy_prog = std::make_shared<program>(engine, config);
    dummy_params.prog = dummy_prog.get();
    _kernels_cache->add_kernels_source(dummy_params, kernel_code_list, false);
    _kernels_cache->build_all();
    auto _out_kernels = _kernels_cache->get_kernels(dummy_params);
    ASSERT_EQ(entry_point_list.size(), _out_kernels.size());
    for (size_t i = 0; i < entry_point_list.size(); i++) {
        ASSERT_EQ(entry_point_list[i], _out_kernels[i]->get_id());
    }
}

// NEO 23.x+ no longer implicitly declares __local overloads of intel_sub_group_block_read*,
// which broke SLM block reads in sub_group_block_read.cl.
// This test compiles a kernel that uses the _sub_group_block_read_slm* family to guard against regressions:
// if those symbols are missing, clBuildProgram fails on any driver.
TEST(kernels_cache, sub_group_block_read_slm_emulation_path_compiles) {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    ov::threading::IStreamsExecutor::Config task_executor_config("slm_block_read_regression", 2);
    auto executor = std::make_shared<ov::threading::CPUStreamsExecutor>(task_executor_config);
    auto _kernels_cache = std::unique_ptr<kernels_cache>(new kernels_cache(
        engine, config, 0, executor, kernel_selector::KernelBase::get_db().get_batch_headers()));

    auto kernel_string = std::make_shared<kernel_selector::KernelString>();
    kernel_string->str =
        R"__krnl(
            __attribute__((intel_reqd_sub_group_size(16)))
            __kernel void slm_block_read_probe(const __global ushort* in, __global uint* out) {
                __local uint   slm_u[256];
                __local ushort slm_s[256];
                const uint gid = get_global_id(0);
                const uint lid = get_local_id(0);
                slm_u[lid] = (uint)in[gid];
                slm_s[lid] = in[gid];
                barrier(CLK_LOCAL_MEM_FENCE);

                uint    a  = _sub_group_block_read_slm(slm_u);
                ushort  b  = _sub_group_block_read_slm_us(slm_s);
                ushort4 c4 = _sub_group_block_read_slm_us4(slm_s);
                ushort8 c8 = _sub_group_block_read_slm_us8(slm_s);
                uint4   d  = as_uint4(BLOCK_READN_SLM(uint, 4, slm_u, 0));

                out[gid] = a + (uint)b + (uint)c4.s0 + (uint)c8.s0 + d.x;
            }
        )__krnl";
    kernel_string->language = kernel_language::OCLC;
    kernel_string->options = "-cl-std=CL2.0";
    kernel_string->entry_point = "slm_block_read_probe";
    kernel_string->batch_compilation = true;

    std::vector<std::shared_ptr<kernel_selector::KernelString>> kernel_code_list{kernel_string};
    kernel_impl_params dummy_params;
    auto dummy_prog = std::make_shared<program>(engine, config);
    dummy_params.prog = dummy_prog.get();
    _kernels_cache->add_kernels_source(dummy_params, kernel_code_list, false);

    ASSERT_NO_THROW(_kernels_cache->build_all());
    ASSERT_EQ(_kernels_cache->get_kernels(dummy_params).size(), size_t(1));
}


TEST(kernels_cache, reuse_kernels_property) {
    auto& engine = get_test_engine();

    auto input0 = engine.allocate_memory({{1, 1, 4, 5}, data_types::f16, format::bfyx});
    auto input1 = engine.allocate_memory({{1, 1, 4, 5}, data_types::f16, format::bfyx});
    auto input2 = engine.allocate_memory({{1, 1, 4, 5}, data_types::f16, format::bfyx});
    auto input3 = engine.allocate_memory({{1, 1, 4, 5}, data_types::f16, format::bfyx});
    auto input4 = engine.allocate_memory({{1, 1, 4, 5}, data_types::f16, format::bfyx});
    auto input5 = engine.allocate_memory({{1, 1, 4, 5}, data_types::f16, format::bfyx});
    auto weights1 = engine.allocate_memory({{1, 3, 2, 3 }, data_types::f16, format::bfyx});
    auto weights2 = engine.allocate_memory({{1, 3, 2, 3 }, data_types::f16, format::bfyx});

    topology topology(input_layout("input0", input0->get_layout()),
                      input_layout("input1", input1->get_layout()),
                      input_layout("input2", input2->get_layout()),
                      input_layout("input3", input3->get_layout()),
                      input_layout("input4", input4->get_layout()),
                      input_layout("input5", input5->get_layout()),
                      data("weights1", weights1),
                      data("weights2", weights2),
                      concatenation("concat1",
                                    { input_info("input0"), input_info("input1"), input_info("input2") },
                                    1,
                                    data_types::f16),
                      convolution("conv1", input_info("concat1"), "weights1", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false),
                      concatenation("concat2",
                                    { input_info("input3"), input_info("input4"), input_info("input5") },
                                    1,
                                    data_types::f16),
                      convolution("conv2", input_info("concat2"), "weights2", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false),
                      eltwise("sum", {input_info("concat1"), input_info("concat2")}, eltwise_mode::sum),
                      reorder("output", input_info("sum"), {{3, 2}, data_types::f16, format::bfyx}));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::hint::enable_kernels_reuse(true));
    auto prog = program::build_program(engine, topology, config, false, false);
    auto& cache = prog->get_kernels_cache();
    auto& conv1_node = prog->get_node("conv1");
    auto& conv2_node = prog->get_node("conv2");
    auto conv1_kernels = conv1_node.get_selected_impl()->get_kernels();
    cache.add_to_cached_kernels(conv1_kernels);
    auto conv2_kernels = conv2_node.get_selected_impl()->get_kernels();
    cache.add_to_cached_kernels(conv2_kernels);
    ASSERT_EQ(conv1_kernels.size(), conv2_kernels.size());
    for (size_t idx = 0; idx < conv1_kernels.size(); idx++) {
        auto conv1_kern = cache.get_cached_kernel_id(conv1_kernels[idx]);
        auto conv2_kern = cache.get_cached_kernel_id(conv2_kernels[idx]);
        ASSERT_EQ(conv1_kern, conv2_kern);
        ASSERT_TRUE(conv1_kernels[idx]->is_same(*conv2_kernels[idx].get()));
    }

    auto& concat1_node = prog->get_node("concat1");
    auto& concat2_node = prog->get_node("concat2");
    auto concat1_kernels = concat1_node.get_selected_impl()->get_kernels();
    cache.add_to_cached_kernels(concat1_kernels);
    auto concat2_kernels = concat2_node.get_selected_impl()->get_kernels();
    cache.add_to_cached_kernels(concat2_kernels);
    ASSERT_EQ(concat1_kernels.size(), concat2_kernels.size());
    for (size_t idx = 0; idx < concat1_kernels.size(); idx++) {
        auto concat1_kern = cache.get_cached_kernel_id(concat1_kernels[idx]);
        auto concat2_kern = cache.get_cached_kernel_id(concat2_kernels[idx]);
        ASSERT_EQ(concat1_kern, concat2_kern);
        ASSERT_TRUE(concat1_kernels[idx]->is_same(*concat2_kernels[idx].get()));
    }
}

namespace {

// The OV_NV_SUB_GROUP_SIZE value the preprocessor sees at the first occurrence of `marker`: the rest of the line of the
// last redefinition ahead of it.
std::string nv_sub_group_size_at(const std::string& src, const std::string& marker) {
    static const std::string define = "#define OV_NV_SUB_GROUP_SIZE ";
    const size_t pos = src.find(marker);
    if (pos == std::string::npos) {
        return "<marker not found>";
    }
    const size_t def = src.rfind(define, pos);
    if (def == std::string::npos) {
        return "<not defined>";
    }
    const size_t value = def + define.size();
    return src.substr(value, src.find('\n', value) - value);
}

}  // namespace

TEST(nvidia_replace_reqd_sub_group_size, both_attribute_spellings) {
    const std::string src = R"__krnl(
int lane_helper(void) { return 0; }
REQD_SUB_GROUP_SIZE(16)
__kernel void macro_kernel(__global int* out) {}
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void attribute_kernel(__global int* out) {}
)__krnl";
    const std::string out = nvidia_replace_reqd_sub_group_size(src);

    // A helper ahead of the first kernel gets the first request in source order, whichever spelling states it.
    EXPECT_EQ(nv_sub_group_size_at(out, "lane_helper"), "(16)");
    EXPECT_EQ(nv_sub_group_size_at(out, "macro_kernel"), "(16)");
    EXPECT_EQ(nv_sub_group_size_at(out, "attribute_kernel"), "(8)");
    // The macro expands to nothing on NVIDIA and is kept; the raw attribute is unknown to the NVIDIA compiler and goes.
    EXPECT_NE(out.find("REQD_SUB_GROUP_SIZE(16)"), std::string::npos);
    EXPECT_EQ(out.find("intel_reqd_sub_group_size"), std::string::npos);
}

TEST(nvidia_replace_reqd_sub_group_size, define_lines_are_skipped) {
    // common.cl defines the macro in terms of the attribute; neither occurrence on that line is a kernel's request.
    const std::string define_line =
        "#define REQD_SUB_GROUP_SIZE(sg_size) __attribute__((intel_reqd_sub_group_size(sg_size)))";
    const std::string src = "\n" + define_line + "\nREQD_SUB_GROUP_SIZE(16)\n__kernel void probe(__global int* out) {}\n";
    const std::string out = nvidia_replace_reqd_sub_group_size(src);

    EXPECT_NE(out.find(define_line), std::string::npos);
    EXPECT_EQ(out.find("OV_NV_SUB_GROUP_SIZE (sg_size)"), std::string::npos);
    EXPECT_EQ(nv_sub_group_size_at(out, "probe"), "(16)");
}

TEST(nvidia_replace_reqd_sub_group_size, request_syntax) {
    // A request's argument can be any expression: it runs to the matching ')', not the first one, and a comma inside
    // nested parentheses does not split it. The raw attribute is found wherever it sits in an attribute list; only it
    // is taken out, and the other attributes of the list stay (a list left with a stray comma is still valid).
    const std::string src = R"__krnl(
__attribute__((intel_reqd_sub_group_size((8))))
__kernel void parenthesized_kernel(__global int* out) {}
REQD_SUB_GROUP_SIZE(MIN((16),(32)))
__kernel void nested_kernel(__global int* out) {}
__attribute__((intel_reqd_sub_group_size(16), reqd_work_group_size(64,1,1)))
__kernel void request_first_kernel(__global int* out) {}
__attribute__((reqd_work_group_size(64,1,1), intel_reqd_sub_group_size(8)))
__kernel void request_last_kernel(__global int* out) {}
)__krnl";
    const std::string out = nvidia_replace_reqd_sub_group_size(src);

    EXPECT_EQ(nv_sub_group_size_at(out, "parenthesized_kernel"), "((8))");
    EXPECT_EQ(nv_sub_group_size_at(out, "nested_kernel"), "(MIN((16),(32)))");
    EXPECT_EQ(nv_sub_group_size_at(out, "request_first_kernel"), "(16)");
    EXPECT_EQ(nv_sub_group_size_at(out, "request_last_kernel"), "(8)");
    EXPECT_EQ(out.find("intel_reqd_sub_group_size"), std::string::npos);
    const std::string wg = "reqd_work_group_size(64,1,1)";
    size_t wg_count = 0;
    for (size_t pos = out.find(wg); pos != std::string::npos; pos = out.find(wg, pos + wg.size())) {
        ++wg_count;
    }
    EXPECT_EQ(wg_count, 2u);
}

TEST(nvidia_replace_reqd_sub_group_size, work_group_size_fallback) {
    // With no stated sub-group size, a work-group of 8 or 16 work-items is taken as the sub-group size. The choice is
    // left to the preprocessor because the dimensions are usually jit constants; these expressions evaluate to 8u
    // and 16u. reqd_work_group_size is found anywhere in an attribute list, and its dimensions are split at the
    // commas outside parentheses only, so each can be an expression.
    const std::string src = R"__krnl(
__attribute__((reqd_work_group_size(8,1,1)))
__kernel void wg8_kernel(__global int* out) {}
__attribute__((reqd_work_group_size(16,1,1)))
__kernel void wg16_kernel(__global int* out) {}
__attribute__((vec_type_hint(float4), reqd_work_group_size((2*4),1,1)))
__kernel void wg_last_kernel(__global int* out) {}
__attribute__((reqd_work_group_size(MAX(8,16),1,1), vec_type_hint(float4)))
__kernel void wg_first_kernel(__global int* out) {}
)__krnl";
    const std::string out = nvidia_replace_reqd_sub_group_size(src);

    EXPECT_EQ(nv_sub_group_size_at(out, "wg8_kernel"),
              "(((8)*(1)*(1)) == 8u ? 8u : (((8)*(1)*(1)) == 16u ? 16u : OV_NV_DEFAULT_SUB_GROUP_SIZE))");
    EXPECT_EQ(nv_sub_group_size_at(out, "wg16_kernel"),
              "(((16)*(1)*(1)) == 8u ? 8u : (((16)*(1)*(1)) == 16u ? 16u : OV_NV_DEFAULT_SUB_GROUP_SIZE))");
    EXPECT_EQ(nv_sub_group_size_at(out, "wg_last_kernel"),
              "((((2*4))*(1)*(1)) == 8u ? 8u : ((((2*4))*(1)*(1)) == 16u ? 16u : OV_NV_DEFAULT_SUB_GROUP_SIZE))");
    EXPECT_EQ(nv_sub_group_size_at(out, "wg_first_kernel"),
              "(((MAX(8,16))*(1)*(1)) == 8u ? 8u : (((MAX(8,16))*(1)*(1)) == 16u ? 16u : OV_NV_DEFAULT_SUB_GROUP_SIZE))");
    // reqd_work_group_size is standard OpenCL and is kept, as are the other attributes of its list. The redefinition is
    // placed right before reqd_work_group_size, so the attributes ahead of it are looked for on their own.
    EXPECT_NE(out.find("reqd_work_group_size(8,1,1)"), std::string::npos);
    EXPECT_NE(out.find("__attribute__((vec_type_hint(float4), "), std::string::npos);
    EXPECT_NE(out.find("reqd_work_group_size((2*4),1,1)))"), std::string::npos);
    EXPECT_NE(out.find("reqd_work_group_size(MAX(8,16),1,1), vec_type_hint(float4)))"), std::string::npos);

    // An explicit declaration wins over the work-group size: softmax_gpu_bf asks for 16 and launches a work-group of 4.
    const std::string explicit_src =
        "REQD_SUB_GROUP_SIZE(16)\n__attribute__((reqd_work_group_size(4,1,1)))\n__kernel void explicit_kernel(__global int* out) {}\n";
    const std::string explicit_out = nvidia_replace_reqd_sub_group_size(explicit_src);
    EXPECT_EQ(nv_sub_group_size_at(explicit_out, "explicit_kernel"), "(16)");
    EXPECT_EQ(explicit_out.find("== 8u"), std::string::npos);
}

TEST(nvidia_replace_reqd_sub_group_size, default_is_restored_between_entries) {
    const std::string entry_a = "REQD_SUB_GROUP_SIZE(16)\n__kernel void kernel_a(__global int* out) {}\n";
    const std::string entry_b = "__kernel void kernel_b(__global int* out) {}\n";

    // A source that asks for no particular sub-group size is passed through as is.
    EXPECT_EQ(nvidia_replace_reqd_sub_group_size(entry_b), entry_b);

    // Entries are concatenated after the rewrite, as kernels_cache::build_batch() does.
    const std::string batch = nvidia_replace_reqd_sub_group_size(entry_a) + nvidia_replace_reqd_sub_group_size(entry_b);
    EXPECT_EQ(nv_sub_group_size_at(batch, "kernel_a"), "(16)");
    EXPECT_EQ(nv_sub_group_size_at(batch, "kernel_b"), "(OV_NV_DEFAULT_SUB_GROUP_SIZE)");
}
