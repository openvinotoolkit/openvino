// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/gemm.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorder.hpp>

#include "intel_gpu/plugin/output_memory_block.hpp"
#include "openvino/reference/matmul.hpp"
#include "primitive_inst.h"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

// ===================================================================
// Base fixture: shared constants, weight data, helpers.
// No network is built here — derived classes define the topology.
//
// Subgraph common to all tests:
//   Parameter{dynamic M, K} ---> Gemm ---> ...
//                                  ^
//                Constant{K, N} ---+
// All tensors are fp16.
// ===================================================================
class ZeroCopyOutputTestBase : public ::testing::Test {
protected:
    static constexpr size_t K = 8;
    static constexpr size_t N = 4;

    cldnn::engine& engine = get_test_engine();

    std::vector<ov::float16> weight_data;

    void SetUp() override {
        if (!engine.get_device_info().supports_usm) {
            GTEST_SKIP() << "USM not supported on this device, skipping zero-copy output test";
        }

        weight_data.resize(K * N);
        for (size_t i = 0; i < K * N; ++i)
            weight_data[i] = ov::float16(0.1f * static_cast<float>(i % 7) - 0.3f);
    }

    // ---- helpers --------------------------------------------------------

    /// Allocate weight memory and fill it with weight_data.
    cldnn::memory::ptr make_weight_mem() {
        layout weight_layout{ov::PartialShape{static_cast<int64_t>(K), static_cast<int64_t>(N)}, data_types::f16, format::bfyx};
        auto mem = engine.allocate_memory(weight_layout);
        set_values(mem, weight_data);
        return mem;
    }

    /// Generate deterministic fp16 input data of shape [M, K].
    std::vector<ov::float16> make_input(size_t M) const {
        std::vector<ov::float16> data(M * K);
        for (size_t i = 0; i < M * K; ++i)
            data[i] = ov::float16(0.2f * static_cast<float>((i + M) % 11) - 1.0f);
        return data;
    }

    /// Compute fp16 reference C[M,N] = A[M,K] * B[K,N] via ov::reference::matmul.
    std::vector<ov::float16> compute_reference(const std::vector<ov::float16>& input, size_t M) const {
        std::vector<ov::float16> out(M * N, ov::float16(0.f));
        ov::reference::matmul(input.data(), weight_data.data(), out.data(), ov::Shape{M, K}, ov::Shape{K, N}, ov::Shape{M, N}, false, false);
        return out;
    }

    /// Assert that the primitive's output memory is a zero-copy view over the
    /// OutputMemoryBlock (same USM host pointer, usm_host allocation type).
    static void verify_zero_copy(const cldnn::memory::ptr& output_mem, const ov::intel_gpu::OutputMemoryBlock& block) {
        auto block_mem = block.memory();
        ASSERT_NE(block_mem, nullptr) << "OutputMemoryBlock should have allocated memory";
        ASSERT_EQ(block_mem->buffer_ptr(), output_mem->buffer_ptr()) << "Zero-copy failed: primitive output and OutputMemoryBlock pointers differ";
        ASSERT_EQ(output_mem->get_allocation_type(), allocation_type::usm_host) << "Output memory should be USM host for zero-copy";
    }
};

// ===================================================================
// Direct-output fixture: gemm IS the network output.
//   input_layout("input") ---> gemm("gemm")  [output]
// ===================================================================
class ZeroCopyOutputTest : public ZeroCopyOutputTestBase {
protected:
    cldnn::memory::ptr weight_mem;
    cldnn::network::ptr net;

    void SetUp() override {
        ZeroCopyOutputTestBase::SetUp();
        if (testing::Test::IsSkipped())
            return;

        weight_mem = make_weight_mem();

        auto input_layout_dyn = layout{ov::PartialShape{ov::Dimension::dynamic(), static_cast<int64_t>(K)}, data_types::f16, format::bfyx};

        topology topo;
        topo.add(input_layout("input", input_layout_dyn));
        topo.add(data("weights", weight_mem));
        topo.add(gemm("gemm", {input_info("input"), input_info("weights")}, data_types::f16, false, false, 1.0f, 0.0f, 2, 2));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        net = get_network(engine, topo, config, get_test_stream_ptr(), false);
    }

    /// Execute the network with a concrete [M, K] input and return the output map.
    std::map<primitive_id, network_output> run_gemm(size_t M, const std::vector<ov::float16>& input_data) {
        auto input_mem = engine.allocate_memory(layout{ov::PartialShape{static_cast<int64_t>(M), static_cast<int64_t>(K)}, data_types::f16, format::bfyx});
        set_values(input_mem, input_data);
        net->set_input_data("input", input_mem);
        return net->execute();
    }
};

// -------------------------------------------------------------------
// Correctness: pointer match + numerical comparison for several shapes
// -------------------------------------------------------------------
TEST_F(ZeroCopyOutputTest, pointer_match_and_correctness) {
    ov::intel_gpu::OutputMemoryBlock output_block(engine);
    net->register_output_memory_block("gemm", &output_block);

    for (size_t M : {1, 5, 10, 3, 7}) {
        SCOPED_TRACE("M = " + std::to_string(M));

        auto input_data = make_input(M);
        auto outputs = run_gemm(M, input_data);

        auto output_mem = outputs.at("gemm").get_memory();
        ASSERT_NE(output_mem, nullptr);

        verify_zero_copy(output_mem, output_block);

        // Numerical correctness
        auto ref = compute_reference(input_data, M);

        auto out_layout = output_mem->get_layout();
        ASSERT_GE(out_layout.count(), M * N) << "Output layout should have at least M*N elements";

        cldnn::mem_lock<ov::float16> out_lock(output_mem, get_test_stream());
        for (size_t i = 0; i < M * N; ++i) {
            ASSERT_NEAR(static_cast<float>(out_lock[i]), static_cast<float>(ref[i]), 0.05f) << "Mismatch at index " << i << " (M=" << M << ")";
        }
    }

    net->clear_output_memory_blocks();
}

// -------------------------------------------------------------------
// Buffer reuse: grow-only semantics (reclaim disabled).
// Pointer must stay identical when the output shrinks and only change
// when the required size exceeds the current capacity.
// -------------------------------------------------------------------
TEST_F(ZeroCopyOutputTest, buffer_reuse_across_shapes) {
    // threshold=0 disables reclaim -> pure grow-only behaviour
    ov::intel_gpu::OutputMemoryBlock output_block(engine, 0);
    net->register_output_memory_block("gemm", &output_block);

    auto run_and_verify = [&](size_t M) {
        auto input_data = make_input(M);
        auto outputs = run_gemm(M, input_data);

        auto output_mem = outputs.at("gemm").get_memory();
        ASSERT_NE(output_mem, nullptr);
        verify_zero_copy(output_mem, output_block);
    };

    // M=10: first allocation (40 elements minimum)
    run_and_verify(10);
    void* first_ptr = output_block.memory()->buffer_ptr();
    size_t first_capacity = output_block.capacity();
    ASSERT_GE(first_capacity, size_t(40));

    // M=3: smaller shape -> buffer reused, pointer unchanged
    run_and_verify(3);
    ASSERT_EQ(output_block.memory()->buffer_ptr(), first_ptr) << "Buffer should be reused when shape shrinks (reclaim disabled)";
    ASSERT_EQ(output_block.capacity(), first_capacity);

    // M=7: still within first capacity -> pointer unchanged
    run_and_verify(7);
    ASSERT_EQ(output_block.memory()->buffer_ptr(), first_ptr) << "Buffer should be reused for shapes within existing capacity";
    ASSERT_EQ(output_block.capacity(), first_capacity);

    // M=20: 80 elements exceeds first capacity -> must grow
    run_and_verify(20);
    size_t grown_capacity = output_block.capacity();
    ASSERT_GE(grown_capacity, size_t(80)) << "Block should have grown to accommodate 80 elements";

    // M=15: back to smaller shape -> pointer unchanged after growth
    void* grown_ptr = output_block.memory()->buffer_ptr();
    run_and_verify(15);
    ASSERT_EQ(output_block.memory()->buffer_ptr(), grown_ptr) << "Buffer should be reused after growth when shape shrinks";
    ASSERT_EQ(output_block.capacity(), grown_capacity);

    net->clear_output_memory_blocks();
}

// -------------------------------------------------------------------
// Reclaim: oversized buffer is released when ratio exceeds threshold.
// -------------------------------------------------------------------
TEST_F(ZeroCopyOutputTest, reclaim_oversized_buffer) {
    // Default reclaim threshold (2x)
    ov::intel_gpu::OutputMemoryBlock output_block(engine);
    net->register_output_memory_block("gemm", &output_block);

    auto run_and_verify = [&](size_t M) {
        auto input_data = make_input(M);
        auto outputs = run_gemm(M, input_data);

        auto output_mem = outputs.at("gemm").get_memory();
        ASSERT_NE(output_mem, nullptr);
        verify_zero_copy(output_mem, output_block);
    };

    // M=25: large allocation (100+ elements with predictor padding)
    run_and_verify(25);
    void* large_ptr = output_block.memory()->buffer_ptr();
    size_t large_bytes = output_block.capacity() * sizeof(ov::float16);

    // M=1: 4 elements * 2B = 8 bytes vs 200+ bytes -> reclaim triggers
    run_and_verify(1);
    void* small_ptr = output_block.memory()->buffer_ptr();
    size_t small_bytes = output_block.capacity() * sizeof(ov::float16);

    ASSERT_NE(large_ptr, small_ptr) << "Buffer should be reclaimed when oversized by >2x";
    ASSERT_LT(small_bytes, large_bytes) << "Reclaimed buffer should be smaller than the original";

    net->clear_output_memory_blocks();
}

// ===================================================================
// Optimized-out output fixture: gemm feeds a redundant reorder that
// becomes the network output and is optimized out at graph-build time.
//   input_layout("input") ---> gemm("gemm") ---> reorder("output")
//                                                  [optimized out]
// ===================================================================
class ZeroCopyOptimizedOutputTest : public ZeroCopyOutputTestBase {
protected:
    cldnn::memory::ptr weight_mem;
    cldnn::network::ptr net;

    void SetUp() override {
        ZeroCopyOutputTestBase::SetUp();
        if (testing::Test::IsSkipped())
            return;

        weight_mem = make_weight_mem();

        auto input_layout_dyn = layout{ov::PartialShape{ov::Dimension::dynamic(), static_cast<int64_t>(K)}, data_types::f16, format::bfyx};

        topology topo;
        topo.add(input_layout("input", input_layout_dyn));
        topo.add(data("weights", weight_mem));
        topo.add(gemm("gemm", {input_info("input"), input_info("weights")}, data_types::f16, false, false, 1.0f, 0.0f, 2, 2));
        // Redundant reorder: same format and data type → will be optimized out
        topo.add(reorder("output", input_info("gemm"), format::bfyx, data_types::f16));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        net = get_network(engine, topo, config, get_test_stream_ptr(), false);
    }

    /// Execute the network with a concrete [M, K] input and return the output map.
    std::map<primitive_id, network_output> run_gemm(size_t M, const std::vector<ov::float16>& input_data) {
        auto input_mem = engine.allocate_memory(layout{ov::PartialShape{static_cast<int64_t>(M), static_cast<int64_t>(K)}, data_types::f16, format::bfyx});
        set_values(input_mem, input_data);
        net->set_input_data("input", input_mem);
        return net->execute();
    }
};

// -------------------------------------------------------------------
// Optimized-out output node: the ext_block is registered on the
// reorder (output) id.  The compute node must look ahead through the
// optimized-out user and write directly into the ext_block.
// -------------------------------------------------------------------
TEST_F(ZeroCopyOptimizedOutputTest, optimized_out_output_node) {
    // Verify the reorder is indeed optimized out
    auto reorder_inst = net->get_primitive("output");
    ASSERT_TRUE(reorder_inst->can_be_optimized()) << "Reorder should be optimized out (same format/type as input)";

    // Register the ext_block on the output (reorder) primitive id
    ov::intel_gpu::OutputMemoryBlock output_block(engine);
    net->register_output_memory_block("output", &output_block);

    for (size_t M : {1, 5, 10, 3, 7}) {
        SCOPED_TRACE("M = " + std::to_string(M));

        auto input_data = make_input(M);
        auto outputs = run_gemm(M, input_data);

        auto output_mem = outputs.at("output").get_memory();
        ASSERT_NE(output_mem, nullptr);

        // The gemm (compute) node should have written directly into the
        // ext_block's buffer, even though it is registered on the reorder.
        auto block_mem = output_block.memory();
        ASSERT_NE(block_mem, nullptr) << "OutputMemoryBlock should have allocated memory";

        // Verify the compute node's output IS the ext_block buffer
        auto gemm_output = net->get_primitive("gemm")->output_memory_ptr();
        ASSERT_EQ(gemm_output->buffer_ptr(), block_mem->buffer_ptr()) << "Compute node (gemm) should write directly into the ext_block "
                                                                         "when its sole user is an optimized-out output node";

        // Also verify the network output points to the same buffer
        ASSERT_EQ(output_mem->buffer_ptr(), block_mem->buffer_ptr()) << "Network output should be the ext_block buffer";

        ASSERT_EQ(output_mem->get_allocation_type(), allocation_type::usm_host) << "Output memory should be USM host for zero-copy";

        // Numerical correctness
        auto ref = compute_reference(input_data, M);
        cldnn::mem_lock<ov::float16> out_lock(output_mem, get_test_stream());
        for (size_t i = 0; i < M * N; ++i) {
            ASSERT_NEAR(static_cast<float>(out_lock[i]), static_cast<float>(ref[i]), 0.05f) << "Mismatch at index " << i << " (M=" << M << ")";
        }
    }

    net->clear_output_memory_blocks();
}

// ===================================================================
// Multi-user output fixture: gemm feeds TWO consumers — a redundant
// reorder (optimized out, network output) and a type-converting
// reorder (NOT optimized, also a network output).  Because gemm has
// more than one user, the look-ahead adoption (users.size()==1 guard)
// is skipped.  The optimized reorder hits the can_be_optimized()
// early return and never touches the ext_block.
//
// The test verifies graceful fallback: no crash, the ext_block is
// never populated, and outputs are still numerically correct.
//
//   input_layout("input") ---> gemm("gemm") ---> reorder("output",       bfyx, f16) [optimized out]
//                                            \-> reorder("other_output", bfyx, f32)  [executed]
// ===================================================================
class ZeroCopyMultiUserOutputTest : public ZeroCopyOutputTestBase {
protected:
    cldnn::memory::ptr weight_mem;
    cldnn::network::ptr net;

    void SetUp() override {
        ZeroCopyOutputTestBase::SetUp();
        if (testing::Test::IsSkipped())
            return;

        weight_mem = make_weight_mem();

        auto input_layout_dyn = layout{ov::PartialShape{ov::Dimension::dynamic(), static_cast<int64_t>(K)}, data_types::f16, format::bfyx};

        topology topo;
        topo.add(input_layout("input", input_layout_dyn));
        topo.add(data("weights", weight_mem));
        topo.add(gemm("gemm", {input_info("input"), input_info("weights")}, data_types::f16, false, false, 1.0f, 0.0f, 2, 2));
        // Same format/type → will be optimized out
        topo.add(reorder("output", input_info("gemm"), format::bfyx, data_types::f16));
        // Different data type → will NOT be optimized out, creates a second user of gemm
        topo.add(reorder("other_output", input_info("gemm"), format::bfyx, data_types::f32));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        net = get_network(engine, topo, config, get_test_stream_ptr(), false);
    }

    /// Execute the network with a concrete [M, K] input and return the output map.
    std::map<primitive_id, network_output> run_gemm(size_t M, const std::vector<ov::float16>& input_data) {
        auto input_mem = engine.allocate_memory(layout{ov::PartialShape{static_cast<int64_t>(M), static_cast<int64_t>(K)}, data_types::f16, format::bfyx});
        set_values(input_mem, input_data);
        net->set_input_data("input", input_mem);
        return net->execute();
    }
};

// -------------------------------------------------------------------
// Multi-user fallback: when the compute node has multiple users the
// look-ahead adoption is skipped.  The ext_block registered on the
// optimized-out output is never populated.  The system must fall back
// to the normal memory pool path without crashing, and the outputs
// must remain numerically correct.
// -------------------------------------------------------------------
TEST_F(ZeroCopyMultiUserOutputTest, fallback_when_multiple_users) {
    // Verify preconditions: the reorder is optimized out and gemm has >1 user
    auto reorder_inst = net->get_primitive("output");
    ASSERT_TRUE(reorder_inst->can_be_optimized()) << "Reorder should be optimized out (same format/type as input)";

    auto gemm_inst = net->get_primitive("gemm");
    ASSERT_GT(gemm_inst->get_user_insts().size(), 1u) << "gemm should have more than one user in this topology";

    // Register the ext_block on the optimized-out output
    ov::intel_gpu::OutputMemoryBlock output_block(engine);
    net->register_output_memory_block("output", &output_block);

    for (size_t M : {1, 5, 10, 3, 7}) {
        SCOPED_TRACE("M = " + std::to_string(M));

        auto input_data = make_input(M);
        auto outputs = run_gemm(M, input_data);

        // The ext_block should NOT have been populated because the look-ahead
        // adoption was blocked by users.size() != 1, and the optimized reorder
        // hit the can_be_optimized() early return without reaching ext_block code.
        ASSERT_EQ(output_block.memory(), nullptr) << "OutputMemoryBlock should NOT be allocated when the compute node "
                                                     "has multiple users (zero-copy look-ahead is disabled)";

        // Both outputs should still be available and correct
        auto output_mem = outputs.at("output").get_memory();
        ASSERT_NE(output_mem, nullptr);

        auto other_output_mem = outputs.at("other_output").get_memory();
        ASSERT_NE(other_output_mem, nullptr);

        // Verify f16 output ("output") correctness
        auto ref = compute_reference(input_data, M);
        {
            cldnn::mem_lock<ov::float16> lock(output_mem, get_test_stream());
            for (size_t i = 0; i < M * N; ++i) {
                ASSERT_NEAR(static_cast<float>(lock[i]), static_cast<float>(ref[i]), 0.05f) << "f16 output mismatch at index " << i << " (M=" << M << ")";
            }
        }

        // Verify f32 output ("other_output") correctness
        {
            cldnn::mem_lock<float> lock(other_output_mem, get_test_stream());
            for (size_t i = 0; i < M * N; ++i) {
                ASSERT_NEAR(lock[i], static_cast<float>(ref[i]), 0.05f) << "f32 output mismatch at index " << i << " (M=" << M << ")";
            }
        }
    }

    net->clear_output_memory_blocks();
}
