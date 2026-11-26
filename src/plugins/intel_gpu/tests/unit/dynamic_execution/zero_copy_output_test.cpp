// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <intel_gpu/primitives/arg_max_min.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/gemm.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <numeric>
#include <optional>

#include "intel_gpu/plugin/output_memory_block.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/reference/matmul.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"
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
        if (engine.get_device_info().dev_type == device_type::discrete_gpu) {
            GTEST_SKIP() << "Dynamic output tensor substitution is disabled on dGPU";
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
    size_t first_alloc_bytes = output_block.memory()->get_mem_tracker()->size();
    ASSERT_GE(first_alloc_bytes, size_t(40) * sizeof(ov::float16));

    // M=3: smaller shape -> buffer reused, pointer unchanged
    run_and_verify(3);
    ASSERT_EQ(output_block.memory()->buffer_ptr(), first_ptr) << "Buffer should be reused when shape shrinks (reclaim disabled)";
    ASSERT_EQ(output_block.memory()->get_mem_tracker()->size(), first_alloc_bytes);

    // M=7: still within first capacity -> pointer unchanged
    run_and_verify(7);
    ASSERT_EQ(output_block.memory()->buffer_ptr(), first_ptr) << "Buffer should be reused for shapes within existing capacity";
    ASSERT_EQ(output_block.memory()->get_mem_tracker()->size(), first_alloc_bytes);

    // M=20: 80 elements exceeds first capacity -> must grow
    run_and_verify(20);
    size_t grown_alloc_bytes = output_block.memory()->get_mem_tracker()->size();
    ASSERT_GE(grown_alloc_bytes, size_t(80) * sizeof(ov::float16)) << "Block should have grown to accommodate 80 elements";

    // M=15: back to smaller shape -> pointer unchanged after growth
    void* grown_ptr = output_block.memory()->buffer_ptr();
    run_and_verify(15);
    ASSERT_EQ(output_block.memory()->buffer_ptr(), grown_ptr) << "Buffer should be reused after growth when shape shrinks";
    ASSERT_EQ(output_block.memory()->get_mem_tracker()->size(), grown_alloc_bytes);

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
    size_t large_bytes = output_block.memory()->get_mem_tracker()->size();

    // M=1: 4 elements * 2B = 8 bytes vs 200+ bytes -> reclaim triggers
    run_and_verify(1);
    void* small_ptr = output_block.memory()->buffer_ptr();
    size_t small_bytes = output_block.memory()->get_mem_tracker()->size();

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

// -------------------------------------------------------------------
// Alternating between a caller binding and the pool-backed copy path
// (as two requests sharing the network do) must return the compute
// node's pool memory each time instead of pinning a new record.
// -------------------------------------------------------------------
TEST_F(ZeroCopyOptimizedOutputTest, alternating_caller_binding_does_not_grow_pool) {
    constexpr size_t M = 6;
    auto caller_mem = engine.allocate_memory(
        layout{ov::PartialShape{static_cast<int64_t>(M), static_cast<int64_t>(N)}, data_types::f16, format::bfyx},
        allocation_type::usm_host);
    const auto input_data = make_input(M);

    auto run_copy_path = [&]() {
        net->release_user_output_memory("output");
        run_gemm(M, input_data);
    };
    auto run_caller_binding = [&]() {
        net->set_output_memory("output", caller_mem, true);
        run_gemm(M, input_data);
    };

    run_copy_path();
    run_caller_binding();
    run_copy_path();
    const auto pool_size = net->get_memory_pool().get_non_padded_pool_size();
    ASSERT_GT(pool_size, 0u) << "Precondition: the copy path allocates gemm output from the pool";

    for (int round = 0; round < 3; ++round) {
        SCOPED_TRACE("round = " + std::to_string(round));
        run_caller_binding();
        run_copy_path();
        ASSERT_EQ(net->get_memory_pool().get_non_padded_pool_size(), pool_size);
    }
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

// ===================================================================
// Multi-output producer fixture: arg_max_min (TopK) has two outputs and
// only one of them feeds the optimized-out network output; the other
// output has no user.  Every output must still get memory, so neither
// the ext_block nor caller-owned output memory, which could only stand
// in for _outputs[0], is taken by the producer.
//   input_layout("input") ---> arg_max_min("topk") --out[port]--> reorder("output") [optimized out]
// ===================================================================
class ZeroCopyMultiOutputProducerTest : public ZeroCopyOutputTestBase {
protected:
    static constexpr size_t L = 16;
    static constexpr uint32_t top_k = 4;

    enum class OutputBinding { ext_block, caller_memory };

    // (i * 5) % L is a permutation of a row, so the values are distinct and the indices unambiguous
    static std::vector<float> make_input(size_t M) {
        std::vector<float> input_data(M * L);
        for (size_t row = 0; row < M; ++row) {
            for (size_t i = 0; i < L; ++i)
                input_data[row * L + i] = static_cast<float>((i * 5 + row * 3) % L);
        }
        return input_data;
    }

    static std::pair<std::vector<float>, std::vector<int32_t>> reference_topk(const std::vector<float>& input_data, size_t M) {
        std::vector<float> ref_values;
        std::vector<int32_t> ref_indices;
        for (size_t row = 0; row < M; ++row) {
            std::vector<int32_t> order(L);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
                return input_data[row * L + a] > input_data[row * L + b];
            });
            for (size_t k = 0; k < top_k; ++k) {
                ref_values.push_back(input_data[row * L + order[k]]);
                ref_indices.push_back(order[k]);
            }
        }
        return {ref_values, ref_indices};
    }

    cldnn::network::ptr build_network(int32_t port) {
        auto input_layout_dyn = layout{ov::PartialShape{ov::Dimension::dynamic(), static_cast<int64_t>(L)}, data_types::f32, format::bfyx};
        auto topk =
            arg_max_min("topk", {input_info("input")}, ov::op::TopKMode::MAX, top_k, 1, ov::op::TopKSortType::SORT_VALUES, true, false, data_types::f32, 2);
        topk.output_data_types = {data_types::f32, data_types::i32};

        topology topo;
        topo.add(input_layout("input", input_layout_dyn));
        topo.add(topk);
        topo.add(reorder("output", input_info("topk", port), format::bfyx, port == 0 ? data_types::f32 : data_types::i32));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        return get_network(engine, topo, config, get_test_stream_ptr(), false);
    }

    void run(int32_t port, OutputBinding binding) {
        auto net = build_network(port);
        const auto output_type = port == 0 ? data_types::f32 : data_types::i32;
        ov::intel_gpu::OutputMemoryBlock output_block(engine);
        if (binding == OutputBinding::ext_block)
            net->register_output_memory_block("output", &output_block);

        for (size_t M : {1, 5, 3}) {
            SCOPED_TRACE("M = " + std::to_string(M));

            const auto input_data = make_input(M);
            auto input_mem = engine.allocate_memory(layout{ov::PartialShape{static_cast<int64_t>(M), static_cast<int64_t>(L)}, data_types::f32, format::bfyx});
            set_values(input_mem, input_data);
            net->set_input_data("input", input_mem);
            if (binding == OutputBinding::caller_memory) {
                auto caller_mem =
                    engine.allocate_memory(layout{ov::PartialShape{static_cast<int64_t>(M), static_cast<int64_t>(top_k)}, output_type, format::bfyx},
                                           allocation_type::usm_host);
                net->set_output_memory("output", caller_mem, true);
            }

            std::map<primitive_id, network_output> outputs;
            OV_ASSERT_NO_THROW(outputs = net->execute());

            auto topk_inst = net->get_primitive("topk");
            ASSERT_NE(topk_inst->output_memory_ptr(0), nullptr);
            ASSERT_NE(topk_inst->output_memory_ptr(1), nullptr);
            ASSERT_EQ(output_block.memory(), nullptr) << "OutputMemoryBlock should NOT be allocated for a multi-output producer";

            const auto [ref_values, ref_indices] = reference_topk(input_data, M);
            auto output_mem = outputs.at("output").get_memory();
            ASSERT_NE(output_mem, nullptr);
            if (output_type == data_types::f32) {
                cldnn::mem_lock<float, mem_lock_type::read> lock(output_mem, get_test_stream());
                for (size_t i = 0; i < ref_values.size(); ++i)
                    ASSERT_EQ(lock[i], ref_values[i]) << "values mismatch at index " << i;
            } else {
                cldnn::mem_lock<int32_t, mem_lock_type::read> lock(output_mem, get_test_stream());
                for (size_t i = 0; i < ref_indices.size(); ++i)
                    ASSERT_EQ(lock[i], ref_indices[i]) << "indices mismatch at index " << i;
            }
        }

        if (binding == OutputBinding::ext_block)
            net->clear_output_memory_blocks();
    }

    // Same graph through an infer request, where wait() has to copy the indices into the output tensor
    void run_infer_request(bool caller_usm_host) {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, static_cast<int64_t>(L)});
        auto k = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {top_k});
        auto topk = std::make_shared<ov::op::v11::TopK>(input, k, 1, ov::op::TopKMode::MAX, ov::op::TopKSortType::SORT_VALUES, ov::element::i32);
        auto model = std::make_shared<ov::Model>(ov::OutputVector{topk->output(1)}, ov::ParameterVector{input});

        ov::Core core;
        auto context = core.get_default_context(get_engine_gpu_device_name()).as<ov::intel_gpu::ocl::ClContext>();
        auto request = core.compile_model(model, context).create_infer_request();

        // Bound once for the largest batch, as an application would
        const ov::Shape max_output_shape{5, top_k};
        std::optional<ov::intel_gpu::ocl::USMTensor> caller_tensor;
        if (caller_usm_host) {
            caller_tensor = context.create_usm_host_tensor(ov::element::i32, max_output_shape);
            request.set_output_tensor(ov::Tensor(ov::element::i32, max_output_shape, caller_tensor->get()));
        }

        for (size_t M : {1, 5, 3}) {
            SCOPED_TRACE("M = " + std::to_string(M));

            const auto input_data = make_input(M);
            ov::Tensor input_tensor(ov::element::f32, ov::Shape{M, L});
            std::copy(input_data.begin(), input_data.end(), input_tensor.data<float>());
            request.set_input_tensor(input_tensor);
            // -1 stays if wait() does not copy into the caller's buffer
            if (caller_tensor)
                std::fill_n(static_cast<int32_t*>(caller_tensor->get()), ov::shape_size(max_output_shape), -1);

            OV_ASSERT_NO_THROW(request.infer());

            auto actual = request.get_output_tensor();
            if (caller_tensor)
                ASSERT_EQ(actual.data(), caller_tensor->get());
            ASSERT_EQ(actual.get_shape(), (ov::Shape{M, top_k}));
            const auto* actual_data = actual.data<const int32_t>();
            ASSERT_EQ(std::vector<int32_t>(actual_data, actual_data + actual.get_size()), reference_topk(input_data, M).second);
        }
    }
};

TEST_F(ZeroCopyMultiOutputProducerTest, values_only) {
    run(0, OutputBinding::ext_block);
}

TEST_F(ZeroCopyMultiOutputProducerTest, indices_only) {
    run(1, OutputBinding::ext_block);
}

TEST_F(ZeroCopyMultiOutputProducerTest, indices_only_caller_memory) {
    run(1, OutputBinding::caller_memory);
}

TEST_F(ZeroCopyMultiOutputProducerTest, infer_request_indices_only) {
    run_infer_request(false);
}

TEST_F(ZeroCopyMultiOutputProducerTest, infer_request_indices_only_caller_usm_host) {
    run_infer_request(true);
}
