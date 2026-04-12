// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for jit_kernel_ir.hpp.
//
// Slice 1 coverage:
//   - Straight-line interval computation on hand-built IR.
//   - Linear scan assignment + peak tracker on pressure-free input.
//   - Register reuse across non-overlapping intervals.
//   - Pool overflow raises allocation_failure (no spill in Slice 1).
//   - Values defined but never read get their register freed promptly.
//   - Dump helpers don't crash on empty or populated IR.
//
// Slice 2 coverage:
//   - End-to-end IR mode: record through DSL operators, allocate, lower,
//     execute generated kernel and verify output.

#include <gtest/gtest.h>
#include <kernels/x64/jit_kernel.hpp>
#include <kernels/x64/jit_kernel_ir.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace ov::intel_cpu;
using namespace ov::intel_cpu::jit_kernel_ir;

namespace {

EmitFn stub() {
    return [](const EmitContext&) {};
}

// Helper: assert every value that appears in `intervals` (with a real start)
// has been assigned a physical register in `assignment`.
void expect_all_assigned(const std::vector<Interval>& intervals, const Assignment& assignment) {
    for (const auto& iv : intervals) {
        if (iv.start == std::numeric_limits<std::uint32_t>::max()) {
            continue;
        }
        EXPECT_TRUE(assignment.reg.count(iv.id) == 1)
            << "value %" << iv.id << " was not assigned a register";
    }
}

// Helper: assert no two intervals that overlap in op-index space landed on
// the same physical register. This is the core correctness property of the
// allocator — anything else is secondary.
void expect_no_overlap_conflict(const std::vector<Interval>& intervals,
                                const Assignment& assignment) {
    for (std::size_t i = 0; i < intervals.size(); ++i) {
        const auto& a = intervals[i];
        if (a.start == std::numeric_limits<std::uint32_t>::max()) {
            continue;
        }
        for (std::size_t j = i + 1; j < intervals.size(); ++j) {
            const auto& b = intervals[j];
            if (b.start == std::numeric_limits<std::uint32_t>::max()) {
                continue;
            }
            const bool overlap = !(a.end < b.start || b.end < a.start);
            if (!overlap) {
                continue;
            }
            const auto ra = assignment.reg.find(a.id);
            const auto rb = assignment.reg.find(b.id);
            if (ra == assignment.reg.end() || rb == assignment.reg.end()) {
                continue;
            }
            EXPECT_NE(ra->second, rb->second)
                << "overlapping intervals %" << a.id << " and %" << b.id
                << " share physical register p" << ra->second.idx;
        }
    }
}

}  // namespace

TEST(JitKernelIR, IntervalsOnStraightLine) {
    // Linear chain: a, b, c = a + b, d = c + a.
    // - a defined at op 0, last read at op 3 (in d's definition).
    // - b defined at op 1, last read at op 2.
    // - c defined at op 2, last read at op 3.
    // - d defined at op 3, never read: end == start == 3.
    IR ir;
    const value_id a = ir.def({}, stub());
    const value_id b = ir.def({}, stub());
    const value_id c = ir.def({a, b}, stub());
    const value_id d = ir.def({c, a}, stub());

    auto intervals = compute_intervals(ir);
    ASSERT_EQ(intervals.size(), 4U);

    EXPECT_EQ(intervals[a].start, 0U);
    EXPECT_EQ(intervals[a].end, 3U);  // extended by d's read
    EXPECT_EQ(intervals[b].start, 1U);
    EXPECT_EQ(intervals[b].end, 2U);  // only read in c's def
    EXPECT_EQ(intervals[c].start, 2U);
    EXPECT_EQ(intervals[c].end, 3U);
    EXPECT_EQ(intervals[d].start, 3U);
    EXPECT_EQ(intervals[d].end, 3U);  // never read
}

TEST(JitKernelIR, LinearScanFitsInPool) {
    // Same 4-op chain as above. Peak overlap is 3 (a, b, c live at op 2).
    // With a pool of 4, allocation must succeed and no overlapping pair can
    // share a register.
    IR ir;
    const value_id a = ir.def({}, stub());
    const value_id b = ir.def({}, stub());
    const value_id c = ir.def({a, b}, stub());
    const value_id d = ir.def({c, a}, stub());
    (void)d;

    auto intervals = compute_intervals(ir);
    const auto assignment = linear_scan(ir, intervals, /*pool_size=*/4);

    expect_all_assigned(intervals, assignment);
    expect_no_overlap_conflict(intervals, assignment);
    EXPECT_EQ(assignment.peak_live, 3U);
}

TEST(JitKernelIR, LinearScanReusesFreedRegisters) {
    // Build a long chain of six values where each value dies before the next
    // fresh def begins. The allocator should recycle a single register across
    // the entire chain, yielding peak_live == 2 (old + new transient window).
    //
    //   %0 = def              // live [0, 1]
    //   %1 = op(%0)           // live [1, 2]  (def), %0 dies after op 1
    //   %2 = op(%1)           // live [2, 3]
    //   %3 = op(%2)           // live [3, 4]
    //   %4 = op(%3)           // live [4, 5]
    //   %5 = op(%4)           // live [5, 5]  (never read)
    IR ir;
    value_id prev = ir.def({}, stub());
    for (int i = 0; i < 5; ++i) {
        prev = ir.def({prev}, stub());
    }
    (void)prev;

    auto intervals = compute_intervals(ir);
    const auto assignment = linear_scan(ir, intervals, /*pool_size=*/2);

    expect_all_assigned(intervals, assignment);
    expect_no_overlap_conflict(intervals, assignment);
    EXPECT_LE(assignment.peak_live, 2U);
    EXPECT_GE(assignment.peak_live, 1U);
}

TEST(JitKernelIR, LinearScanThrowsOnOverflow) {
    // Five long-lived non-rematerializable values (each reads its predecessor),
    // all bundled into one final op. Pool of 4 must fail — no remat possible
    // because every value has dependencies.
    IR ir;
    std::vector<value_id> values;
    value_id prev = ir.def({}, stub());  // first one has no reads (remat-able)
    values.push_back(prev);
    for (int i = 1; i < 5; ++i) {
        prev = ir.def({prev}, stub());   // depends on predecessor — not remat-able
        values.push_back(prev);
    }
    ir.use(values, stub());  // single op reading all five

    auto intervals = compute_intervals(ir);
    EXPECT_THROW(linear_scan(ir, intervals, /*pool_size=*/4), allocation_failure);

    // Same IR with a pool large enough should succeed.
    IR ir2;
    std::vector<value_id> values2;
    prev = ir2.def({}, stub());
    values2.push_back(prev);
    for (int i = 1; i < 5; ++i) {
        prev = ir2.def({prev}, stub());
        values2.push_back(prev);
    }
    ir2.use(values2, stub());
    auto intervals2 = compute_intervals(ir2);
    const auto assignment = linear_scan(ir2, intervals2, /*pool_size=*/5);
    expect_all_assigned(intervals2, assignment);
    expect_no_overlap_conflict(intervals2, assignment);
    EXPECT_EQ(assignment.peak_live, 5U);
}

TEST(JitKernelIR, UseWithoutDefDoesNotCreateValue) {
    // use() records an op with no def — verifies ir.value_count() doesn't
    // grow for uses, only for defs. Important for peak_live accounting: a
    // store at the end of a kernel must not be counted as a fresh live value.
    IR ir;
    const value_id v = ir.def({}, stub());
    ir.use({v}, stub());

    EXPECT_EQ(ir.value_count(), 1U);
    EXPECT_EQ(ir.ops().size(), 2U);

    auto intervals = compute_intervals(ir);
    ASSERT_EQ(intervals.size(), 1U);
    EXPECT_EQ(intervals[v].start, 0U);
    EXPECT_EQ(intervals[v].end, 1U);  // extended by the use

    const auto assignment = linear_scan(ir, intervals, /*pool_size=*/1);
    EXPECT_EQ(assignment.peak_live, 1U);
}

TEST(JitKernelIR, CopyOpRecordedAsCopy) {
    // copy() sets is_copy=true on the recorded Op. Slice 1 does not use this
    // flag for coalescing, but the recording path must be correct so Slice 4
    // can consume it.
    IR ir;
    const value_id a = ir.def({}, stub());
    const value_id b = ir.copy(a, stub());

    ASSERT_EQ(ir.ops().size(), 2U);
    auto it = ir.ops().begin();
    EXPECT_FALSE(it->is_copy);
    ++it;
    EXPECT_TRUE(it->is_copy);
    EXPECT_EQ(it->def, b);
    ASSERT_EQ(it->reads.size(), 1U);
    EXPECT_EQ(it->reads[0], a);
}

TEST(JitKernelIR, DumpProducesNonEmptyText) {
    // Smoke test for the dump helpers. The exact format is not specified —
    // the tests only require that the dump is non-empty and includes some
    // structural tokens so future refactors don't silently break the dump
    // path.
    IR ir;
    const value_id a = ir.def({}, stub());
    const value_id b = ir.def({}, stub());
    const value_id c = ir.def({a, b}, stub());
    (void)c;

    auto intervals = compute_intervals(ir);
    const auto assignment = linear_scan(ir, intervals, /*pool_size=*/3);

    std::ostringstream op_dump;
    dump_ops(op_dump, ir);
    EXPECT_NE(op_dump.str().find('%'), std::string::npos);
    EXPECT_NE(op_dump.str().find("op"), std::string::npos);

    std::ostringstream assign_dump;
    dump_assignment(assign_dump, intervals, assignment);
    EXPECT_NE(assign_dump.str().find('%'), std::string::npos);
    EXPECT_NE(assign_dump.str().find("peak_live"), std::string::npos);
}

TEST(JitKernelIR, DeadValueFreesRegisterImmediately) {
    // A value that is defined but never read should have its register
    // freed after the def op, so a subsequent def can reuse it with a
    // pool of size 1.
    IR ir;
    (void)ir.def({}, stub());  // dead
    (void)ir.def({}, stub());  // dead, would fail on pool=1 if dead value held its reg

    auto intervals = compute_intervals(ir);
    const auto assignment = linear_scan(ir, intervals, /*pool_size=*/1);
    EXPECT_EQ(assignment.peak_live, 1U);
    EXPECT_EQ(assignment.reg.size(), 2U);
}

// ── Slice 2: end-to-end integration tests ──────────────────────────────
//
// These tests exercise the full IR-mode path through the real jit_kernel
// DSL: record ops via variable operators, run allocation, lower to xbyak,
// execute the generated kernel, and verify output.

namespace {

struct VecAddParams {
    const float* a;
    const float* b;
    float* result;
};

// Kernel that adds two vectors using IR mode.
// The operator+ goes through vec_op() → IR recording → linear_scan → lowering.
template <size_t N>
struct jit_ir_vec_add_kernel : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_ir_vec_add_kernel)

    jit_ir_vec_add_kernel() : jit_kernel(jit_name()) {}

    using fn_t = void (*)(const VecAddParams*);
    fn_t fn_ = nullptr;

    void init() {
        if (create_kernel() != dnnl::impl::status::success)
            OPENVINO_THROW("Can't generate jit kernel");
        fn_ = (fn_t)(jit_ker());  // NOLINT
    }

    void operator()(const VecAddParams& args) const { fn_(&args); }

    void generate() override {
        preamble();

        auto a_ptr = arg(&VecAddParams::a);
        auto b_ptr = arg(&VecAddParams::b);
        auto r_ptr = arg(&VecAddParams::result);

        begin_ir();

        auto a = ir_load<N>(a_ptr);
        auto b = ir_load<N>(b_ptr);
        auto sum = vaddps(a, b);
        ir_store<N>(r_ptr, sum);

        end_ir();

        postamble();
    }
};

}  // namespace

TEST(JitKernelIR, EndToEndVecAdd) {
    using namespace dnnl::impl::cpu::x64;

    // Test with AVX2 (8 floats) if available
    if (mayiuse(cpu_isa_t::avx2)) {
        constexpr size_t N = 8;
        jit_ir_vec_add_kernel<N> kernel;
        kernel.init();

        alignas(32) std::array<float, N> a{}, b{}, result{};
        for (size_t i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(100 + i);
        }
        VecAddParams args{a.data(), b.data(), result.data()};
        kernel(args);

        for (size_t i = 0; i < N; ++i) {
            EXPECT_FLOAT_EQ(result[i], a[i] + b[i])
                << "mismatch at index " << i;
        }
    }
}

// Kernel that computes (a + b) * (a - b) using IR mode with sugar syntax.
// Exercises vaddps, vsubps, vmulps functors and multi-op IR.
template <size_t N>
struct jit_ir_vec_expr_kernel : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_ir_vec_expr_kernel)

    jit_ir_vec_expr_kernel() : jit_kernel(jit_name()) {}

    using fn_t = void (*)(const VecAddParams*);
    fn_t fn_ = nullptr;

    void init() {
        if (create_kernel() != dnnl::impl::status::success)
            OPENVINO_THROW("Can't generate jit kernel");
        fn_ = (fn_t)(jit_ker());  // NOLINT
    }

    void operator()(const VecAddParams& args) const { fn_(&args); }

    void generate() override {
        preamble();

        auto a_ptr = arg(&VecAddParams::a);
        auto b_ptr = arg(&VecAddParams::b);
        auto r_ptr = arg(&VecAddParams::result);

        begin_ir();

        auto a = ir_load<N>(a_ptr);
        auto b = ir_load<N>(b_ptr);

        // (a + b) * (a - b)
        auto sum  = vaddps(a, b);
        auto diff = vsubps(a, b);
        auto res  = vmulps(sum, diff);

        // Also compute a*a + b*b via fma to verify FMA path
        // (not stored — just exercises the recording path for dead values)
        (void)fma(a, a, vmulps(b, b));

        ir_store<N>(r_ptr, res);

        end_ir();
        postamble();
    }
};

TEST(JitKernelIR, EndToEndVecExpr) {
    using namespace dnnl::impl::cpu::x64;

    if (mayiuse(cpu_isa_t::avx2)) {
        constexpr size_t N = 8;
        jit_ir_vec_expr_kernel<N> kernel;
        kernel.init();

        alignas(32) std::array<float, N> a{}, b{}, result{};
        for (size_t i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i + 1);
            b[i] = static_cast<float>(i) * 0.5f;
        }
        VecAddParams args{a.data(), b.data(), result.data()};
        kernel(args);

        for (size_t i = 0; i < N; ++i) {
            EXPECT_FLOAT_EQ(result[i], (a[i] + b[i]) * (a[i] - b[i]))
                << "mismatch at index " << i;
        }
    }
}

// Kernel that computes fma(a, b, c) = a*b + c using IR mode.
struct FmaParams {
    const float* a;
    const float* b;
    const float* c;
    float* result;
};

template <size_t N>
struct jit_ir_fma_kernel : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_ir_fma_kernel)

    jit_ir_fma_kernel() : jit_kernel(jit_name()) {}

    using fn_t = void (*)(const FmaParams*);
    fn_t fn_ = nullptr;

    void init() {
        if (create_kernel() != dnnl::impl::status::success)
            OPENVINO_THROW("Can't generate jit kernel");
        fn_ = (fn_t)(jit_ker());  // NOLINT
    }

    void operator()(const FmaParams& args) const { fn_(&args); }

    void generate() override {
        preamble();

        auto a_ptr = arg(&FmaParams::a);
        auto b_ptr = arg(&FmaParams::b);
        auto c_ptr = arg(&FmaParams::c);
        auto r_ptr = arg(&FmaParams::result);

        begin_ir();

        auto a = ir_load<N>(a_ptr);
        auto b = ir_load<N>(b_ptr);
        auto c = ir_load<N>(c_ptr);

        auto res = fma(a, b, c);

        ir_store<N>(r_ptr, res);

        end_ir();
        postamble();
    }
};

TEST(JitKernelIR, EndToEndFma) {
    using namespace dnnl::impl::cpu::x64;

    if (mayiuse(cpu_isa_t::avx2)) {
        constexpr size_t N = 8;
        jit_ir_fma_kernel<N> kernel;
        kernel.init();

        alignas(32) std::array<float, N> a{}, b{}, c{}, result{};
        for (size_t i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i + 1);
            b[i] = 2.0f;
            c[i] = static_cast<float>(i) * 0.5f;
        }
        FmaParams args{a.data(), b.data(), c.data(), result.data()};
        kernel(args);

        for (size_t i = 0; i < N; ++i) {
            EXPECT_FLOAT_EQ(result[i], a[i] * b[i] + c[i])
                << "fma mismatch at index " << i;
        }
    }
}

// ── Slice 3: foreach in IR mode ────────────────────────────────────────

// Kernel that scales an array by a constant: dst[i] = src[i] * scale.
// The scale is loaded BEFORE the loop (loop-invariant). If the allocator
// doesn't extend its interval across the loop, the scale register gets
// reused and the kernel produces garbage.
struct ScaleParams {
    const float* src;
    float* dst;
    size_t count;    // number of vectors to process
    float scale;
};

template <size_t N>
struct jit_ir_scale_kernel : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_ir_scale_kernel)

    jit_ir_scale_kernel() : jit_kernel(jit_name()) {}

    using fn_t = void (*)(const ScaleParams*);
    fn_t fn_ = nullptr;

    void init() {
        if (create_kernel() != dnnl::impl::status::success)
            OPENVINO_THROW("Can't generate jit kernel");
        fn_ = (fn_t)(jit_ker());  // NOLINT
    }

    void operator()(const ScaleParams& args) const { fn_(&args); }

    void generate() override {
        using reg_type = typename reg_traits<float[N]>::type;
        preamble();

        auto src_ptr = arg(&ScaleParams::src);
        auto dst_ptr = arg(&ScaleParams::dst);
        auto count   = arg(&ScaleParams::count);

        begin_ir();

        // Broadcast scale — loop-invariant, defined before foreach.
        // Tests that the allocator extends scale's interval across the loop.
        auto scale_addr = argPtr(&ScaleParams::scale);
        auto scale = ir_def<N>({}, [this, scale_addr](const jit_kernel_ir::EmitContext& ctx) {
            uni_vbroadcastss(reg_type(ctx.def->idx), scale_addr);
        });

        auto src_reg_idx = static_cast<std::uint32_t>(src_ptr.reg().getIdx());
        auto dst_reg_idx = static_cast<std::uint32_t>(dst_ptr.reg().getIdx());

        foreach(size_t{0}, count, [&](const variable<size_t>& idx) {
            // Load *src_ptr (current vector)
            auto x = ir_load<N>(src_ptr);

            auto result = vmulps(x, scale);

            // Store to *dst_ptr
            ir_store<N>(dst_ptr, result);

            // Advance pointers — must be inside an IR use() so the add
            // instructions are emitted at lowering time, not recording time.
            ir_use({}, [this, src_reg_idx, dst_reg_idx](
                           const jit_kernel_ir::EmitContext&) {
                add(Xbyak::Reg64(src_reg_idx), N * sizeof(float));
                add(Xbyak::Reg64(dst_reg_idx), N * sizeof(float));
            });
        });

        end_ir();
        postamble();
    }
};

TEST(JitKernelIR, EndToEndForeach) {
    using namespace dnnl::impl::cpu::x64;

    if (mayiuse(cpu_isa_t::avx2)) {
        constexpr size_t N = 8;
        constexpr size_t num_vectors = 4;
        constexpr size_t total = N * num_vectors;

        jit_ir_scale_kernel<N> kernel;
        kernel.init();

        alignas(32) std::array<float, total> src{}, dst{};
        for (size_t i = 0; i < total; ++i) {
            src[i] = static_cast<float>(i + 1);
        }
        const float scale = 2.5f;
        ScaleParams args{src.data(), dst.data(), num_vectors, scale};
        kernel(args);

        for (size_t i = 0; i < total; ++i) {
            EXPECT_FLOAT_EQ(dst[i], src[i] * scale)
                << "mismatch at index " << i;
        }
    }
}

// ── Slice 3b: _if/_then/_else in IR mode ───────────────────────────────

// Kernel: if (flag == 0) dst = a + b; else dst = a - b;
struct IfElseParams {
    const float* a;
    const float* b;
    float* result;
    size_t flag;
};

template <size_t N>
struct jit_ir_if_else_kernel : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_ir_if_else_kernel)

    jit_ir_if_else_kernel() : jit_kernel(jit_name()) {}

    using fn_t = void (*)(const IfElseParams*);
    fn_t fn_ = nullptr;

    void init() {
        if (create_kernel() != dnnl::impl::status::success)
            OPENVINO_THROW("Can't generate jit kernel");
        fn_ = (fn_t)(jit_ker());  // NOLINT
    }

    void operator()(const IfElseParams& args) const { fn_(&args); }

    void generate() override {
        preamble();

        auto a_ptr = arg(&IfElseParams::a);
        auto b_ptr = arg(&IfElseParams::b);
        auto r_ptr = arg(&IfElseParams::result);
        auto flag  = arg(&IfElseParams::flag);

        begin_ir();

        auto a = ir_load<N>(a_ptr);
        auto b = ir_load<N>(b_ptr);

        // if (flag == 0) result = a + b; else result = a - b;
        ir_cmp(flag, size_t{0});
        ir_if(&Xbyak::CodeGenerator::jne,
            [&]() {
                // then: flag == 0 → store a + b
                auto sum = vaddps(a, b);
                ir_store<N>(r_ptr, sum);
            },
            [&]() {
                // else: flag != 0 → store a - b
                auto diff = vsubps(a, b);
                ir_store<N>(r_ptr, diff);
            });

        end_ir();
        postamble();
    }
};

TEST(JitKernelIR, EndToEndIfElse) {
    using namespace dnnl::impl::cpu::x64;

    if (mayiuse(cpu_isa_t::avx2)) {
        constexpr size_t N = 8;

        alignas(32) std::array<float, N> a{}, b{}, result{};
        for (size_t i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i + 1);
            b[i] = static_cast<float>(i) * 0.5f;
        }

        // Test flag == 0 → a + b
        {
            jit_ir_if_else_kernel<N> kernel;
            kernel.init();
            IfElseParams args{a.data(), b.data(), result.data(), 0};
            kernel(args);
            for (size_t i = 0; i < N; ++i) {
                EXPECT_FLOAT_EQ(result[i], a[i] + b[i])
                    << "then-branch mismatch at index " << i;
            }
        }

        // Test flag != 0 → a - b
        {
            jit_ir_if_else_kernel<N> kernel;
            kernel.init();
            IfElseParams args{a.data(), b.data(), result.data(), 1};
            kernel(args);
            for (size_t i = 0; i < N; ++i) {
                EXPECT_FLOAT_EQ(result[i], a[i] - b[i])
                    << "else-branch mismatch at index " << i;
            }
        }
    }
}
