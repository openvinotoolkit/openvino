// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "emitters/plugin/x64/jit_eltwise_emitters.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/parameter.hpp"
#include "utils/rt_info/approximate_exp_attribute.hpp"

using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu {
namespace {

size_t simd_width(cpu_isa_t isa) {
    return isa == avx512_core ? 16U : (isa == avx2 ? 8U : 4U);
}

// Runs one jit_exp_emitter over a buffer, one vector at a time. The destination register can be
// pointed at the source register, which is the case the fast path has to survive: it reads vmm_src
// after it has begun writing its auxiliaries.
class jit_exp_test_kernel : public jit_generator_t {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_exp_test_kernel)

    jit_exp_test_kernel(bool approximate, cpu_isa_t isa, bool alias_dst_onto_src, bool with_node = true)
        : jit_generator_t(jit_name(), isa),
          m_approximate(approximate),
          m_with_node(with_node),
          m_isa(isa),
          m_dst_idx(alias_dst_onto_src ? 1U : 2U) {}

    void create() {
        ASSERT_EQ(create_kernel(), dnnl::impl::status::success);
        m_fn = reinterpret_cast<fn_t>(const_cast<uint8_t*>(jit_ker()));
    }

    std::vector<float> operator()(const std::vector<float>& src) const {
        std::vector<float> dst(src.size(), 0.F);
        for (size_t i = 0; i < src.size(); i += simd_width(m_isa)) {
            m_fn(src.data() + i, dst.data() + i);
        }
        return dst;
    }

private:
    using fn_t = void (*)(const float*, float*);

    void generate() override {
        if (m_with_node) {
            const auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
            const auto exp = std::make_shared<ov::op::v0::Exp>(param);
            if (m_approximate) {
                mark_as_approximate_exp(exp);
            }
            m_emitter = std::make_unique<jit_exp_emitter>(this, m_isa, exp, ov::element::f32);
        } else {
            m_emitter = std::make_unique<jit_exp_emitter>(this, m_isa, ov::element::f32);
        }

        preamble();
        load(1, ptr[abi_param1]);
        m_emitter->emit_code({1}, {m_dst_idx}, {3, 4, 5});
        store(ptr[abi_param2], m_dst_idx);
        postamble();
        m_emitter->emit_data();
    }

    void load(size_t idx, const Xbyak::Address& src) {
        if (m_isa == avx512_core) {
            uni_vmovups(Xbyak::Zmm(static_cast<int>(idx)), src);
        } else if (m_isa == avx2) {
            uni_vmovups(Xbyak::Ymm(static_cast<int>(idx)), src);
        } else {
            uni_vmovups(Xbyak::Xmm(static_cast<int>(idx)), src);
        }
    }

    void store(const Xbyak::Address& dst, size_t idx) {
        if (m_isa == avx512_core) {
            uni_vmovups(dst, Xbyak::Zmm(static_cast<int>(idx)));
        } else if (m_isa == avx2) {
            uni_vmovups(dst, Xbyak::Ymm(static_cast<int>(idx)));
        } else {
            uni_vmovups(dst, Xbyak::Xmm(static_cast<int>(idx)));
        }
    }

    bool m_approximate;
    bool m_with_node;
    cpu_isa_t m_isa;
    size_t m_dst_idx;
    std::unique_ptr<jit_exp_emitter> m_emitter;
    fn_t m_fn{nullptr};
};

cpu_isa_t widest_supported_isa() {
    if (mayiuse(avx512_core)) {
        return avx512_core;
    }
    if (mayiuse(avx2)) {
        return avx2;
    }
    // sse41 is the narrowest the emitter supports, and vroundps needs it.
    OPENVINO_ASSERT(mayiuse(sse41), "jit_exp_emitter requires at least sse41");
    return sse41;
}

// The interval a softmax numerator lives on after its row maximum has been subtracted, stopping
// just above the fast path's saturation knee at -100 * ln2 = -69.3147. Below that the two paths
// deliberately part company -- the accurate one keeps evaluating its polynomial and then flushes to
// zero, this one saturates -- and each behaviour is pinned by its own test rather than folded into
// an error bound. The length is a multiple of the widest vector so the same buffer drives every ISA.
std::vector<float> sweep() {
    std::vector<float> xs;
    xs.reserve(1 << 16);
    for (int i = 0; i < (1 << 16); ++i) {
        xs.push_back(-69.F + (69.F * static_cast<float>(i)) / static_cast<float>((1 << 16) - 1));
    }
    return xs;
}

struct Summary {
    double max_rel_err;
    size_t n_not_finite;  // inf or nan
    size_t n_denormal;    // subnormal or zero, i.e. anything the fast path claims never to emit
                          // (the accurate path does emit zeros, but not on this sweep)
    float min_out;
    float max_out;
};

Summary summarise(const std::vector<float>& xs, const std::vector<float>& ys) {
    Summary s{0.0, 0U, 0U, ys.front(), ys.front()};
    for (size_t i = 0; i < xs.size(); ++i) {
        const float y = ys[i];
        if (!std::isfinite(y)) {
            ++s.n_not_finite;
            continue;
        }
        if (std::fabs(y) < std::numeric_limits<float>::min()) {
            ++s.n_denormal;
        }
        s.min_out = std::min(s.min_out, y);
        s.max_out = std::max(s.max_out, y);
        const double exact = std::exp(static_cast<double>(xs[i]));
        s.max_rel_err = std::max(s.max_rel_err, std::fabs(static_cast<double>(y) - exact) / exact);
    }
    return s;
}

std::pair<double, double> signed_rel_err_range(const std::vector<float>& xs, const std::vector<float>& ys) {
    double lo = 0.0;
    double hi = 0.0;
    for (size_t i = 0; i < xs.size(); ++i) {
        const double exact = std::exp(static_cast<double>(xs[i]));
        const double e = (static_cast<double>(ys[i]) - exact) / exact;
        lo = std::min(lo, e);
        hi = std::max(hi, e);
    }
    return {lo, hi};
}

std::vector<float> run(bool approximate, cpu_isa_t isa, const std::vector<float>& xs, bool alias = false) {
    jit_exp_test_kernel kernel(approximate, isa, alias);
    kernel.create();
    return kernel(xs);
}

constexpr float FAST_EXP_C0 = 1.02901411F;       // the polynomial's constant term
constexpr float FAST_EXP_FLOOR = 8.117490e-31F;  // c0 * 2^-100
constexpr float FAST_EXP_CEIL = 1.7507768e38F;   // c0 * 2^127

}  // namespace

// Pins the whole trade the property advertises: the error it costs, and the fact that it buys that
// error without ever producing an infinity, a NaN or a denormal. A single wrong hex digit in any of
// the four fast-path constants moves one of these.
TEST(JitExpEmitter, approximate_path_matches_its_advertised_error_and_range) {
    const auto isa = widest_supported_isa();
    const auto xs = sweep();
    const auto ys = run(true, isa, xs);
    const auto s = summarise(xs, ys);

    EXPECT_LT(s.max_rel_err, 3.0e-2);
    EXPECT_GT(s.max_rel_err, 2.0e-2);  // it really is the degree-1 fit, not the accurate path
    EXPECT_EQ(s.n_not_finite, 0U);
    EXPECT_EQ(s.n_denormal, 0U);
    EXPECT_GE(s.min_out, FAST_EXP_FLOOR);
    EXPECT_FLOAT_EQ(s.max_out, FAST_EXP_C0);  // the sweep ends at x = 0
}

// The property advertises two error figures and the sweep above pins the one on the exponential.
// This pins the other, which is the one a caller sees: normalising divides one fast-path value by a
// sum of them, and because a minimax fit equioscillates about zero there is no common bias to
// cancel, so a probability carries more error than the exponential did rather than less. The worst
// case is (1 + hi) / (1 + lo) - 1, approached when one entry of a row carries the largest positive
// error and the row sum is dominated by entries carrying the largest negative one.
TEST(JitExpEmitter, approximate_path_error_on_a_normalised_probability_stays_within_its_bound) {
    const auto isa = widest_supported_isa();
    const auto xs = sweep();
    const auto [lo, hi] = signed_rel_err_range(xs, run(true, isa, xs));
    const double worst_ratio = (1.0 + hi) / (1.0 + lo) - 1.0;

    EXPECT_LT(worst_ratio, 6.2e-2);
    EXPECT_GT(worst_ratio, 5.9e-2);  // and it really is worse than the 2.98e-2 on the exponential
}

// The fast path reads its source register after it has begun writing its auxiliaries, so a caller
// that points the destination at the source is the interesting register case: reordering those two
// writes passes every other test here and fails this one.
TEST(JitExpEmitter, approximate_path_is_correct_when_the_destination_aliases_the_source) {
    const auto isa = widest_supported_isa();
    const auto xs = sweep();
    const auto separate = run(true, isa, xs, false);
    const auto aliased = run(true, isa, xs, true);

    EXPECT_EQ(aliased, separate);
    EXPECT_FLOAT_EQ(aliased.back(), FAST_EXP_C0);  // the sweep ends at x = 0, so this is the fast path
}

// The accurate path is what every other Exp in the plugin gets, so the same kernel has to keep
// producing it for an unmarked node.
TEST(JitExpEmitter, accurate_path_is_untouched_for_an_unmarked_exp) {
    const auto isa = widest_supported_isa();
    const auto xs = sweep();
    const auto s = summarise(xs, run(false, isa, xs));

    EXPECT_LT(s.max_rel_err, 1.0e-6);
    EXPECT_EQ(s.n_not_finite, 0U);
}

// The constructor that takes a node has to stay indistinguishable from the pre-existing one that
// takes none whenever the node is unmarked -- same amount of code, same bits out. A table mistake
// would be invisible here, since table_val resolves by name and both sides share one table; what
// this catches is the accurate path acquiring anything that depends on the node being present.
TEST(JitExpEmitter, unmarked_exp_is_bit_identical_to_the_constructor_that_takes_no_node) {
    const auto isa = widest_supported_isa();
    const auto xs = sweep();
    jit_exp_test_kernel with_node(false, isa, false, true);
    jit_exp_test_kernel without_node(false, isa, false, false);
    with_node.create();
    without_node.create();

    EXPECT_EQ(with_node.getSize(), without_node.getSize());
    EXPECT_EQ(with_node(xs), without_node(xs));
}

TEST(JitExpEmitter, approximate_path_saturates_instead_of_overflowing_or_underflowing) {
    const auto isa = widest_supported_isa();
    const size_t w = simd_width(isa);
    std::vector<float> xs(w, 0.F);
    xs[0] = 0.F;
    xs[1] = -1000.F;
    xs[2] = 1000.F;
    xs[3] = std::numeric_limits<float>::quiet_NaN();

    const auto ys = run(true, isa, xs);

    EXPECT_FLOAT_EQ(ys[0], FAST_EXP_C0);  // exp(0) is c0, not 1
    EXPECT_FLOAT_EQ(ys[1], FAST_EXP_FLOOR);
    EXPECT_FLOAT_EQ(ys[2], FAST_EXP_CEIL);
    EXPECT_FLOAT_EQ(ys[3], FAST_EXP_CEIL);  // vminps returns src2 on NaN, so a NaN saturates high
}

// The guard on where the floor sits, which register_table_entries explains: the quotient is what
// has to stay normal, not the floor, so this drives a masked-out entry through the emitter and
// normalises it by the largest row the table comment claims to cover. At 2^-126 every one of these
// would be a subnormal, and a masked attention row is made of nothing else.
TEST(JitExpEmitter, normalising_a_saturated_entry_still_yields_a_normal) {
    const auto isa = widest_supported_isa();
    const size_t w = simd_width(isa);
    std::vector<float> xs(w, 0.F);
    xs[0] = -1.0e4F;  // a masked-out attention entry

    const auto ys = run(true, isa, xs);

    ASSERT_FLOAT_EQ(ys[0], FAST_EXP_FLOOR);
    for (const size_t n : {w, size_t{1} << 10, size_t{1} << 25}) {
        const float smallest = ys[0] / (static_cast<float>(n) * FAST_EXP_C0);
        EXPECT_GE(smallest, std::numeric_limits<float>::min()) << "row of " << n;
    }
}

// sse41 has no FMA and no 256-bit integer shifts, so it is the ISA on which a wrongly chosen
// uni_ helper would show up first.
TEST(JitExpEmitter, approximate_path_is_the_same_function_on_sse41) {
    if (!mayiuse(sse41)) {
        GTEST_SKIP() << "sse41 not available";
    }
    const auto xs = sweep();
    const auto s = summarise(xs, run(true, sse41, xs));

    EXPECT_LT(s.max_rel_err, 3.0e-2);
    EXPECT_GT(s.max_rel_err, 2.0e-2);  // and it is the degree-1 fit, not the accurate path
    EXPECT_EQ(s.n_not_finite, 0U);
    EXPECT_EQ(s.n_denormal, 0U);
}

}  // namespace ov::intel_cpu
