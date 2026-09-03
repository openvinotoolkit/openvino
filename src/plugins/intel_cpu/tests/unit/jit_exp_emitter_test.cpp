// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
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

// Runs one jit_exp_emitter over a buffer, one vector at a time. The destination register can be
// pointed at the source register, which is the case the fast path has to survive: it reads vmm_src
// after it has begun writing its auxiliaries.
class jit_exp_test_kernel : public jit_generator_t {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_exp_test_kernel)

    jit_exp_test_kernel(bool approximate, cpu_isa_t isa, bool alias_dst_onto_src)
        : jit_generator_t(jit_name(), isa),
          m_approximate(approximate),
          m_isa(isa),
          m_dst_idx(alias_dst_onto_src ? 1U : 2U) {}

    void create() {
        ASSERT_EQ(create_kernel(), dnnl::impl::status::success);
        m_fn = reinterpret_cast<fn_t>(const_cast<uint8_t*>(jit_ker()));
    }

    size_t simd_width() const {
        return m_isa == avx512_core ? 16U : (m_isa == avx2 ? 8U : 4U);
    }

    std::vector<float> operator()(const std::vector<float>& src) const {
        std::vector<float> dst(src.size(), 0.F);
        for (size_t i = 0; i < src.size(); i += simd_width()) {
            m_fn(src.data() + i, dst.data() + i);
        }
        return dst;
    }

private:
    using fn_t = void (*)(const float*, float*);

    void generate() override {
        const auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
        const auto exp = std::make_shared<ov::op::v0::Exp>(param);
        if (m_approximate) {
            mark_as_approximate_exp(exp);
        }
        m_emitter = std::make_unique<jit_exp_emitter>(this, m_isa, exp, ov::element::f32);

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
// just above ln(FLT_MIN) = -87.3365. Below that the two paths deliberately part company -- the
// accurate one flushes to zero, this one saturates -- and each behaviour is pinned by its own test
// rather than folded into an error bound. The length is a multiple of the widest vector so the
// same buffer drives every ISA.
std::vector<float> sweep() {
    std::vector<float> xs;
    xs.reserve(1 << 16);
    for (int i = 0; i < (1 << 16); ++i) {
        xs.push_back(-87.F + (87.F * static_cast<float>(i)) / static_cast<float>((1 << 16) - 1));
    }
    return xs;
}

struct Summary {
    double max_rel_err;
    size_t n_not_finite;  // inf or nan
    size_t n_denormal;    // subnormal or zero, i.e. anything the fast path claims never to emit
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
        if (y != 0.F && std::fabs(y) < 1.17549435e-38F) {
            ++s.n_denormal;
        }
        s.min_out = std::min(s.min_out, y);
        s.max_out = std::max(s.max_out, y);
        const double exact = std::exp(static_cast<double>(xs[i]));
        s.max_rel_err = std::max(s.max_rel_err, std::fabs(static_cast<double>(y) - exact) / exact);
    }
    return s;
}

std::vector<float> run(bool approximate, cpu_isa_t isa, const std::vector<float>& xs, bool alias = false) {
    jit_exp_test_kernel kernel(approximate, isa, alias);
    kernel.create();
    return kernel(xs);
}

constexpr float FAST_EXP_C0 = 1.02901411F;      // the polynomial's constant term
constexpr float FAST_EXP_FLOOR = 1.2096e-38F;   // c0 * 2^-126
constexpr float FAST_EXP_CEIL = 1.7507768e38F;  // c0 * 2^127

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

TEST(JitExpEmitter, approximate_path_saturates_instead_of_overflowing_or_underflowing) {
    const auto isa = widest_supported_isa();
    const size_t w = isa == avx512_core ? 16U : (isa == avx2 ? 8U : 4U);
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
