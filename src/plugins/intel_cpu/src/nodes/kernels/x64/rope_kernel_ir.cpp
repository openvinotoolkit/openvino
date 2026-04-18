// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rope_kernel_ir.hpp"

#if defined(OPENVINO_ARCH_X86_64)

#include <cstddef>
#include <cstdint>

#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov::intel_cpu::kernel {

jit_rotary_kernel_ir::jit_rotary_kernel_ir(const jit_rotary_compile_params& jcp)
    : jit_kernel(jit_name()), m_jcp(jcp) {}

void jit_rotary_kernel_ir::init() {
    OPENVINO_ASSERT(create_kernel() == dnnl::impl::status::success,
                    "Can't generate IR-mode rotary kernel");
    _fn = reinterpret_cast<function_t>(const_cast<uint8_t*>(jit_ker()));

    // Dump generated code when ONEDNN_JIT_DUMP=1
    const char* dump_env = std::getenv("ONEDNN_JIT_DUMP");
    if (dump_env && std::string(dump_env) == "1") {
        auto code_size = getSize();
        const char* filename = "dnnl_dump_cpu_jit_rotary_kernel_ir.bin";
        if (auto* f = std::fopen(filename, "wb")) {
            std::fwrite(jit_ker(), 1, code_size, f);
            std::fclose(f);
            std::fprintf(stderr, "[ IR ] dump_jit_code: %s (%zu bytes)\n", filename, code_size);
        }
    }
}

// Dispatch to the right template instantiation based on compile params.
void jit_rotary_kernel_ir::generate() {
    this->preamble();

    // Determine vector width from ISA
    using namespace dnnl::impl::cpu::x64;
    constexpr size_t N_avx512 = 16;
    constexpr size_t N_avx2 = 8;

    auto dispatch_type = [this](auto type_tag, auto N_tag) {
        using T = decltype(type_tag);
        constexpr size_t N = decltype(N_tag)::value;
        if (m_jcp.interleave) {
            rotary_interleave_ir<T, N>();
        } else {
            rotary_half_ir<T, N>();
        }
    };

    auto dispatch_width = [&](auto type_tag) {
        if (mayiuse(avx512_core)) {
            dispatch_type(type_tag, std::integral_constant<size_t, N_avx512>{});
        } else {
            dispatch_type(type_tag, std::integral_constant<size_t, N_avx2>{});
        }
    };

    if (m_jcp.src_prc == ov::element::f32) {
        dispatch_width(float{});
    } else if (m_jcp.src_prc == ov::element::f16) {
        dispatch_width(ov::float16{});
    } else if (m_jcp.src_prc == ov::element::bf16) {
        dispatch_width(ov::bfloat16{});
    } else {
        OPENVINO_THROW("jit_rotary_kernel_ir: unsupported precision ", m_jcp.src_prc);
    }

    this->postamble();
}

template <typename T, size_t N>
void jit_rotary_kernel_ir::rotary_half_ir() {
    const auto half_rotary_ndims = m_jcp.rotary_ndims / 2;
    const auto half_byte_offset = half_rotary_ndims * sizeof(T);
    const bool shift_cos_sin = (m_jcp.cos_sin_ndims != half_rotary_ndims);

    auto loop_count = var<size_t>(half_rotary_ndims / N);

    begin_ir();

    auto src = make_ir_ptr(arg<T*>(&Params::src), N);
    auto cos = make_ir_ptr(arg<const float*>(&Params::cos), N);
    auto sin = make_ir_ptr(arg<const float*>(&Params::sin), N);
    auto dst = make_ir_ptr(arg<T*>(&Params::dst), N);

    // @todo claude: foreach_with_epilogue generates an epilogue even when
    // half_rotary_ndims is a multiple of N (tail never executes). The
    // epilogue's ir_load_partial creates ~20 GPR values that exhaust the
    // pool. Use plain foreach when no tail is needed.
    if (half_rotary_ndims % N != 0) {
        OPENVINO_THROW("rotary_half_ir: half_rotary_ndims must be a multiple of N");
    }

    foreach(size_t{0}, loop_count, [&](const variable<size_t>&) {
        auto v_src0 = ir_load<N>(src);
        auto v_src1 = ir_load<N>(src.ptr, half_byte_offset);
        auto v_cos = ir_load<N>(cos);
        auto v_sin = ir_load<N>(sin);

        auto v_dst0 = fmsub(v_cos, v_src0, v_sin * v_src1);
        ir_store(dst.ptr, size_t{0}, v_dst0);

        if (shift_cos_sin) {
            v_cos = ir_load<N>(cos.ptr, half_rotary_ndims * sizeof(float));
            v_sin = ir_load<N>(sin.ptr, half_rotary_ndims * sizeof(float));
        }

        auto v_dst1 = fma(v_sin, v_src0, v_cos * v_src1);
        ir_store(dst.ptr, half_byte_offset, v_dst1);

        ir_advance(src);
        ir_advance(cos);
        ir_advance(sin);
        ir_advance(dst);
    });

    end_ir();
}

template <typename T, size_t N>
void jit_rotary_kernel_ir::rotary_interleave_ir() {
    const auto half_rotary_ndims = m_jcp.rotary_ndims / 2;
    constexpr size_t src_step = 2 * N * sizeof(T);
    const size_t cos_step = m_jcp.mix_cos_sin ? 2 * N * sizeof(float) : N * sizeof(float);

    begin_ir();

    auto src = arg<T*>(&Params::src);
    auto cos = arg<const float*>(&Params::cos);
    auto sin = arg<const float*>(&Params::sin);
    auto dst = arg<T*>(&Params::dst);

    for (size_t i = 0; i < half_rotary_ndims / N; i++) {
        auto v_src0 = ir_load<N>(src);
        auto v_src1 = ir_load<N>(src, N * sizeof(T));

        auto [v_even, v_odd] = deinterleave2(v_src0, v_src1);

        auto v_cos = ir_load<N>(cos);
        variable<float[N]> v_sin(*this, jit_kernel_ir::invalid_value);
        if (m_jcp.mix_cos_sin) {
            auto v_sin_raw = ir_load<N>(cos, N * sizeof(float));
            auto [v_cos_d, v_sin_d] = deinterleave2(v_cos, v_sin_raw);
            v_cos = std::move(v_cos_d);
            v_sin = std::move(v_sin_d);
        } else {
            v_sin = ir_load<N>(sin);
        }

        auto v_dst0 = fmsub(v_cos, v_even, v_sin * v_odd);
        auto v_dst1 = fma(v_sin, v_even, v_cos * v_odd);

        auto [v_out0, v_out1] = interleave2(v_dst0, v_dst1);

        ir_store(dst, size_t{0}, v_out0);
        ir_store(dst, N * sizeof(T), v_out1);

        if (i + 1 < half_rotary_ndims / N) {
            ir_advance(src, src_step);
            ir_advance(dst, src_step);
            ir_advance(cos, cos_step);
            if (!m_jcp.mix_cos_sin) {
                ir_advance(sin, cos_step);
            }
        }
    }

    end_ir();
}

}  // namespace ov::intel_cpu::kernel

#endif  // OPENVINO_ARCH_X86_64
