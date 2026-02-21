#pragma once

#include <cstddef>

namespace ov::intel_cpu::x64::gemmv_jit {

enum class kernel_kind {
    amx_int8,
    amx_bf16,
    vnni_int8,
    avx512_fp32,
    avx512_simple,
    minigemm_avx512_fp32
};

class jit_prebuilt_pool {
public:
    using jit_fn = const void*;

    static jit_fn get(kernel_kind kind);
    static void preload(); // force all kernels to build at module load

    template <typename Fn>
    static Fn get_typed(kernel_kind kind) {
        return reinterpret_cast<Fn>(get(kind));
    }
};

} // namespace ov::intel_cpu::x64::gemmv_jit
