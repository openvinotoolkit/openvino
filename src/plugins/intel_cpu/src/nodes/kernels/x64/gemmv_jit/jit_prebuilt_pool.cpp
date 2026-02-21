#include "jit_prebuilt_pool.hpp"

#include <memory>
#include <vector>

#include "amx_prebuilt.hpp"
#include "jit_gemmv_amx_bf16.hpp"
#include "jit_gemmv_amx_int8.hpp"
#include "jit_gemmv_avx512_fp32.hpp"
#include "jit_gemmv_avx512_simple.hpp"
#include "jit_gemmv_avx512_vnni_s32.hpp"
#include "jit_minigemm_avx512_fp32.hpp"
#include "cpu/jit_utils/jit_utils.hpp"

namespace ov::intel_cpu::x64::gemmv_jit {

namespace {

using jit_generator_base = dnnl::impl::cpu::x64::jit_generator_t;

struct kernel_slot {
    kernel_kind kind;
    std::unique_ptr<jit_generator_base> jit;
    const void* fn = nullptr;
};

template <typename JitClass>
kernel_slot make_slot(kernel_kind kind) {
    auto jit = std::make_unique<JitClass>();
    const void* fn = reinterpret_cast<const void*>(jit->get());
    return kernel_slot{kind, std::move(jit), fn};
}

std::vector<kernel_slot>& slots() {
    static std::vector<kernel_slot> storage = [] {
        std::vector<kernel_slot> s;
        s.emplace_back(make_slot<jit_amx_gemmv_int8_t>(kernel_kind::amx_int8));
        s.emplace_back(make_slot<jit_amx_gemmv_bf16_t>(kernel_kind::amx_bf16));
        s.emplace_back(make_slot<JitGemmvAvx512VnniS32>(kernel_kind::vnni_int8));
        s.emplace_back(make_slot<jit_gemmv_avx512_fp32_kernel>(kernel_kind::avx512_fp32));
        s.emplace_back(make_slot<jit_gemmv_avx512_simple_kernel>(kernel_kind::avx512_simple));
        s.emplace_back(make_slot<JitMiniGemmAvx512Fp32>(kernel_kind::minigemm_avx512_fp32));
        return s;
    }();
    return storage;
}

struct pool_initializer {
    pool_initializer() { (void)slots(); }
};

static pool_initializer g_initializer{};

} // namespace

jit_prebuilt_pool::jit_fn jit_prebuilt_pool::get(kernel_kind kind) {
    for (const auto& slot : slots()) {
        if (slot.kind == kind) {
            return slot.fn;
        }
    }
    return nullptr;
}

void jit_prebuilt_pool::preload() {
    (void)slots();
}

} // namespace ov::intel_cpu::x64::gemmv_jit
