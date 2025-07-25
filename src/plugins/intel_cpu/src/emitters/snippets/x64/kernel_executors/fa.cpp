// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fa.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

#include <common/utils.hpp>

#include "dnnl_extension_utils.h"
#include "emitters/utils.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/utils/utils.hpp"

#define DTYPE_CAST(X) static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(X))

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu::x64 {

FAKernelConfig::FAKernelConfig(const fa_utils::FAConfig& fa_config)
    : m_static_params(std::make_shared<StaticParams>(fa_config.src_dt(),
                                                     fa_config.wei_dt(),
                                                     fa_config.orig_wei_dt(),
                                                     fa_config.isa(),
                                                     fa_config.transposed_b(),
                                                     fa_config.q_len_blk(),
                                                     fa_config.kv_len_blk())),
      m_hash(compute_hash()) {}

void FAKernelConfig::update(dnnl_dim_t q_len, dnnl_dim_t kv_len, dnnl_dim_t head_size_1, dnnl_dim_t head_size_2) {
    if (snippets::utils::one_of(0, q_len, kv_len, head_size_1, head_size_2)) {
        m_q_seq_len = 0;
        m_kv_seq_len = 0;
        m_qk_head_size = 0;
        m_v_head_size = 0;
    } else {
        m_q_seq_len = q_len;
        m_kv_seq_len = kv_len;
        m_qk_head_size = head_size_1;
        m_v_head_size = head_size_2;
    }
    m_hash = compute_hash();
}

size_t FAKernelConfig::compute_hash() const {
    size_t seed = m_static_params->m_hash;
#define HASH(X) seed = hash_combine(seed, X)
    HASH(m_q_seq_len);
    HASH(m_kv_seq_len);
    HASH(m_qk_head_size);
    HASH(m_v_head_size);
#undef HASH
    return seed;
}

bool FAKernelConfig::operator==(const FAKernelConfig& rhs) const {
#define EQ(X) X == rhs.X
    return EQ(m_hash) && EQ(m_q_seq_len) && EQ(m_kv_seq_len) && EQ(m_qk_head_size) && EQ(m_v_head_size) &&
           (EQ(m_static_params.get()) || *m_static_params == *(rhs.m_static_params));
#undef EQ
}

[[nodiscard]] bool FAKernelConfig::is_completed() const {
    return !ov::intel_cpu::one_of(0, m_q_seq_len, m_kv_seq_len, m_qk_head_size, m_v_head_size) || is_empty();
}
[[nodiscard]] bool FAKernelConfig::is_empty() const {
    return ov::intel_cpu::everyone_is(0, m_q_seq_len, m_kv_seq_len, m_qk_head_size, m_v_head_size);
}

FAKernelConfig::StaticParams::StaticParams(const element::Type& src_type,
                                           const element::Type& wei_type,
                                           const element::Type& original_wei_type,
                                           dnnl::impl::cpu::x64::cpu_isa_t isa,
                                           bool is_transposed_B,
                                           dnnl_dim_t q_len_blk,
                                           dnnl_dim_t kv_len_blk)
    : m_src_dt(DTYPE_CAST(src_type)),
      m_wei_dt(DTYPE_CAST(wei_type)),
      m_original_wei_dt(DTYPE_CAST(original_wei_type)),
      m_isa(isa),
      m_is_transposed_B(is_transposed_B),
      m_q_len_blk(q_len_blk),
      m_kv_len_blk(kv_len_blk),
      m_hash(init_hash(m_src_dt,
                       m_wei_dt,
                       m_original_wei_dt,
                       m_isa,
                       m_is_transposed_B,
                       m_q_len_blk,
                       m_kv_len_blk)) {}

bool FAKernelConfig::StaticParams::operator==(const StaticParams& rhs) const {
#define EQ(X) X == rhs.X
    return EQ(m_hash) && EQ(m_src_dt) && EQ(m_wei_dt) && EQ(m_original_wei_dt) && EQ(m_isa) &&
           EQ(m_is_transposed_B) && EQ(m_q_len_blk) && EQ(m_kv_len_blk);
#undef EQ
}

size_t FAKernelConfig::StaticParams::init_hash(const dnnl_data_type_t& src_dt,
                                               const dnnl_data_type_t& wei_dt,
                                               const dnnl_data_type_t& original_wei_dt,
                                               dnnl::impl::cpu::x64::cpu_isa_t primitive_isa,
                                               bool is_transposed_B,
                                               dnnl_dim_t q_len_blk,
                                               dnnl_dim_t kv_len_blk) {
    size_t seed = 0;
#define HASH(X) seed = hash_combine(seed, X)
    HASH(src_dt);
    HASH(wei_dt);
    HASH(original_wei_dt);
    HASH(primitive_isa);
    HASH(is_transposed_B);
    HASH(q_len_blk);
    HASH(kv_len_blk);
#undef HASH
    return seed;
}

#ifdef SNIPPETS_DEBUG_CAPS
#    define PRINT(X) ss << #X << " = " << (X) << "\n"
std::string FAKernelConfig::to_string() const {
    std::stringstream ss;
    ss << m_static_params->to_string() << "\n";
    PRINT(m_hash);
    PRINT(m_q_seq_len);
    PRINT(m_kv_seq_len);
    PRINT(m_qk_head_size);
    PRINT(m_v_head_size);
    return ss.str();
}
std::string FAKernelConfig::StaticParams::to_string() const {
    std::stringstream ss;
    PRINT(m_src_dt);
    PRINT(m_wei_dt);
    PRINT(m_original_wei_dt);
    PRINT(m_isa);
    PRINT(m_is_transposed_B);
    PRINT(m_q_len_blk);
    PRINT(m_kv_len_blk);
    return ss.str();
}
#    undef PRINT
#endif

FAKernelExecutor::FAKernelExecutor(FAKernelConfig config)
    : snippets::KernelExecutor<FAKernelConfig, FACompiledKernel>(std::move(config)) {}

void FAKernelExecutor::update_kernel([[maybe_unused]] const FAKernelConfig& config,
                                          std::shared_ptr<FACompiledKernel>& kernel) const {
    if (kernel == nullptr) {
        // universal kernel could be used in any config and shape, as excuted peice by peice as binary call.
        // config is passed as binary call parameters.
        kernel = std::make_shared<FACompiledKernel>();
        dnnl_post_ops post_ops;
        create_brgemm_kernel(kernel->brgemm_qk_ukernel,
                             dnnl_data_type_t::dnnl_f32,
                             dnnl_data_type_t::dnnl_f32,
                             dnnl_data_type_t::dnnl_f32,
                             config.get_isa(),
                             config.get_q_len_blk(),              // M
                             config.get_kv_len_blk(),             // N
                             config.get_qk_head_size(),           // K
                             config.get_qk_head_size(),           // lda
                             config.get_kv_len_blk(),             // ldb
                             config.get_kv_len_blk(),             // ldc
                             0.0f,                                // beta
                             post_ops);
        create_brgemm_kernel(kernel->brgemm_sv_ukernel_init,
                             dnnl_data_type_t::dnnl_f32,
                             dnnl_data_type_t::dnnl_f32,
                             dnnl_data_type_t::dnnl_f32,
                             config.get_isa(),
                             config.get_q_len_blk(),               // M
                             config.get_v_head_size(),             // N
                             config.get_kv_len_blk(),              // K
                             config.get_kv_len_blk(),              // lda
                             config.get_v_head_size(),             // ldb
                             config.get_v_head_size(),             // ldc
                             0.0f,                                 // beta is 0
                             post_ops);
        create_brgemm_kernel(kernel->brgemm_sv_ukernel,
                             dnnl_data_type_t::dnnl_f32,
                             dnnl_data_type_t::dnnl_f32,
                             dnnl_data_type_t::dnnl_f32,
                             config.get_isa(),
                             config.get_q_len_blk(),               // M
                             config.get_v_head_size(),             // N
                             config.get_kv_len_blk(),              // K
                             config.get_kv_len_blk(),              // lda
                             config.get_v_head_size(),             // ldb
                             config.get_v_head_size(),             // ldc
                             1.0f,                                 // beta is 1
                             post_ops);
        // create online softmax jit kernel:
        auto jcp = jit_params_online_softmax();
        jcp.src_prc = ov::element::f32;
        jcp.dst_prc = ov::element::f32;
        jcp.with_calibration = true;
        if (mayiuse(cpu::x64::avx512_core)) {
            kernel->online_softmax_ukernel =
                std::make_shared<jit_uni_online_softmax_kernel_f32<cpu::x64::avx512_core>>(jcp);
        } else if (mayiuse(cpu::x64::avx2)) {
            kernel->online_softmax_ukernel =
                std::make_shared<jit_uni_online_softmax_kernel_f32<cpu::x64::avx2>>(jcp);
        }
        jcp.with_calibration = false;
        if (mayiuse(cpu::x64::avx512_core)) {
            kernel->online_softmax_ukernel_init =
                std::make_shared<jit_uni_online_softmax_kernel_f32<cpu::x64::avx512_core>>(jcp);
        } else if (mayiuse(cpu::x64::avx2)) {
            kernel->online_softmax_ukernel_init =
                std::make_shared<jit_uni_online_softmax_kernel_f32<cpu::x64::avx2>>(jcp);
        }
        if (kernel->online_softmax_ukernel) {
            kernel->online_softmax_ukernel->create_ker();
        }
        if (kernel->online_softmax_ukernel_init) {
            kernel->online_softmax_ukernel_init->create_ker();
        }

        // buffer allocation
        size_t qk_result_size = config.get_q_len_blk() * config.get_kv_len_blk();
        size_t coefficient_size = config.get_q_seq_len() * 4;  // max_old, max_new, denominator_old, denominator_new
        auto threads = parallel_get_max_threads();
        kernel->buffer->resize((qk_result_size + coefficient_size) * sizeof(float) * threads, 0);
    }
}

void FAKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                     const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                     FAKernelConfig& config) const {
    const auto& input_pds = expr->get_input_port_descriptors();
    const auto& output_pds = expr->get_output_port_descriptors();
    OV_CPU_JIT_EMITTER_ASSERT(input_pds.size() == 3 && output_pds.size() == 1,
                              "Invalid number of in/out port descriptors");

    const auto& in0_shape = snippets::utils::get_planar_vdims(input_pds[0]->get_shape(), input_pds[0]->get_layout());
    const auto& in1_shape = snippets::utils::get_planar_vdims(input_pds[1]->get_shape(), input_pds[1]->get_layout());
    const auto& in2_shape = snippets::utils::get_planar_vdims(input_pds[2]->get_shape(), input_pds[2]->get_layout());
    auto q_len = *++in0_shape.rbegin();
    auto qk_head_size = *in0_shape.rbegin();
    auto kv_len = *in1_shape.rbegin();
    auto v_head_size = *in2_shape.rbegin();
    config.update(q_len, kv_len, qk_head_size, v_head_size);
    // std::cout << "q_len:" << q_len << std::endl;
    // std::cout << "kv_len:" << kv_len << std::endl;
    // std::cout << "qk_head_size:" << qk_head_size << std::endl;
    // std::cout << "v_head_size:" << v_head_size << std::endl;
}

void FAKernelExecutor::execute(const FAKernelExecutor* executor, void* in0, void* in1, void* in2, void* out) {
    OV_CPU_JIT_EMITTER_ASSERT(executor, "has nullptr executor");
    float* q = static_cast<float*>(in0);
    float* k = static_cast<float*>(in1);
    float* v = static_cast<float*>(in2);
    float* out_f32 = static_cast<float*>(out);
    const auto& config = static_cast<const FAKernelConfig&>(executor->get_config());
    const auto& q_len = config.get_q_seq_len();
    const auto& kv_len = config.get_kv_seq_len();
    const auto& qk_head_size = config.get_qk_head_size();
    const auto& v_head_size = config.get_v_head_size();
    const auto& q_len_blk = config.get_q_len_blk();
    const auto& kv_len_blk = config.get_kv_len_blk();

    const auto& kernel = executor->get_kernel();
    const auto& qk_kernel = kernel->brgemm_qk_ukernel;
    const auto& sv_kernel = kernel->brgemm_sv_ukernel;
    const auto& sv_kernel_init = kernel->brgemm_sv_ukernel_init;
    const auto& online_softmax_kernel = kernel->online_softmax_ukernel;
    const auto& online_softmax_kernel_init = kernel->online_softmax_ukernel_init;

    size_t qk_result_size = q_len_blk * kv_len_blk;
    size_t coefficient_size = q_len * 4;
    float* buffer = reinterpret_cast<float*>(kernel->buffer->data());
    auto thread_idx = parallel_get_thread_num();
    float* qk_result = buffer + (qk_result_size + coefficient_size) * thread_idx;
    float* coeff_max_past = qk_result + (q_len_blk * kv_len_blk);
    float* coeff_max = coeff_max_past + q_len;
    float* coeff_denominator_past = coeff_max  + q_len;
    float* coeff_denominator = coeff_denominator_past  + q_len;

    size_t kv_block_num = ov::snippets::utils::div_up(kv_len, kv_len_blk);
    size_t q_block_num = ov::snippets::utils::div_up(q_len, q_len_blk);
    // std::cout << "kv_block_num:" << kv_block_num << std::endl;
    // std::cout << "q_block_num:" << q_block_num << std::endl;
    for (size_t i = 0; i < kv_block_num; i++) {
        size_t kv_start = i * kv_len_blk;
        // size_t kv_end = std::min(kv_start + kv_len_blk, static_cast<size_t>(kv_len));
        // size_t rt_kv_len_blk = kv_end - kv_start;  // no used as not support tails for now
        float* k_ptr = k + kv_start * qk_head_size;   // k is repacked
        float* v_ptr = v + kv_start * v_head_size;
        bool is_first_kv = (i == 0);
        for (size_t j = 0; j < q_block_num; j++) {
            size_t q_start = j * q_len_blk;
            // size_t q_end = std::min(q_start + q_len_blk, static_cast<size_t>(q_len));
            // size_t rt_q_len_blk = q_end - q_start;  // no used as not support tails for now
            float* q_ptr = q + q_start * qk_head_size;
            // q*k
            execute_brgemm_kernel(qk_kernel,
                                  q_ptr,
                                  k_ptr,
                                  qk_result,
                                  nullptr,           // args->scratch
                                  nullptr,           // post ops args
                                  false,             // compensation
                                  false);            // apply post_ops
            // softmax and on line calibration coeff
            const auto& max_past_q = coeff_max_past + q_start;
            const auto& max_q = coeff_max + q_start;
            const auto& denominator_past_q = coeff_denominator_past + q_start;
            const auto& denominator_q = coeff_denominator + q_start;

            float* out_ptr = out_f32 + q_start * v_head_size;
            jit_args_online_softmax args{};
            args.data = reinterpret_cast<void*>(qk_result);
            args.max_past = reinterpret_cast<void*>(max_past_q);
            args.denominator_past = reinterpret_cast<void*>(denominator_past_q);
            args.max = reinterpret_cast<void*>(max_q);
            args.denominator = reinterpret_cast<void*>(denominator_q);
            args.out = reinterpret_cast<void*>(out_ptr);
            args.work_amount_inner = kv_len_blk;
            args.work_amount_inner_head_size = v_head_size;
            args.work_amount_outer = q_len_blk;
            if (is_first_kv) {
                (*online_softmax_kernel_init)(&args);
            } else {
                (*online_softmax_kernel)(&args);
            }

            // sv
            if (is_first_kv) {
                execute_brgemm_kernel(sv_kernel_init,
                                      qk_result,
                                      v_ptr,
                                      out_ptr,
                                      nullptr,           // args->scratch
                                      nullptr,           // post ops args
                                      false,             // compensation
                                      false);            // apply post_ops
            } else {
                execute_brgemm_kernel(sv_kernel,
                                      qk_result,
                                      v_ptr,
                                      out_ptr,
                                      nullptr,           // args->scratch
                                      nullptr,           // post ops args
                                      false,             // compensation
                                      false);            // apply post_ops
            }
        }
    }
}

}  // namespace ov::intel_cpu::x64
