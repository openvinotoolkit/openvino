// Copyright (C) 2020-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type/element_type.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/snippets/cpu_kernel_executor_table.hpp"
#include <cpu/x64/brgemm/brgemm.hpp>

#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/loop_info.hpp"

namespace ov {
namespace intel_cpu {

struct BrgemmBaseKernelConfig : public snippets::KernelExecutorBase::GenericConfig {
public:
    BrgemmBaseKernelConfig() = default;

    bool is_completed() const override;
    size_t hash() const override { return m_hash; }

    bool is_empty() const;
    void update(dnnl_dim_t M, dnnl_dim_t N, dnnl_dim_t K, dnnl_dim_t LDA, dnnl_dim_t LDB, dnnl_dim_t LDC, float beta);

    bool operator==(const BrgemmBaseKernelConfig& rhs) const;
    bool operator!=(const BrgemmBaseKernelConfig& rhs) const {return !(*this == rhs);}

    dnnl_data_type_t get_dt_in0() const { return get_static_params()->dt_in0; }
    dnnl_data_type_t get_dt_in1() const { return get_static_params()->dt_in1; }

    dnnl::impl::cpu::x64::cpu_isa_t get_isa() const { return get_static_params()->isa; }
    float get_beta() const { return m_beta; }

    dnnl_dim_t get_M() const { return m_M; }
    dnnl_dim_t get_N() const { return m_N; }
    dnnl_dim_t get_K() const { return m_K; }

    dnnl_dim_t get_LDA() const { return m_LDA; }
    dnnl_dim_t get_LDB() const { return m_LDB; }
    dnnl_dim_t get_LDC() const { return m_LDC; }

#ifdef SNIPPETS_DEBUG_CAPS
    std::string to_string() const override;
#endif

protected:
    struct StaticBaseParams {
        StaticBaseParams(const element::Type& in0_dtype, const element::Type& in1_dtype, dnnl::impl::cpu::x64::cpu_isa_t primitive_isa, size_t hash_seed);
        virtual ~StaticBaseParams() = default;

        const dnnl_data_type_t dt_in0 {dnnl_f32}, dt_in1 {dnnl_f32};
        const dnnl::impl::cpu::x64::cpu_isa_t isa {dnnl::impl::cpu::x64::isa_undef};

        size_t hash() const { return m_hash; }

        bool operator==(const StaticBaseParams& rhs) const;
        bool operator!=(const StaticBaseParams& rhs) const { return !(*this == rhs); }
#ifdef SNIPPETS_DEBUG_CAPS
        std::string to_string() const;
#endif
    protected:
        static size_t compute_hash(size_t hash_seed, dnnl_data_type_t dt_in0, dnnl_data_type_t dt_in1, dnnl::impl::cpu::x64::cpu_isa_t isa);

        const size_t m_hash {0};
    };

    virtual std::shared_ptr<StaticBaseParams> get_static_params() const = 0;
    size_t compute_hash() const;

    dnnl_dim_t m_M {0}, m_N {0}, m_K {0}, m_LDA {0}, m_LDB {0}, m_LDC {0};
    float m_beta {0};
    size_t m_hash {SIZE_MAX};
};

class BrgemmBaseKernelExecutor {
public:
    virtual ~BrgemmBaseKernelExecutor() = default;
protected:
    static float get_beta(const ov::snippets::lowered::LoopManagerPtr& loop_manager, int loop_id,
                          const ov::snippets::lowered::ExpandedLoopInfoPtr& current_expanded_loop_info);

    static void update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                              const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                              BrgemmBaseKernelConfig& config);

    static void create_brgemm_kernel(std::shared_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t>& kernel, dnnl_data_type_t dt0, dnnl_data_type_t dt1,
                                     dnnl::impl::cpu::x64::cpu_isa_t isa, dnnl_dim_t M, dnnl_dim_t N, dnnl_dim_t K,
                                     dnnl_dim_t LDA, dnnl_dim_t LDB, dnnl_dim_t LDC, float beta, bool with_amx = false, char* palette = nullptr);

    static void execute_brgemm_kernel(const std::shared_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t>& kernel, const void* src, const void* wei,
                                      void* dst, void* scratch, bool with_comp);
};

}   // namespace intel_cpu
}   // namespace ov
