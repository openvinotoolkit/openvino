// Copyright (C) 2020-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/x64/brgemm/brgemm.hpp>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "emitters/snippets/brgemm_generic.hpp"
#include "onednn/dnnl.h"

namespace ov::intel_cpu::x64 {

struct BrgemmBaseKernelConfig : public ov::intel_cpu::BrgemmGenericKernelConfig {
public:
    BrgemmBaseKernelConfig() = default;

    size_t hash() const override {
        return m_hash;
    }

    bool operator==(const BrgemmBaseKernelConfig& rhs) const;
    bool operator!=(const BrgemmBaseKernelConfig& rhs) const {
        return !(*this == rhs);
    }

    void update(int64_t M, int64_t N, int64_t K, int64_t LDA, int64_t LDB, int64_t LDC, float beta) override;

    dnnl_data_type_t get_dt_in0() const {
        return get_static_params()->dt_in0;
    }
    dnnl_data_type_t get_dt_in1() const {
        return get_static_params()->dt_in1;
    }
    dnnl_data_type_t get_dt_out() const {
        return get_static_params()->dt_out;
    }

    dnnl::impl::cpu::x64::cpu_isa_t get_isa() const {
        return get_static_params()->isa;
    }

    const dnnl_post_ops& get_post_ops() const {
        return get_static_params()->post_ops;
    }

#ifdef SNIPPETS_DEBUG_CAPS
    std::string to_string() const override;
#endif

protected:
    struct StaticBaseParams {
        StaticBaseParams(const element::Type& in0_dtype,
                         const element::Type& in1_dtype,
                         const element::Type& out_dtype,
                         dnnl::impl::cpu::x64::cpu_isa_t primitive_isa,
                         const dnnl_post_ops& post_ops,
                         size_t hash_seed);
        virtual ~StaticBaseParams() = default;

        const dnnl_data_type_t dt_in0{dnnl_f32}, dt_in1{dnnl_f32}, dt_out{dnnl_f32};
        const dnnl::impl::cpu::x64::cpu_isa_t isa{dnnl::impl::cpu::x64::isa_undef};
        const dnnl_post_ops post_ops;

        size_t hash() const {
            return m_hash;
        }

        bool operator==(const StaticBaseParams& rhs) const;
        bool operator!=(const StaticBaseParams& rhs) const {
            return !(*this == rhs);
        }
#ifdef SNIPPETS_DEBUG_CAPS
        std::string to_string() const;
#endif
    protected:
        static size_t compute_hash(size_t hash_seed,
                                   dnnl_data_type_t dt_in0,
                                   dnnl_data_type_t dt_in1,
                                   dnnl_data_type_t dt_out,
                                   dnnl::impl::cpu::x64::cpu_isa_t isa,
                                   const dnnl_post_ops& post_ops);

        const size_t m_hash{0};
    };

    virtual std::shared_ptr<StaticBaseParams> get_static_params() const = 0;
    size_t compute_hash() const;
    size_t m_hash{SIZE_MAX};
};

class BrgemmBaseKernelExecutor {
public:
    virtual ~BrgemmBaseKernelExecutor() = default;

protected:
    static void update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                              const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                              BrgemmBaseKernelConfig& config);

    static void create_brgemm_kernel(std::shared_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t>& kernel,
                                     dnnl_data_type_t dt_in0,
                                     dnnl_data_type_t dt_in1,
                                     dnnl_data_type_t dt_out,
                                     dnnl::impl::cpu::x64::cpu_isa_t isa,
                                     dnnl_dim_t M,
                                     dnnl_dim_t N,
                                     dnnl_dim_t K,
                                     dnnl_dim_t LDA,
                                     dnnl_dim_t LDB,
                                     dnnl_dim_t LDC,
                                     float beta,
                                     const dnnl_post_ops& post_ops,
                                     bool with_amx = false,
                                     char* palette = nullptr);

    static void execute_brgemm_kernel(const std::shared_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t>& kernel,
                                      const void* src,
                                      const void* wei,
                                      void* dst,
                                      void* scratch,
                                      const void* post_ops_binary_arg_vec,
                                      bool with_comp,
                                      bool apply_post_ops);
};

}  // namespace ov::intel_cpu::x64
