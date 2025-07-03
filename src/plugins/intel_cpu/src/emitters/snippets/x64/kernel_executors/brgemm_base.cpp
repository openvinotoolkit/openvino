// Copyright (C) 2020-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_base.hpp"

#include <oneapi/dnnl/dnnl_common_types.h>
#include <oneapi/dnnl/dnnl_types.h>

#include <common/primitive_attr.hpp>
#include <cpu/x64/brgemm/brgemm.hpp>
#include <cpu/x64/brgemm/brgemm_types.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <sstream>
#include <string>

#include "common/primitive_hashing_utils.hpp"
#include "common/utils.hpp"
#include "dnnl_extension_utils.h"
#include "emitters/snippets/brgemm_generic.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

#define DTYPE_CAST(X) static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(X))
#define PRINT(X)      ss << #X << " = " << (X) << "\n"
#define EQ(X)         X == rhs.X
#define HASH(X)       seed = hash_combine(seed, X)

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu::x64 {

bool BrgemmBaseKernelConfig::operator==(const BrgemmBaseKernelConfig& rhs) const {
    return BrgemmGenericKernelConfig::operator==(rhs) &&
           (EQ(get_static_params()) || *get_static_params() == *(rhs.get_static_params()));
}

size_t BrgemmBaseKernelConfig::compute_hash() const {
    size_t seed = get_static_params()->hash();
    HASH(BrgemmGenericKernelConfig::compute_hash());
    return seed;
}

void BrgemmBaseKernelConfig::update(int64_t M,
                                    int64_t N,
                                    int64_t K,
                                    int64_t LDA,
                                    int64_t LDB,
                                    int64_t LDC,
                                    float beta) {
    BrgemmGenericKernelConfig::update(M, N, K, LDA, LDB, LDC, beta);
    m_hash = compute_hash();
}

BrgemmBaseKernelConfig::StaticBaseParams::StaticBaseParams(const element::Type& in0_dtype,
                                                           const element::Type& in1_dtype,
                                                           const element::Type& out_dtype,
                                                           cpu_isa_t primitive_isa,
                                                           const dnnl_post_ops& post_ops,
                                                           size_t hash_seed)
    : dt_in0(DTYPE_CAST(in0_dtype)),
      dt_in1(DTYPE_CAST(in1_dtype)),
      dt_out(DTYPE_CAST(out_dtype)),
      isa(primitive_isa),
      post_ops(post_ops),
      m_hash(compute_hash(hash_seed, dt_in0, dt_in1, dt_out, isa, post_ops)) {}

bool BrgemmBaseKernelConfig::StaticBaseParams::operator==(const StaticBaseParams& rhs) const {
    return EQ(hash()) && EQ(dt_in0) && EQ(dt_in1) && EQ(dt_out) && EQ(isa) && EQ(post_ops);
}

size_t BrgemmBaseKernelConfig::StaticBaseParams::compute_hash(size_t hash_seed,
                                                              dnnl_data_type_t dt_in0,
                                                              dnnl_data_type_t dt_in1,
                                                              dnnl_data_type_t dt_out,
                                                              cpu_isa_t isa,
                                                              const dnnl_post_ops& post_ops) {
    size_t seed = hash_seed;
    HASH(dt_in0);
    HASH(dt_in1);
    HASH(dt_out);
    HASH(isa);
    dnnl::impl::primitive_hashing::get_post_op_hash(seed, post_ops);
    return seed;
}

#ifdef SNIPPETS_DEBUG_CAPS
std::string BrgemmBaseKernelConfig::StaticBaseParams::to_string() const {
    std::stringstream ss;
    PRINT(dt_in0);
    PRINT(dt_in1);
    PRINT(dt_out);
    PRINT(isa);
    PRINT(post_ops.len());
    return ss.str();
}

std::string BrgemmBaseKernelConfig::to_string() const {
    std::stringstream ss;
    ss << get_static_params()->to_string() << "\n";
    ss << BrgemmGenericKernelConfig::to_string() << "\n";
    return ss.str();
}
#endif

void BrgemmBaseKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                             const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                             BrgemmBaseKernelConfig& config) {
    // update M/N/K/beta
    auto [M, N, K, beta] = BrgemmKernelExecutorHelper::get_runtime_brgemm_params(expr, linear_ir);

    const auto LDA = snippets::utils::get_dim_stride(expr->get_input_port(0));
    const auto LDC = snippets::utils::get_dim_stride(expr->get_output_port(0));
    auto LDB = snippets::utils::get_dim_stride(expr->get_input_port(1));

    const auto& brgemm_node = as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(brgemm_node, "Got invalid node type in update_config");
    // In case of data repacking LDB is chosen in accordance with repacking buffer size
    const auto& brgemm_config = brgemm_node->get_config();
    if (brgemm_config.with_wei_repacking()) {
        LDB = brgemm_utils::repacking::compute_K_blocked_stride(LDB,
                                                                brgemm_config.wei_n_blk(),
                                                                brgemm_config.are_wei_blocked());
    }

    config.update(M, N, K, LDA, LDB, LDC, beta);
}

void BrgemmBaseKernelExecutor::create_brgemm_kernel(std::shared_ptr<brgemm_kernel_t>& kernel,
                                                    dnnl_data_type_t dt_in0,
                                                    dnnl_data_type_t dt_in1,
                                                    dnnl_data_type_t dt_out,
                                                    cpu_isa_t isa,
                                                    dnnl_dim_t M,
                                                    dnnl_dim_t N,
                                                    dnnl_dim_t K,
                                                    dnnl_dim_t LDA,
                                                    dnnl_dim_t LDB,
                                                    dnnl_dim_t LDC,
                                                    float beta,
                                                    const dnnl_post_ops& post_ops,
                                                    bool with_amx,
                                                    char* palette) {
    cpu::x64::brgemm_desc_t desc;
    OV_CPU_JIT_EMITTER_ASSERT(brgemm_desc_init(&desc,
                                               isa,
                                               cpu::x64::brgemm_strd,
                                               dt_in0,
                                               dt_in1,
                                               false,
                                               false,
                                               cpu::x64::brgemm_row_major,
                                               1.F,
                                               beta,
                                               LDA,
                                               LDB,
                                               LDC,
                                               M,
                                               N,
                                               K,
                                               nullptr) == dnnl_success,
                              "Cannot initialize brgemm descriptor due to invalid params");

    dnnl_primitive_attr attr;
    attr.set_post_ops(post_ops);
    dnnl::memory::desc dst_desc({M, N}, static_cast<dnnl::memory::data_type>(dt_out), dnnl::memory::format_tag::ab);
    OV_CPU_JIT_EMITTER_ASSERT(brgemm_desc_set_postops(&desc, &attr, dst_desc.get(), LDC) == dnnl_success,
                              "Cannot set postops to brgemm descriptor");

    if (with_amx) {
        OV_CPU_JIT_EMITTER_ASSERT(palette && brgemm_init_tiles(desc, palette) == dnnl_success,
                                  "Cannot initialize brgemm tiles due to invalid params");
    }

    cpu::x64::brgemm_kernel_t* kernel_ = nullptr;
    OV_CPU_JIT_EMITTER_ASSERT(brgemm_kernel_create(&kernel_, desc) == dnnl_success,
                              "Cannot create brgemm kernel due to invalid params");
    kernel = std::unique_ptr<brgemm_kernel_t>(kernel_);
}

void BrgemmBaseKernelExecutor::execute_brgemm_kernel(
    const std::shared_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t>& kernel,
    const void* src,
    const void* wei,
    void* dst,
    void* scratch,
    const void* post_ops_binary_arg_vec,
    bool with_comp,
    bool apply_post_ops) {
    cpu::x64::brgemm_kernel_params_t brgemm_p;
    brgemm_p.batch = nullptr;  // default value
    brgemm_p.ptr_A = src;
    brgemm_p.ptr_B = wei;
    brgemm_p.ptr_C = dst;
    brgemm_p.ptr_D = dst;
    brgemm_p.ptr_buf = scratch;
    brgemm_p.ptr_bias = nullptr;
    brgemm_p.do_post_ops = static_cast<size_t>(apply_post_ops);
    brgemm_p.do_apply_comp = static_cast<size_t>(with_comp);
    brgemm_p.skip_accm = 0;
    brgemm_p.BS = 1;  // default value
    brgemm_p.post_ops_binary_rhs_arg_vec = post_ops_binary_arg_vec;
    // This ptr must be initialized if binary postops are applied
    brgemm_p.data_C_ptr_ = reinterpret_cast<char*>(dst);

    OV_CPU_JIT_EMITTER_ASSERT(kernel, "has nullptr Brgemm kernel");
    (*kernel)(&brgemm_p);
}

#undef DTYPE_CAST
#undef PRINT
#undef EQ
#undef HASH

}  // namespace ov::intel_cpu::x64
