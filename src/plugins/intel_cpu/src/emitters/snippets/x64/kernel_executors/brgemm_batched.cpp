// Copyright (C) 2020-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_batched.hpp"

#include "common/utils.hpp"
#include "dnnl_extension_utils.h"
#include "snippets/lowered/pass/insert_specific_iterations.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"
#include "transformations/snippets/x64/op/gemm_cpu.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

#define DIM_CAST(X) static_cast<dnnl_dim_t>(X)

namespace ov::intel_cpu {

BrgemmBatchedKernelConfig::BrgemmBatchedKernelConfig(const element::Type& in0_dtype,
                                                     const element::Type& in1_dtype,
                                                     size_t iter_count,
                                                     bool is_with_comp,
                                                     dnnl::impl::cpu::x64::cpu_isa_t primitive_isa)
    : BrgemmBaseKernelConfig(),
      m_static_params(std::make_shared<StaticParams>(in0_dtype, in1_dtype, is_with_comp, primitive_isa)),
      m_iter_count(iter_count) {
    m_hash = compute_hash();
}

BrgemmBatchedKernelConfig::StaticParams::StaticParams(const element::Type& in0_dtype,
                                                      const element::Type& in1_dtype,
                                                      bool is_with_comp,
                                                      dnnl::impl::cpu::x64::cpu_isa_t primitive_isa)
    : StaticBaseParams(in0_dtype, in1_dtype, primitive_isa, compute_hash(is_with_comp)),
      is_with_comp(is_with_comp) {}

bool BrgemmBatchedKernelConfig::StaticParams::operator==(const StaticParams& rhs) const {
    return StaticBaseParams::operator==(rhs) && is_with_comp == rhs.is_with_comp;
}

size_t BrgemmBatchedKernelConfig::StaticParams::compute_hash(bool is_with_comp) {
    return hash_combine(0, is_with_comp);
}

#ifdef SNIPPETS_DEBUG_CAPS
std::string BrgemmBatchedKernelConfig::StaticParams::to_string() const {
    std::stringstream ss;
    ss << StaticBaseParams::to_string();
    ss << "is_with_comp = " << is_with_comp << "\n";
    return ss.str();
}
#endif

BrgemmBatchedKernelExecutor::BrgemmBatchedKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache,
                                                         BrgemmBatchedKernelConfig config)
    : CPUKernelExecutor<BrgemmBatchedKernelConfig, BrgemmBatchedCompiledKernel>(std::move(kernel_cache),
                                                                                std::move(config)) {}

std::shared_ptr<BrgemmBatchedCompiledKernel> BrgemmBatchedKernelExecutor::compile_kernel(
    const BrgemmBatchedKernelConfig& config) const {
    std::shared_ptr<BrgemmBatchedCompiledKernel> compiled_kernel = std::make_shared<BrgemmBatchedCompiledKernel>();

    // Brgemm is not executable - nothing to compile
    if (config.is_empty()) {
        return compiled_kernel;
    }

    cpu::x64::brgemm_desc_t desc;
    OV_CPU_JIT_EMITTER_ASSERT(brgemm_desc_init(&desc,
                                               config.get_isa(),
                                               cpu::x64::brgemm_addr,
                                               config.get_dt_in0(),
                                               config.get_dt_in1(),
                                               false,
                                               false,
                                               cpu::x64::brgemm_row_major,
                                               1.f,
                                               config.get_beta(),
                                               config.get_LDA(),
                                               config.get_LDB(),
                                               config.get_LDC(),
                                               config.get_M(),
                                               config.get_N(),
                                               config.get_K(),
                                               nullptr) == dnnl_success,
                              "Cannot initialize brgemm descriptor due to invalid params");

    cpu::x64::brgemm_kernel_t* kernel_ = nullptr;
    OV_CPU_JIT_EMITTER_ASSERT(brgemm_kernel_create(&kernel_, desc) == dnnl_success,
                              "Cannot create brgemm kernel due to invalid params");
    compiled_kernel->brgemm_kernel = std::unique_ptr<brgemm_kernel_t>(kernel_);

    return compiled_kernel;
}

void BrgemmBatchedKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                                const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                                BrgemmBatchedKernelConfig& config) const {
    const auto& input_pds = expr->get_input_port_descriptors();
    const auto& output_pds = expr->get_output_port_descriptors();
    OV_CPU_JIT_EMITTER_ASSERT((input_pds.size() == 2 || input_pds.size() == 3) && output_pds.size() == 1,
                              "Invalid number of in/out port descriptors");

    const auto in0_shape = snippets::utils::get_planar_vdims(input_pds[0]->get_shape(), input_pds[0]->get_layout());
    const auto in1_shape = snippets::utils::get_planar_vdims(input_pds[1]->get_shape(), input_pds[1]->get_layout());
    auto in0_subtensor = input_pds[0]->get_subtensor();
    OPENVINO_ASSERT(!in0_subtensor.empty(), "Incorrect in0 subtensor size");
    auto in1_subtensor = input_pds[1]->get_subtensor();
    OPENVINO_ASSERT(!in1_subtensor.empty(), "Incorrect in1 subtensor size");

    // Need to update M, K, N
    // 1. If the original value in subtensor is `FULL_DIM`, it means that
    //    Brgemm block should process full tensor by this dim -> take dimension from shape
    // 2. Otherwise, Brgemm block processes part of the tensor by this dim
    //    (there is blocking by this dimension) -> take from Loop increment

    auto M = *++in0_subtensor.rbegin();
    auto K = *in0_subtensor.rbegin();
    auto N = *in1_subtensor.rbegin();

    size_t loop_idx = 0;
    const auto& loop_ids = expr->get_loop_ids();
    const auto& loop_manager = linear_ir->get_loop_manager();
    auto get_loop_info = [&]() {
        OPENVINO_ASSERT(loop_idx < loop_ids.size(), "Loop is missed");
        return loop_manager->get_loop_info<ov::snippets::lowered::ExpandedLoopInfo>(loop_ids[loop_idx++]);
    };

    /* ------- Dimension M ----------*/
    if (ov::snippets::utils::is_full_dim_value(M)) {
        M = *++in0_shape.rbegin();
    } else {
        const auto& current_expanded_loop_info = get_loop_info();
        const auto& in_ports = current_expanded_loop_info->get_input_ports();
        const auto& out_ports = current_expanded_loop_info->get_output_ports();
        // Quick validation check: Should we check that port is really Brgemm port?
        // If BrgemmCopyB in the Loop by M -> first input port will be BrgemmCopyB with `incremented=false`
        // to avoid extra checks, we validate only first input port
        auto check_port = [&](const ov::snippets::lowered::LoopPort& p) {
            return p.get_dim_idx() == 1 && p.is_processed();
        };
        OPENVINO_ASSERT(
            in_ports.size() > 1 && check_port(in_ports[0]) && out_ports.size() == 1 && check_port(out_ports[0]),
            "Incorrect Loop by Brgemm dimension M");
        M = current_expanded_loop_info->get_work_amount() > 0 ? current_expanded_loop_info->get_increment() : 0;
        input_pds[0]->set_subtensor_dim(1, M);
        output_pds[0]->set_subtensor_dim(1, M);
    }

    /* ------- Dimension N ----------*/
    if (ov::snippets::utils::is_full_dim_value(N)) {
        N = *in1_shape.rbegin();
    } else {
        const auto& current_expanded_loop_info = get_loop_info();
        const auto& in_ports = current_expanded_loop_info->get_input_ports();
        const auto& out_ports = current_expanded_loop_info->get_output_ports();
        // Quick validation check: Should we check that port is really Brgemm port?
        auto check_port = [&](const ov::snippets::lowered::LoopPort& p) {
            return p.get_dim_idx() == 0 && p.is_processed();
        };
        OPENVINO_ASSERT(in_ports.size() >= 2 && !in_ports.front().is_processed() &&
                            std::all_of(in_ports.cbegin() + 1, in_ports.cend(), check_port) && out_ports.size() == 1 &&
                            check_port(out_ports.back()),
                        "Incorrect Loop by Brgemm dimension N");
        N = current_expanded_loop_info->get_work_amount() > 0 ? current_expanded_loop_info->get_increment() : 0;
        input_pds[1]->set_subtensor_dim(0, N);
        output_pds[0]->set_subtensor_dim(0, N);
    }

    /* ------- Dimension K ----------*/
    // 1. If Brgemm block processes full dimension K -> `beta = 0`
    // 2. If Brgemm block processes part of the dimension K (there is blocking), need to find
    //    the most first executed Brgemm Block in Loops which iterate through dimension K (work_amount > 0).
    //    First of them will have `beta = 0`, other - `beta = 1`
    float beta = 0;
    if (ov::snippets::utils::is_full_dim_value(K)) {
        K = *in0_shape.rbegin();
    } else {
        const auto& current_expanded_loop_info = get_loop_info();
        const auto& in_ports = current_expanded_loop_info->get_input_ports();
        const auto& out_ports = current_expanded_loop_info->get_output_ports();
        // Quick validation check: Should we check that port is really Brgemm port?
        OPENVINO_ASSERT(in_ports.size() >= 2 && in_ports.front().get_dim_idx() == 0 &&
                            in_ports.front().is_processed() && in_ports.back().get_dim_idx() == 1 &&
                            in_ports.back().is_processed() && out_ports.size() == 1 &&
                            !out_ports.front().is_processed(),
                        "Incorrect Loop by Brgemm dimension K");
        K = current_expanded_loop_info->get_work_amount() > 0 ? current_expanded_loop_info->get_increment() : 0;
        input_pds[0]->set_subtensor_dim(0, K);
        input_pds[1]->set_subtensor_dim(1, K);
        if (K > 0) {
            beta = BrgemmKernelExecutorHelper::get_beta(loop_manager,
                                                        static_cast<int>(loop_ids.back()),
                                                        current_expanded_loop_info);
        }
    }

    const auto LDA = DIM_CAST(snippets::utils::get_dim_stride(expr->get_input_port(0)));
    const auto LDC = DIM_CAST(snippets::utils::get_dim_stride(expr->get_output_port(0)));
    auto LDB = DIM_CAST(snippets::utils::get_dim_stride(expr->get_input_port(1)));

    const auto& brgemm_node = as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(brgemm_node, "has nullptr BrgemmCPU node");
    // In case of data repacking LDB is chosen in accordance with repacking buffer size
    if (with_repacking(brgemm_node->get_type())) {
        LDB = DIM_CAST(brgemm_utils::repacking::compute_repacked_n_dim(LDB, brgemm_node->get_input_element_type(1)));
    }

    config.update(DIM_CAST(M), DIM_CAST(N), DIM_CAST(K) / config.get_iter_count(), LDA, LDB, LDC, beta);
}

void BrgemmBatchedKernelExecutor::execute(const BrgemmBatchedKernelExecutor* executor, call_args* args) {
    OV_CPU_JIT_EMITTER_ASSERT(executor, "has nullptr executor");
    auto kernel = executor->get_kernel();
    const auto& config = static_cast<const BrgemmBatchedKernelConfig&>(executor->get_config());
    OV_CPU_JIT_EMITTER_ASSERT(kernel, "has nullptr compiler kernel or invalid config");

    // Note: compensations should be applied only once, so we do it only on the first iteration, when beta == 0
    const auto is_with_comp = config.get_beta() == 0 && config.is_with_comp();

    auto iter_count = config.get_iter_count();
    auto iter_size = config.get_K();

    size_t stride_A = iter_size * dnnl_data_type_size(config.get_dt_in0());
    size_t stride_B = (iter_size * config.get_LDB()) * dnnl_data_type_size(config.get_dt_in1());

    execute_brgemm(kernel->brgemm_kernel,
                   iter_count,
                   stride_A,
                   stride_B,
                   args->A,
                   args->B,
                   args->C,
                   args->scratch,
                   is_with_comp);
}

void BrgemmBatchedKernelExecutor::execute_brgemm(const std::shared_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t>& kernel,
                                                 size_t bs,
                                                 size_t stride_A,
                                                 size_t stride_B,
                                                 const void* pin0,
                                                 const void* pin1,
                                                 void* dst,
                                                 void* scratch,
                                                 bool with_comp) {
    cpu::x64::brgemm_kernel_params_t brgemm_p;
    std::vector<brgemm_batch_element_t> addr_batch(bs);

    for (size_t i = 0; i < bs; i++) {
        addr_batch[i].ptr.A = (char*)(pin0) + i * stride_A;
        addr_batch[i].ptr.B = (char*)(pin1) + i * stride_B;
    }
    brgemm_p.batch = addr_batch.data();
    brgemm_p.ptr_A = nullptr;
    brgemm_p.ptr_B = nullptr;
    brgemm_p.ptr_C = dst;
    brgemm_p.ptr_D = dst;
    brgemm_p.ptr_buf = scratch;
    brgemm_p.ptr_bias = nullptr;
    brgemm_p.do_post_ops = with_comp;
    brgemm_p.do_apply_comp = with_comp;
    brgemm_p.skip_accm = 0;
    brgemm_p.BS = bs;
    OV_CPU_JIT_EMITTER_ASSERT(kernel, "has nullptr Brgemm kernel");
    fprintf(stderr,
            "BrgemmBatchedKernelExecutor::execute_brgemm: bs=%zu, stride_A=%zu, stride_B=%zu, pin0=%p, pin1=%p, "
            "dst=%p, scratch=%p, with_comp=%d\n",
            bs,
            stride_A,
            stride_B,
            pin0,
            pin1,
            dst,
            scratch,
            with_comp);
    (*kernel)(&brgemm_p);
}

}  // namespace ov::intel_cpu
