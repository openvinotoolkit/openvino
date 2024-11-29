// Copyright (C) 2020-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_base.hpp"

#include "common/utils.hpp"
#include "dnnl_extension_utils.h"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

#define DIM_CAST(X) static_cast<dnnl_dim_t>(X)
#define DTYPE_CAST(X) static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(X))
#define PRINT(X) ss << #X  << " = " << X << "\n"
#define EQ(X) X == rhs.X
#define HASH(X) seed = hash_combine(seed, X)

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {

bool BrgemmBaseKernelConfig::is_completed() const {
    return !utils::one_of(0, m_M, m_N, m_K, m_LDA, m_LDB, m_LDC) || is_empty();
}

bool BrgemmBaseKernelConfig::is_empty() const {
    return everyone_is(0, m_M, m_N, m_K, m_LDA, m_LDB, m_LDC, m_beta);
}

bool BrgemmBaseKernelConfig::operator==(const BrgemmBaseKernelConfig& rhs) const {
    return EQ(m_hash) && EQ(m_beta) &&
           EQ(m_M) && EQ(m_N) && EQ(m_K) &&
           EQ(m_LDA) && EQ(m_LDB) && EQ(m_LDC) &&
           (EQ(get_static_params()) || *get_static_params() == *(rhs.get_static_params()));
}

void BrgemmBaseKernelConfig::update(dnnl_dim_t M, dnnl_dim_t N, dnnl_dim_t K, dnnl_dim_t LDA, dnnl_dim_t LDB, dnnl_dim_t LDC, float beta) {
    // If M is zero, it means that Brgemm won't be executed (in Loop with work_amount = 0, for example)
    // To process this case, we have to make this Config as empty (nullify runtime parameters)
    if (utils::one_of(0, M, N, K)) {
        m_M = 0; m_N = 0; m_K = 0;
        m_LDA = 0; m_LDB = 0; m_LDC = 0;
        m_beta = 0;
    } else {
        m_M = M; m_N = N; m_K = K;
        m_LDA = LDA; m_LDB = LDB; m_LDC = LDC;
        m_beta = beta;
    }
    m_hash = compute_hash();
}

size_t BrgemmBaseKernelConfig::compute_hash() const {
    size_t seed = get_static_params()->hash();
    HASH(m_M); HASH(m_N); HASH(m_K);
    HASH(m_LDA); HASH(m_LDB); HASH(m_LDC);
    HASH(m_beta);
    return seed;
}

BrgemmBaseKernelConfig::StaticBaseParams::StaticBaseParams(const element::Type& in0_dtype, const element::Type& in1_dtype,
                                                           cpu_isa_t primitive_isa, size_t hash_seed)
    : dt_in0(DTYPE_CAST(in0_dtype)), dt_in1(DTYPE_CAST(in1_dtype)), isa(primitive_isa), m_hash(compute_hash(hash_seed, dt_in0, dt_in1, isa)) {}

bool BrgemmBaseKernelConfig::StaticBaseParams::operator==(const StaticBaseParams& rhs) const {
    return EQ(hash()) && EQ(dt_in0) && EQ(dt_in1) && EQ(isa);
}

size_t BrgemmBaseKernelConfig::StaticBaseParams::compute_hash(size_t hash_seed, dnnl_data_type_t dt_in0, dnnl_data_type_t dt_in1, cpu_isa_t isa) {
    size_t seed = hash_seed;
    HASH(dt_in0); HASH(dt_in1); HASH(isa);
    return seed;
}

#ifdef SNIPPETS_DEBUG_CAPS
std::string BrgemmBaseKernelConfig::StaticBaseParams::to_string() const {
    std::stringstream ss;
    PRINT(dt_in0); PRINT(dt_in1);
    PRINT(isa);
    return ss.str();
}

std::string BrgemmBaseKernelConfig::to_string() const {
    std::stringstream ss;
    ss << get_static_params()->to_string() << "\n";
    PRINT(m_M); PRINT(m_N); PRINT(m_K);
    PRINT(m_LDA); PRINT(m_LDB); PRINT(m_LDC);
    PRINT(m_beta);
    return ss.str();
}
#endif

float BrgemmBaseKernelExecutor::get_beta(const ov::snippets::lowered::LoopManagerPtr& loop_manager, int loop_id,
                                         const ov::snippets::lowered::ExpandedLoopInfoPtr& current_expanded_loop_info) {
    // Find all Expanded loops with the same Unified loop information -> they were decomposed from this Unified Loop.
    // Note that LoopInfo are normalized and sorted (due to NormalizedLoopIDs pass).
    // It means that previous executed Loops have Loop ID less the current Loop ID.
    // - If there is executed Loop (work_amount > 0) and evaluated before the current -> the current Brgemm should have `beta = 1`.
    // - If there is not this Loop -> the current executed Brgemm should have `beta = 0`.
    if (loop_id > 0) {
        const auto& current_unified_loop_info = current_expanded_loop_info->get_unified_loop_info();
        // Check the previous Loops
        --loop_id;
        while (loop_id >= 0) {
            const auto& expanded_loop_info = loop_manager->get_loop_info<ov::snippets::lowered::ExpandedLoopInfo>(loop_id);
            if (expanded_loop_info->get_unified_loop_info() != current_unified_loop_info)
                return 0;
            if (expanded_loop_info->get_work_amount() > 0) {
                // there is previous executed Brgemm with `beta = 0` -> the current Brgemm should have `beta = 1`
                return 1;
            }
            --loop_id;
        }
    }
    return 0;
}

void BrgemmBaseKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                             const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                             BrgemmBaseKernelConfig& config) {
    const auto& input_pds = expr->get_input_port_descriptors();
    const auto& output_pds = expr->get_output_port_descriptors();
    OV_CPU_JIT_EMITTER_ASSERT((input_pds.size() == 2 || input_pds.size() == 3) && output_pds.size() == 1,
                              "Invalid number of in/out port descriptors");

    const auto in0_shape = snippets::utils::get_planar_vdims(input_pds[0]->get_shape(), input_pds[0]->get_layout());
    const auto in1_shape = snippets::utils::get_planar_vdims(input_pds[1]->get_shape(), input_pds[1]->get_layout());
    auto in0_subtensor = input_pds[0]->get_subtensor();
    auto in1_subtensor = input_pds[1]->get_subtensor();

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
    auto get_loop_info = [&](){
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
        // Note: We check `is_incremented` attribute only for not incremented ports because
        //       this `is_incremented = true` can be changed by `CleanRepeatedDataPointerShifts` optimization
        auto check_port = [&](const ov::snippets::lowered::LoopPort& p) { return p.dim_idx == 1; };
        OPENVINO_ASSERT(in_ports.size() > 1 && std::all_of(in_ports.cbegin(), in_ports.cend(), check_port) &&
                        out_ports.size() == 1 && check_port(out_ports.back()),
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
        // Note: We check `is_incremented` attribute only for not incremented ports because
        //       this `is_incremented = true` can be changed by `CleanRepeatedDataPointerShifts` optimization
        auto check_port = [&](const ov::snippets::lowered::LoopPort& p) { return p.dim_idx == 0; };
        OPENVINO_ASSERT(in_ports.size() >= 2 && !in_ports.front().is_incremented && std::all_of(in_ports.cbegin(), in_ports.cend(), check_port) &&
                        out_ports.size() == 1 && check_port(out_ports.back()),
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
        // Note: We check `is_incremented` attribute only for not incremented ports because
        //       this `is_incremented = true` can be changed by `CleanRepeatedDataPointerShifts` optimization
        OPENVINO_ASSERT(in_ports.size() >= 2 && in_ports.front().dim_idx == 0 && in_ports.back().dim_idx == 1 &&
                        out_ports.size() == 1 && !out_ports.front().is_incremented,
                        "Incorrect Loop by Brgemm dimension K");
        K = current_expanded_loop_info->get_work_amount() > 0 ? current_expanded_loop_info->get_increment() : 0;
        input_pds[0]->set_subtensor_dim(0, K);
        input_pds[1]->set_subtensor_dim(1, K);
        if (K > 0)
            beta = get_beta(loop_manager, static_cast<int>(loop_ids.back()), current_expanded_loop_info);
    }

    const auto LDA = DIM_CAST(snippets::utils::get_dim_stride(expr->get_input_port(0)));
    const auto LDC = DIM_CAST(snippets::utils::get_dim_stride(expr->get_output_port(0)));
    auto LDB = DIM_CAST(snippets::utils::get_dim_stride(expr->get_input_port(1)));

    const auto& brgemm_node = as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(brgemm_node, "Got invalid node type in update_config");
    // In case of data repacking LDB is chosen in accordance with repacking buffer size
    if (with_repacking(brgemm_node->get_type()))
        LDB = DIM_CAST(brgemm_utils::repacking::compute_LDB(LDB, brgemm_node->get_input_element_type(1)));

    config.update(DIM_CAST(M), DIM_CAST(N), DIM_CAST(K), LDA, LDB, LDC, beta);
}

void BrgemmBaseKernelExecutor::create_brgemm_kernel(std::shared_ptr<brgemm_kernel_t>& kernel, dnnl_data_type_t dt0, dnnl_data_type_t dt1,
                                                    cpu_isa_t isa, dnnl_dim_t M, dnnl_dim_t N, dnnl_dim_t K,
                                                    dnnl_dim_t LDA, dnnl_dim_t LDB, dnnl_dim_t LDC, float beta, bool with_amx, char* palette) {
    cpu::x64::brgemm_desc_t desc;
    OV_CPU_JIT_EMITTER_ASSERT(brgemm_desc_init(&desc, isa, cpu::x64::brgemm_strd, dt0, dt1,
                                               false, false, cpu::x64::brgemm_row_major, 1.f,
                                               beta, LDA, LDB, LDC, M, N, K, nullptr) == dnnl_success,
                              "Cannot initialize brgemm descriptor due to invalid params");

    if (with_amx) {
        OV_CPU_JIT_EMITTER_ASSERT(palette && brgemm_init_tiles(desc, palette) == dnnl_success,
                                  "Cannot initialize brgemm tiles due to invalid params");
    }

    cpu::x64::brgemm_kernel_t* kernel_ = nullptr;
    OV_CPU_JIT_EMITTER_ASSERT(brgemm_kernel_create(&kernel_, desc) == dnnl_success, "Cannot create brgemm kernel due to invalid params");
    kernel = std::unique_ptr<brgemm_kernel_t>(kernel_);
}

void BrgemmBaseKernelExecutor::execute_brgemm_kernel(const std::shared_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t>& kernel,
                                                     const void* src, const void* wei, void* dst, void* scratch, bool with_comp) {
    cpu::x64::brgemm_kernel_params_t brgemm_p;
    brgemm_p.batch = nullptr;  // default value
    brgemm_p.ptr_A = src;
    brgemm_p.ptr_B = wei;
    brgemm_p.ptr_C = dst;
    brgemm_p.ptr_D = dst;
    brgemm_p.ptr_buf = scratch;
    brgemm_p.ptr_bias = nullptr;
    brgemm_p.do_post_ops = with_comp;
    brgemm_p.do_apply_comp = with_comp;
    brgemm_p.skip_accm = 0;
    brgemm_p.BS = 1;  // default value
    OV_CPU_JIT_EMITTER_ASSERT(kernel, "has nullptr Brgemm kernel");
    (*kernel)(&brgemm_p);
}

#undef DIM_CAST
#undef DTYPE_CAST
#undef PRINT
#undef EQ
#undef HASH

}   // namespace intel_cpu
}   // namespace ov
