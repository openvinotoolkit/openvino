// Copyright (C) 2020-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_generic.hpp"

#include "common/utils.hpp"
#include "dnnl_extension_utils.h"
#include "utils/general_utils.h"

#define PRINT(X) ss << #X << " = " << (X) << "\n"
#define EQ(X)    X == rhs.X
#define HASH(X)  seed = dnnl::impl::hash_combine(seed, X)

namespace ov::intel_cpu {

bool BrgemmGenericKernelConfig::is_completed() const {
    return !one_of(0, m_M, m_N, m_K, m_LDA, m_LDB, m_LDC) || is_empty();
}

bool BrgemmGenericKernelConfig::is_empty() const {
    return everyone_is(0, m_M, m_N, m_K, m_LDA, m_LDB, m_LDC, m_beta);
}

bool BrgemmGenericKernelConfig::operator==(const BrgemmGenericKernelConfig& rhs) const {
    return EQ(m_beta) && EQ(m_M) && EQ(m_N) && EQ(m_K) && EQ(m_LDA) && EQ(m_LDB) && EQ(m_LDC);
}

void BrgemmGenericKernelConfig::update(int64_t M,
                                       int64_t N,
                                       int64_t K,
                                       int64_t LDA,
                                       int64_t LDB,
                                       int64_t LDC,
                                       float beta) {
    // If M/N/K is zero, it means that Brgemm won't be executed (in Loop with work_amount = 0, for example)
    // To process this case, we have to make this Config as empty (nullify runtime parameters)
    if (one_of(0, M, N, K)) {
        m_M = 0;
        m_N = 0;
        m_K = 0;
        m_LDA = 0;
        m_LDB = 0;
        m_LDC = 0;
        m_beta = 0;
    } else {
        m_M = M;
        m_N = N;
        m_K = K;
        m_LDA = LDA;
        m_LDB = LDB;
        m_LDC = LDC;
        m_beta = beta;
    }
}

size_t BrgemmGenericKernelConfig::compute_hash() const {
    size_t seed = 0;
    HASH(m_M);
    HASH(m_N);
    HASH(m_K);
    HASH(m_LDA);
    HASH(m_LDB);
    HASH(m_LDC);
    HASH(m_beta);
    return seed;
}

#ifdef SNIPPETS_DEBUG_CAPS
std::string BrgemmGenericKernelConfig::to_string() const {
    std::stringstream ss;
    PRINT(m_M);
    PRINT(m_N);
    PRINT(m_K);
    PRINT(m_LDA);
    PRINT(m_LDB);
    PRINT(m_LDC);
    PRINT(m_beta);
    return ss.str();
}
#endif

float BrgemmKernelExecutorHelper::get_beta(
    const ov::snippets::lowered::LoopManagerPtr& loop_manager,
    int loop_id,
    const ov::snippets::lowered::ExpandedLoopInfoPtr& current_expanded_loop_info) {
    // Find all Expanded loops with the same Unified loop information -> they were decomposed from this Unified Loop.
    // Note that LoopInfo are normalized and sorted (due to NormalizedLoopIDs pass).
    // It means that previous executed Loops have Loop ID less the current Loop ID.
    // - If there is executed Loop (work_amount > 0) and evaluated before the current -> the current Brgemm should have
    // `beta = 1`.
    // - If there is not this Loop -> the current executed Brgemm should have `beta = 0`.
    if (loop_id > 0) {
        const auto& current_unified_loop_info = current_expanded_loop_info->get_unified_loop_info();
        // Check the previous Loops
        --loop_id;
        while (loop_id >= 0) {
            const auto& expanded_loop_info =
                loop_manager->get_loop_info<ov::snippets::lowered::ExpandedLoopInfo>(loop_id);
            if (expanded_loop_info->get_unified_loop_info() != current_unified_loop_info) {
                return 0;
            }
            if (expanded_loop_info->get_work_amount() > 0) {
                // there is previous executed Brgemm with `beta = 0` -> the current Brgemm should have `beta = 1`
                return 1;
            }
            --loop_id;
        }
    }
    return 0;
}

std::tuple<int64_t, int64_t, int64_t, float> BrgemmKernelExecutorHelper::get_runtime_brgemm_params(
    const ov::snippets::lowered::ExpressionPtr& expr,
    const ov::snippets::lowered::LinearIRCPtr& linear_ir) {
    const auto& input_pds = expr->get_input_port_descriptors();
    const auto& output_pds = expr->get_output_port_descriptors();
    OV_CPU_JIT_EMITTER_ASSERT((input_pds.size() == 2 || input_pds.size() == 3) && output_pds.size() == 1,
                              "Invalid number of in/out port descriptors");

    const auto& in0_shape = snippets::utils::get_planar_vdims(input_pds[0]->get_shape(), input_pds[0]->get_layout());
    const auto& in1_shape = snippets::utils::get_planar_vdims(input_pds[1]->get_shape(), input_pds[1]->get_layout());
    const auto& in0_subtensor = input_pds[0]->get_subtensor();
    const auto& in1_subtensor = input_pds[1]->get_subtensor();

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
            beta = get_beta(loop_manager, static_cast<int>(loop_ids.back()), current_expanded_loop_info);
        }
    }

    return std::make_tuple(M, N, K, beta);
}

#undef PRINT
#undef EQ
#undef HASH

}  // namespace ov::intel_cpu
