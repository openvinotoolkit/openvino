// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_generic.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "common/utils.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/except.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/expression_port.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/loop_port.hpp"
#include "snippets/utils/utils.hpp"
#include "utils/general_utils.h"

#define PRINT(X) ss << #X << " = " << (X) << "\n"
#define EQ(X)    X == rhs.X
#define HASH(X)  seed = dnnl::impl::hash_combine(seed, X)

namespace ov::intel_cpu {
using namespace ov::snippets::lowered;

bool BrgemmGenericKernelConfig::is_completed() const {
    return none_of(0, m_M, m_N, m_K, m_LDA, m_LDB, m_LDC) || is_empty();
}

bool BrgemmGenericKernelConfig::is_empty() const {
    return all_of(0, m_M, m_N, m_K, m_LDA, m_LDB, m_LDC, m_beta);
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
    if (any_of(0, M, N, K)) {
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

float BrgemmKernelExecutorHelper::get_beta(const LoopManagerPtr& loop_manager,
                                           int loop_id,
                                           const ExpandedLoopInfoPtr& current_expanded_loop_info) {
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
            const auto& expanded_loop_info = loop_manager->get_loop_info<ExpandedLoopInfo>(loop_id);
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

bool BrgemmKernelExecutorHelper::is_out_buffer_inside_n_loop(const std::vector<LoopPort>& n_loop_out_ports,
                                                             const ExpressionPort& cur_brgemm_out_port,
                                                             const LoopManagerPtr& loop_manager,
                                                             const std::optional<size_t>& inner_loop_idx) {
    auto is_port_inside_n_loop = [&n_loop_out_ports](const ExpressionPort& port) {
        return std::none_of(n_loop_out_ports.cbegin(), n_loop_out_ports.cend(), [&port](const LoopPort& lp) {
            return *lp.get_expr_port() == port;
        });
    };
    if (!inner_loop_idx) {
        return is_port_inside_n_loop(cur_brgemm_out_port);
    }

    // Note: if there is inner blocking loop, only brgemm from last inner loop iteration is connected
    // to the output of N blocking loop. So we need to find the last inner loop info
    const auto& cur_inner_loop_info = loop_manager->get_loop_info<ExpandedLoopInfo>(*inner_loop_idx);
    const auto last_inner_loop_info = [&]() {
        auto last_iter_info = cur_inner_loop_info;
        size_t next_loop_id = *inner_loop_idx + 1;
        const auto& loops_map = loop_manager->get_map();
        const auto unified_loop_info = cur_inner_loop_info->get_unified_loop_info();

        while (loops_map.count(next_loop_id)) {
            auto next_info = loop_manager->get_loop_info<ExpandedLoopInfo>(next_loop_id);
            if (next_info->get_unified_loop_info() != unified_loop_info) {
                return last_iter_info;
            }
            last_iter_info = next_info;
            ++next_loop_id;
        }
        return last_iter_info;
    }();

    // Since all ExpandedLoopInfo represent the same loop, their output loop ports order is the same,
    // so we can compute distance_till_needed_port for cur_inner_loop_info to use it for last_inner_loop_info ports
    const auto& cur_inner_loop_out_ports = cur_inner_loop_info->get_output_ports();
    const auto found_inner_loop_port_it = std::find_if(cur_inner_loop_out_ports.cbegin(),
                                                       cur_inner_loop_out_ports.cend(),
                                                       [&cur_brgemm_out_port](const LoopPort& lp) {
                                                           return *lp.get_expr_port() == cur_brgemm_out_port;
                                                       });

    const auto& last_inner_loop_out_ports = last_inner_loop_info->get_output_ports();
    const auto it_increment = std::distance(cur_inner_loop_out_ports.cbegin(), found_inner_loop_port_it);
    const auto& last_brgemm_out_port = std::next(last_inner_loop_out_ports.cbegin(), it_increment)->get_expr_port();
    return is_port_inside_n_loop(*last_brgemm_out_port);
}

std::tuple<int64_t, int64_t, int64_t, float, int64_t> BrgemmKernelExecutorHelper::get_runtime_brgemm_params(
    const ExpressionPtr& expr,
    const LinearIRCPtr& linear_ir) {
    const auto& input_pds = expr->get_input_port_descriptors();
    const auto& output_pds = expr->get_output_port_descriptors();
    OV_CPU_JIT_EMITTER_ASSERT(input_pds.size() >= 2 && output_pds.size() == 1,
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

    const auto& loop_ids = expr->get_loop_ids();
    const auto& loop_manager = linear_ir->get_loop_manager();
    const auto [m_loop_idx, n_loop_idx, k_loop_idx] = [&]() {
        size_t loop_idx = 0;
        auto get_loop_idx = [&]() {
            assert(loop_idx < loop_ids.size() && "Loop is missed");
            return loop_ids[loop_idx++];
        };
        std::optional<size_t> m_loop_idx, n_loop_idx, k_loop_idx;
        // Note: order of get_loop_idx() calls is important!
        if (!ov::snippets::utils::is_full_dim_value(M)) {
            m_loop_idx = get_loop_idx();
        }
        if (!ov::snippets::utils::is_full_dim_value(N)) {
            n_loop_idx = get_loop_idx();
        }
        if (!ov::snippets::utils::is_full_dim_value(K)) {
            k_loop_idx = get_loop_idx();
        }
        return std::make_tuple(m_loop_idx, n_loop_idx, k_loop_idx);
    }();

    /* ------- Dimension M ----------*/
    if (!m_loop_idx) {
        M = *++in0_shape.rbegin();
    } else {
        const auto& current_expanded_loop_info = loop_manager->get_loop_info<ExpandedLoopInfo>(*m_loop_idx);
        const auto& in_ports = current_expanded_loop_info->get_input_ports();
        const auto& out_ports = current_expanded_loop_info->get_output_ports();
        // Quick validation check: Should we check that port is really Brgemm port?
        // If BrgemmCopyB in the Loop by M -> first input port will be BrgemmCopyB with `incremented=false`
        // to avoid extra checks, we validate only first input port
        auto check_port = [&](const LoopPort& p) {
            return p.get_dim_idx() == 1 && p.is_processed();
        };
        OPENVINO_ASSERT(
            in_ports.size() > 1 && check_port(in_ports[0]) && out_ports.size() == 1 && check_port(out_ports[0]),
            "Incorrect Loop by Brgemm dimension M");
        M = current_expanded_loop_info->get_work_amount() > 0 ? current_expanded_loop_info->get_increment() : 0;
        input_pds[0]->set_subtensor_dim(1, M);
        output_pds[0]->set_subtensor_dim(1, M);
    }

    // Default LDC value if the N blocking loop is absent or cur brgemm is its output port
    auto LDC = snippets::utils::get_dim_stride(expr->get_output_port(0));
    /* ------- Dimension N ----------*/
    if (!n_loop_idx) {
        N = *in1_shape.rbegin();
    } else {
        const auto& current_expanded_loop_info = loop_manager->get_loop_info<ExpandedLoopInfo>(*n_loop_idx);
        const auto& in_ports = current_expanded_loop_info->get_input_ports();
        const auto& out_ports = current_expanded_loop_info->get_output_ports();
        // Quick validation check: Should we check that port is really Brgemm port?
        auto check_port = [&](const LoopPort& p) {
            return p.get_dim_idx() == 0 && p.is_processed();
        };
        OPENVINO_ASSERT(in_ports.size() >= 2 && !in_ports.front().is_processed() && check_port(in_ports[1]) &&
                            check_port(out_ports.back()),
                        "Incorrect Loop by Brgemm dimension N");
        N = current_expanded_loop_info->get_work_amount() > 0 ? current_expanded_loop_info->get_increment() : 0;
        input_pds[1]->set_subtensor_dim(0, N);
        output_pds[0]->set_subtensor_dim(0, N);

        const auto cur_out_port = expr->get_output_port(0);
        // Note: if brgemm output buffer is inside N blocking loop,
        // output leading dimension must be equal to block size
        if (is_out_buffer_inside_n_loop(out_ports, cur_out_port, loop_manager, k_loop_idx)) {
            LDC = N;
        }
    }

    /* ------- Dimension K ----------*/
    // 1. If Brgemm block processes full dimension K -> `beta = 0`
    // 2. If Brgemm block processes part of the dimension K (there is blocking), need to find
    //    the most first executed Brgemm Block in Loops which iterate through dimension K (work_amount > 0).
    //    First of them will have `beta = 0`, other - `beta = 1`
    float beta = 0;
    if (!k_loop_idx) {
        K = *in0_shape.rbegin();
    } else {
        const auto& current_expanded_loop_info = loop_manager->get_loop_info<ExpandedLoopInfo>(*k_loop_idx);
        const auto& in_ports = current_expanded_loop_info->get_input_ports();
        const auto& out_ports = current_expanded_loop_info->get_output_ports();
        // Quick validation check: Should we check that port is really Brgemm port?
        OPENVINO_ASSERT(in_ports.size() >= 2 && in_ports.front().get_dim_idx() == 0 &&
                            in_ports.front().is_processed() && in_ports[1].get_dim_idx() == 1 &&
                            in_ports[1].is_processed() && out_ports.size() == 1 && !out_ports.front().is_processed(),
                        "Incorrect Loop by Brgemm dimension K");
        K = current_expanded_loop_info->get_work_amount() > 0 ? current_expanded_loop_info->get_increment() : 0;
        input_pds[0]->set_subtensor_dim(0, K);
        input_pds[1]->set_subtensor_dim(1, K);
        if (K > 0) {
            beta = get_beta(loop_manager, static_cast<int>(loop_ids.back()), current_expanded_loop_info);
        }
    }

    return std::make_tuple(M, N, K, beta, LDC);
}

#undef PRINT
#undef EQ
#undef HASH

}  // namespace ov::intel_cpu
