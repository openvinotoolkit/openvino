// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "set_brgemm_cpu_blocking_params.hpp"

#include "snippets/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/matcher.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"

#include "cpu_shape.h"
#include "utils/general_utils.h"

namespace ov {
namespace intel_cpu {
pass::SetBrgemmCPUBlockingParams::SetBrgemmCPUBlockingParams() {
    MATCHER_SCOPE(SetBrgemmCPUBlockingParams);

    auto m_brgemm = ov::pass::pattern::wrap_type<BrgemmCPU>();

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::SetBrgemmCPUBlockingParams")
        const auto node = m.get_match_root();
        auto brgemm = ov::as_type_ptr<BrgemmCPU>(node);
        if (brgemm->is_dynamic()) {
            return false;
        }

        const auto& input_1_precision = brgemm->get_input_element_type(1);
        // Ticket: 113745
        // TODO: extend block size selection heuristics
        auto get_block_size_m = [&](const size_t M) {
            return 32;
        };
        auto get_block_size_k = [&](const size_t K) {
            if (input_1_precision != ov::element::f32)
                return K;
            return K > 1024 ? 1024 : K > 512 ? 512 : K;
        };
        auto get_block_size_n = [&](const size_t N) {
            return input_1_precision != ov::element::f32 ? N : 64;
        };

        const auto brgemm_in0_dims = snippets::utils::get_planar_pshape(brgemm->input(0)).get_shape();
        const auto M = *(brgemm_in0_dims.rbegin() + 1);
        const auto K = *brgemm_in0_dims.rbegin();
        const auto brgemm_in1_dims = snippets::utils::get_planar_pshape(brgemm->input(1)).get_shape();
        const auto N = *brgemm_in1_dims.rbegin();
        if (brgemm->is_with_data_repacking()) {
            const auto brgemm_copy_b = brgemm->get_brgemm_copy();
            const bool isAMXSupported = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx);
            const auto precision = brgemm_copy_b->get_src_element_type();
            const auto brgemmVNNIFactor = brgemm_copy_b->get_brgemm_vnni_factor();
            const bool use_amx = isAMXSupported && precision != ov::element::f32 && (K % brgemmVNNIFactor == 0) && (N % brgemmVNNIFactor == 0);

            const size_t copy_b_block_size_k = use_amx ? get_block_size_k(K) : K;
            const size_t copy_b_block_size_n = 64;

            brgemm_copy_b->set_k_block_size(copy_b_block_size_k);
            brgemm_copy_b->set_n_block_size(copy_b_block_size_n);
        }

        brgemm->set_m_block_size(get_block_size_m(M));
        brgemm->set_k_block_size(get_block_size_k(K));
        brgemm->set_n_block_size(get_block_size_n(N));
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(m_brgemm, matcher_name);
    register_matcher(m, callback);
}
} // namespace intel_cpu
} // namespace ov