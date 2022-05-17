/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#include <cassert>
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace injector {

bool is_supported(const post_ops_ok_args_t &post_ops_ok_args) {
    const cpu_isa_t isa = post_ops_ok_args.isa;
    const post_ops_t &post_ops = post_ops_ok_args.post_ops;
    const memory_desc_wrapper *dst_d = post_ops_ok_args.dst_d;
    const auto &enabled_bcast_strategy
            = post_ops_ok_args.enabled_bcast_strategy;

    for (const auto &post_op : post_ops.entry_) {
        if (post_op.is_eltwise()) {
            const auto res
                    = eltwise_injector::is_supported(isa, post_op.eltwise.alg);
            if (!res) return false;
        } else if (post_op.is_binary()) {
            const auto &src1_desc = post_op.binary.src1_desc;
            const auto res = binary_injector::is_supported(
                    isa, src1_desc, *dst_d, enabled_bcast_strategy);
            if (!res) return false;
        }
    }
    return true;
}

template <cpu_isa_t isa, typename Vmm>
jit_uni_postops_injector_t<isa, Vmm>::jit_uni_postops_injector_t(
        jit_generator *host, const post_ops_t &post_ops,
        const eltwise_injector::static_params_t &eltwise_static_params,
        const quantization_injector::static_params_t &quantization_static_params)
        : post_ops_(post_ops)
        , host_(host)
        , binary_injector_(nullptr) {

    const auto &esp = eltwise_static_params;
    const auto &qsp = quantization_static_params;

    for (const auto &post_op : post_ops.entry_) {
        if (post_op.is_eltwise()) {
            alg_to_eltwise_injector_.emplace(post_op.eltwise.alg,
                                             jit_uni_eltwise_injector_f32<isa>(host_, post_op.eltwise,
                                                                               esp.save_state, esp.p_table, esp.k_mask, esp.is_fwd,
                                                                               esp.use_dst));
        } else if (post_op.is_depthwise()) {
            depthwise_injectors.emplace_back(new jit_uni_depthwise_injector_f32<isa>(
                    host,
                    post_op
            ));
        } else if (post_op.is_quantization()) {
            quantization_injectors.emplace_back(new jit_uni_quantization_injector_f32<isa, Vmm>(
                    host,
                    post_op,
                    Vmm(qsp.vmm_d_weights_idx), Vmm(qsp.vmm_d_bias_idx), qsp.reg_d_weights, qsp.reg_d_bias
            ));
        }
    }
}

template <cpu_isa_t isa, typename Vmm>
jit_uni_postops_injector_t<isa, Vmm>::jit_uni_postops_injector_t(
        jit_generator *host, const post_ops_t &post_ops,
        const binary_injector::static_params_t &binary_static_params,
        const eltwise_injector::static_params_t &eltwise_static_params,
        const quantization_injector::static_params_t &quantization_static_params,
        const lambda_jit_injectors_t &lambda_jit_injectors)
    : post_ops_(post_ops)
    , host_(host)
    , binary_injector_(nullptr)
    , lambda_jit_injectors_(lambda_jit_injectors) {

    const auto &esp = eltwise_static_params;
    const auto &qsp = quantization_static_params;
    bool is_binary = false;
    bool is_eltwise = false;

    for (const auto &post_op : post_ops.entry_) {
        if (post_op.is_eltwise()) {
            is_eltwise = true;
            alg_to_eltwise_injector_.emplace(post_op.eltwise.alg,
                    jit_uni_eltwise_injector_f32<isa>(host_, post_op.eltwise,
                            esp.save_state, esp.p_table, esp.k_mask, esp.is_fwd,
                            esp.use_dst));
        } else if (post_op.is_binary()) {
            is_binary = true;
        } else if (post_op.is_depthwise()) {
            depthwise_injectors.emplace_back(new jit_uni_depthwise_injector_f32<isa>(
                    host,
                    post_op
            ));
        } else if (post_op.is_quantization()) {
            quantization_injectors.emplace_back(new jit_uni_quantization_injector_f32<isa, Vmm>(
                    host,
                    post_op,
                    Vmm(qsp.vmm_d_weights_idx), Vmm(qsp.vmm_d_bias_idx), qsp.reg_d_weights, qsp.reg_d_bias
            ));
        }
    }

    if (is_superset(isa, avx512_common) && is_eltwise && is_binary
            && binary_static_params.rhs_arg_static_params.tail_size)
        assert(eltwise_static_params.k_mask
                != binary_static_params.rhs_arg_static_params.tail_opmask &&
                "Binary tail opmask should be different than eltwise injector \
                opmask. Otherwise eltwise injector will overwrite binary tail \
                opmask.");

    if (is_binary)
        binary_injector_ = utils::make_unique<
                binary_injector::jit_uni_binary_injector_t<isa, Vmm>>(
                host, binary_static_params);
}

template <cpu_isa_t isa, typename Vmm>
jit_uni_postops_injector_t<isa, Vmm>::jit_uni_postops_injector_t(
        jit_generator *host, const post_ops_t &post_ops,
        const binary_injector::static_params_t &binary_static_params)
    : jit_uni_postops_injector_t(host, post_ops, binary_static_params,
            eltwise_injector::static_params_t(), quantization_injector::static_params_t(),
            lambda_jit_injectors_t()) {}

template <cpu_isa_t isa, typename Vmm>
jit_uni_postops_injector_t<isa, Vmm>::jit_uni_postops_injector_t(
        jit_generator *host, const post_ops_t &post_ops,
        const binary_injector::static_params_t &binary_static_params,
        const lambda_jit_injectors_t &lambda_jit_injectors)
    : jit_uni_postops_injector_t(host, post_ops, binary_static_params,
            eltwise_injector::static_params_t(), quantization_injector::static_params_t(),
            lambda_jit_injectors) {}

template <cpu_isa_t isa, typename Vmm>
jit_uni_postops_injector_t<isa, Vmm>::jit_uni_postops_injector_t(
        jit_generator *host, const post_ops_t &post_ops,
        const binary_injector::static_params_t &binary_static_params,
        const eltwise_injector::static_params_t &eltwise_static_params)
    : jit_uni_postops_injector_t(host, post_ops, binary_static_params,
            eltwise_static_params,
            quantization_injector::static_params_t(), lambda_jit_injectors_t()) {}

template <cpu_isa_t isa, typename Vmm>
jit_uni_postops_injector_t<isa, Vmm>::jit_uni_postops_injector_t(jit_generator *host,
                                                            const post_ops_t &post_ops,
                                                            const binary_injector::static_params_t &binary_static_params,
                                                            const quantization_injector::static_params_t &quantization_static_params)
        : jit_uni_postops_injector_t(host, post_ops, binary_static_params,
                                     eltwise_injector::static_params_t(),
                                     quantization_static_params, lambda_jit_injectors_t()) {}

template <cpu_isa_t isa, typename Vmm>
jit_uni_postops_injector_t<isa, Vmm>::jit_uni_postops_injector_t(jit_generator *host,
        const post_ops_t &post_ops,
        const binary_injector::static_params_t &binary_static_params,
        const eltwise_injector::static_params_t &eltwise_static_params,
        const quantization_injector::static_params_t &quantization_static_params)
        : jit_uni_postops_injector_t(host, post_ops, binary_static_params,
                eltwise_static_params, quantization_static_params, lambda_jit_injectors_t()) {}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_postops_injector_t<isa, Vmm>::compute_vector_range(
        size_t start_idx, size_t end_idx,
        const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params) {

    injector_utils::vmm_index_set_t vmm_idxs;
    for (size_t i = start_idx; i < end_idx; i++)
        vmm_idxs.emplace(i);
    compute_vector_range(vmm_idxs, rhs_arg_params);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_postops_injector_t<isa, Vmm>::compute_vector_range(
        size_t start_idx, size_t end_idx,
        const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params,
        const depthwise_injector::dynamic_params_t &ddp,
        const quantization_injector::dynamic_params_t &qdp) {

    injector_utils::vmm_index_set_t vmm_idxs;
    for (size_t i = start_idx; i < end_idx; i++)
        vmm_idxs.emplace(i);
    compute_vector_range(vmm_idxs, rhs_arg_params, ddp, qdp);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_postops_injector_t<isa, Vmm>::compute_vector_range(
        size_t start_idx, size_t end_idx) {
    compute_vector_range(
            start_idx, end_idx, binary_injector::rhs_arg_dynamic_params_t());
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_postops_injector_t<isa, Vmm>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs,
        const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params,
        const depthwise_injector::dynamic_params_t &ddp,
        const quantization_injector::dynamic_params_t &qdp, bool is_broadcast) {

    std::size_t rhs_arg_idx = 0;
    std::size_t quantization_inj_idx = 0;
    std::size_t depthwise_inj_idx = 0;
    std::size_t post_ops_data_offset = 0;
    for (int i = 0; i < post_ops_.len(); i++) {
        const auto &post_op = post_ops_.entry_[i];

        if (post_op.is_eltwise()) {
            alg_to_eltwise_injector_.at(post_op.eltwise.alg)
                    .compute_vector_range(vmm_idxs);
        } else if (post_op.is_binary()) {
            binary_injector_->compute_vector_range(
                    vmm_idxs, rhs_arg_idx, post_op, rhs_arg_params);
            ++rhs_arg_idx;
        } else if (post_op.is_depthwise()) {
            const Xbyak::RegExp depthwise_arg_base = ddp.reg_post_ops_data + ddp.base_post_ops_data_offset + post_ops_data_offset;
            if (ddp.useAddr)
                depthwise_injectors[depthwise_inj_idx]->init_ptrs(depthwise_arg_base, ddp.reg_d_weights, ddp.reg_d_bias, ddp.reg_init_off_addr, false);
            else
                depthwise_injectors[depthwise_inj_idx]->init_ptrs(depthwise_arg_base, ddp.reg_d_weights, ddp.reg_d_bias, ddp.reg_init_off, false);

            bool need_to_preserve = false;
            if (post_op.depthwise.alg == dnnl_depthwise_prelu && isa == sse41)
                need_to_preserve = true;

            for (auto vmm_idx : vmm_idxs) {
                depthwise_injectors[depthwise_inj_idx]->compute(vmm_idx, vmm_idx + 1,
                                                                need_to_preserve ? 0 : ddp.vmm_d_weights_idx, ddp.vmm_d_bias_idx,
                                                                ddp.reg_d_weights, ddp.reg_d_bias,
                                                                is_broadcast, ddp.vmm_idx_off.at(vmm_idx), need_to_preserve);
            }

            post_ops_data_offset += depthwise_injectors[depthwise_inj_idx]->memoryStep();
            ++rhs_arg_idx;
            depthwise_inj_idx++;
        } else if (post_op.is_quantization()) {
            std::vector<std::pair<int, std::set<size_t>>> vecOfVmmIdxsSets;

            std::multimap<int, size_t> offsetVmmIdxMap;
            for (auto vmm_idx : vmm_idxs) {
                offsetVmmIdxMap.insert({qdp.vmm_idx_off.at(vmm_idx), vmm_idx});
            }

            auto externalIt = offsetVmmIdxMap.begin();
            while (externalIt != offsetVmmIdxMap.end()) {
                auto internalIt = externalIt;
                auto endInternalIt = offsetVmmIdxMap.upper_bound(externalIt->first);

                std::set<size_t> vmmIndexesToProcess;
                while (internalIt != endInternalIt) {
                    vmmIndexesToProcess.insert(internalIt->second);
                    internalIt++;
                }
                vecOfVmmIdxsSets.push_back({externalIt->first, vmmIndexesToProcess});

                externalIt = endInternalIt;
            }

            bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
            bool do_rounding = do_dequantization || qdp.dst_dt == dnnl_f32 || i != post_ops_.len() - 1;

            const Xbyak::RegExp quant_arg_base = qdp.reg_post_ops_data + qdp.base_post_ops_data_offset + post_ops_data_offset;
            if (qdp.useAddr)
                quantization_injectors[quantization_inj_idx]->init_crop_ptrs(quant_arg_base, qdp.reg_oc_off_addr);
            else
                quantization_injectors[quantization_inj_idx]->init_crop_ptrs(quant_arg_base, qdp.reg_oc_off);

            for (auto &IdxSetPair : vecOfVmmIdxsSets) {
                quantization_injectors[quantization_inj_idx]->compute_crop(IdxSetPair.second, IdxSetPair.first, false, is_broadcast);
            }

            if (qdp.useAddr)
                quantization_injectors[quantization_inj_idx]->init_input_scale_shift_ptrs(quant_arg_base, qdp.reg_oc_off_addr);
            else
                quantization_injectors[quantization_inj_idx]->init_input_scale_shift_ptrs(quant_arg_base, qdp.reg_oc_off);

            for (auto &IdxSetPair : vecOfVmmIdxsSets) {
                quantization_injectors[quantization_inj_idx]->compute_input_scale_shift(IdxSetPair.second, IdxSetPair.first, do_rounding,
                                                                                        false, is_broadcast);
            }

            if (qdp.useAddr)
                quantization_injectors[quantization_inj_idx]->init_output_scale_shift_ptrs(quant_arg_base, qdp.reg_oc_off_addr);
            else
                quantization_injectors[quantization_inj_idx]->init_output_scale_shift_ptrs(quant_arg_base, qdp.reg_oc_off);

            for (auto &IdxSetPair : vecOfVmmIdxsSets) {
                quantization_injectors[quantization_inj_idx]->compute_output_scale_shift(IdxSetPair.second, IdxSetPair.first, false, is_broadcast);
            }

            post_ops_data_offset += quantization_injectors[quantization_inj_idx]->memoryStep();
            ++rhs_arg_idx;
            quantization_inj_idx++;
        } else {
            const auto lam = lambda_jit_injectors_.find(post_op.kind);
            if (lam != lambda_jit_injectors_.end()) lam->second();
        }
    }
}
template <cpu_isa_t isa, typename Vmm>
void jit_uni_postops_injector_t<isa, Vmm>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs) {
    compute_vector_range(vmm_idxs, binary_injector::rhs_arg_dynamic_params_t());
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_postops_injector_t<isa, Vmm>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs,
        const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params) {
    compute_vector_range(vmm_idxs, rhs_arg_params, depthwise_injector::dynamic_params_t(), quantization_injector::dynamic_params_t());
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_postops_injector_t<isa, Vmm>::prepare_table(bool gen_table) {
    for (auto &alg_elt_inject : alg_to_eltwise_injector_)
        alg_elt_inject.second.prepare_table(gen_table);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_postops_injector_t<isa, Vmm>::compute_vector(size_t idx,
        const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params) {
    compute_vector_range({idx}, rhs_arg_params);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_postops_injector_t<isa, Vmm>::compute_vector(size_t idx) {
    compute_vector_range({idx});
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_postops_injector_t<isa, Vmm>::compute_vector(size_t idx,
        const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params,
        const depthwise_injector::dynamic_params_t &ddp,
        const quantization_injector::dynamic_params_t &qdp) {
    compute_vector_range({idx}, rhs_arg_params, ddp, qdp);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_postops_injector_t<isa, Vmm>::compute_vector(size_t idx,
        const depthwise_injector::dynamic_params_t &ddp,
        const quantization_injector::dynamic_params_t &qdp, bool is_broadcast) {
    compute_vector_range({idx}, binary_injector::rhs_arg_dynamic_params_t(), ddp, qdp, is_broadcast);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_postops_injector_t<isa, Vmm>::set_lambda_injector(
        dnnl_primitive_kind_t kind, const std::function<void()> &jit_injector) {
    lambda_jit_injectors_[kind] = jit_injector;
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_postops_injector_t<isa, Vmm>::push_post_ops_data_on_stack(const Xbyak::Reg64& post_ops_data_reg, std::size_t post_ops_data_offset,
        const Xbyak::Reg64& aux_reg0, const Xbyak::Reg64& aux_reg1) {
    for (int i = 0; i < post_ops_.len(); i++) {
        if (post_ops_.entry_[i].is_depthwise() || post_ops_.entry_[i].is_quantization()) {
            post_ops_pointers_count++;
        }
    }

    if (post_ops_pointers_count != 0) {
        host_->sub(host_->rsp, post_ops_pointers_count * sizeof(float *));

        host_->mov(aux_reg0, host_->ptr[post_ops_data_reg + post_ops_data_offset]);
        for (size_t i = 0; i < post_ops_pointers_count; i++) {
            host_->mov(aux_reg1, host_->ptr[aux_reg0 + i * sizeof(float *)]);
            host_->mov(host_->ptr[host_->rsp + i * sizeof(float *)], aux_reg1);
        }
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_postops_injector_t<isa, Vmm>::reset_stack_pointer() {
    if (post_ops_pointers_count != 0) {
        host_->add(host_->rsp, post_ops_pointers_count * sizeof(float *));
    }
}

post_ops_ok_args_t::post_ops_ok_args_t(const cpu_isa_t isa,
        const std::vector<post_op_type> &accepted_post_op_types,
        const post_ops_t &post_ops, const memory_desc_wrapper *dst_d,
        const bool sum_at_pos_0_only, const bool sum_requires_scale_one,
        const bool sum_requires_zp_zero,
        const bcast_set_t &enabled_bcast_strategy)
    : isa(isa)
    , accepted_post_op_types(accepted_post_op_types)
    , post_ops(post_ops)
    , dst_d(dst_d)
    , sum_at_pos_0_only(sum_at_pos_0_only)
    , sum_requires_scale_one(sum_requires_scale_one)
    , sum_requires_zp_zero(sum_requires_zp_zero)
    , enabled_bcast_strategy(enabled_bcast_strategy) {};

bool post_ops_ok(const post_ops_ok_args_t &post_ops_ok_args) {
    const cpu_isa_t isa = post_ops_ok_args.isa;
    const std::vector<post_op_type> &accepted_post_op_types
            = post_ops_ok_args.accepted_post_op_types;
    const post_ops_t &post_ops = post_ops_ok_args.post_ops;
    const memory_desc_wrapper *dst_d = post_ops_ok_args.dst_d;
    const bool sum_at_pos_0_only = post_ops_ok_args.sum_at_pos_0_only;
    const bool sum_requires_scale_one = post_ops_ok_args.sum_requires_scale_one;
    const bool sum_requires_zp_zero = post_ops_ok_args.sum_requires_zp_zero;
    const auto &enabled_bcast_strategy
            = post_ops_ok_args.enabled_bcast_strategy;

    const auto is_accepted_postop = [&](const int idx) {
        for (const auto &post_op : accepted_post_op_types) {
            const auto &entry = post_ops.entry_[idx];
            switch (post_op) {
                case sum:
                    if (entry.is_sum(false, false)) {
                        if (sum_requires_scale_one && entry.sum.scale != 1)
                            return false;
                        if (sum_requires_zp_zero && entry.sum.zero_point != 0)
                            return false;
                        return IMPLICATION(sum_at_pos_0_only, idx == 0);
                    }
                    break;
                case eltwise:
                    if (entry.is_eltwise()) {
                        const auto alg = entry.eltwise.alg;
                        return eltwise_injector::is_supported(isa, alg);
                    }
                    break;
                case binary:
                    if (entry.is_binary()) {
                        assert(dst_d != nullptr && "dst_d is null");
                        return binary_injector::is_supported(isa,
                                entry.binary.src1_desc, *dst_d,
                                enabled_bcast_strategy);
                    }
                    break;
                case depthwise: if (entry.is_depthwise()) return true; break;
                case quantization: if (entry.is_quantization()) return true; break;
                default: assert(false && "Unhandled post_op type");
            }
        }
        return false;
    };

    for (int i = 0; i < post_ops.len(); i++) {
        if (!is_accepted_postop(i)) return false;
    }

    return true;
}

template class jit_uni_postops_injector_t<avx512_core_bf16>;
template class jit_uni_postops_injector_t<avx512_core>;
template class jit_uni_postops_injector_t<avx512_core, Xbyak::Ymm>;
template class jit_uni_postops_injector_t<avx512_core, Xbyak::Xmm>;
template class jit_uni_postops_injector_t<avx512_common>;
template class jit_uni_postops_injector_t<avx512_common, Xbyak::Ymm>;
template class jit_uni_postops_injector_t<avx2>;
template class jit_uni_postops_injector_t<avx2, Xbyak::Xmm>;
template class jit_uni_postops_injector_t<avx>;
template class jit_uni_postops_injector_t<avx, Xbyak::Xmm>;
template class jit_uni_postops_injector_t<sse41>;

} // namespace injector
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
