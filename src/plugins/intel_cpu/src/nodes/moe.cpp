// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe.hpp"

#include <oneapi/dnnl/dnnl_types.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <common/utils.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dnnl_extension_utils.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "ov_ops/moe.hpp"
#include "shape_inference/shape_inference_pass_through.hpp"
#include "utils/general_utils.h"
#include "utils/plain_tensor.hpp"

using namespace ov;

namespace ov::intel_cpu::node {

using namespace dnnl;

namespace {

struct MoEKey {
    memory::data_type act_dtype;
    memory::data_type weight_dtype;
    int hidden_size;
    int intermediate_size;
    int ic_group_size;
    size_t expert_num;
    bool has_scale;
    bool has_zp;

    [[nodiscard]] size_t hash() const;
    bool operator==(const MoEKey& rhs) const;
};

size_t MoEKey::hash() const {
    using namespace dnnl::impl;
    size_t seed = 0;
    seed = hash_combine(seed, act_dtype);
    seed = hash_combine(seed, weight_dtype);
    seed = hash_combine(seed, hidden_size);
    seed = hash_combine(seed, intermediate_size);
    seed = hash_combine(seed, ic_group_size);
    seed = hash_combine(seed, expert_num);
    seed = hash_combine(seed, has_scale);
    seed = hash_combine(seed, has_zp);
    return seed;
}

bool MoEKey::operator==(const MoEKey& rhs) const {
    bool retVal = act_dtype == rhs.act_dtype;
    retVal = retVal && weight_dtype == rhs.weight_dtype;
    retVal = retVal && hidden_size == rhs.hidden_size;
    retVal = retVal && intermediate_size == rhs.intermediate_size;
    retVal = retVal && ic_group_size == rhs.ic_group_size;
    retVal = retVal && expert_num == rhs.expert_num;
    retVal = retVal && has_scale == rhs.has_scale;
    retVal = retVal && has_zp == rhs.has_zp;
    return retVal;
}
}  // namespace

struct onednn_matmul {
    primitive m_prim;
    memory::desc m_input_md;
    memory::desc m_output_md;
    memory::desc m_wei_md;
    memory::desc m_sc_md;
    memory::desc m_zp_md;
    memory::desc m_bin_md;
    memory::data_type m_w_type = memory::data_type::undef;
    memory::data_type m_a_type = memory::data_type::undef;
    memory::data_type m_sc_dtype = memory::data_type::undef;
    memory::data_type m_zp_dtype = memory::data_type::undef;
    memory::dim m_K = 0;
    memory::dim m_N = 0;
    memory::dim m_M = 0;
    memory::dim m_K_groups = 0;
    primitive_attr attr;
    post_ops postops;
    int bin_post_id = -1;

    const bool m_use_ip = true;
    onednn_matmul() = default;

    onednn_matmul& init(memory::data_type act_dtype,
                        memory::data_type weight_dtype,
                        int batch_size,
                        int ic,
                        int oc,
                        int ic_group_size,
                        bool has_scale,
                        bool has_zp) {
        m_a_type = act_dtype;
        m_w_type = weight_dtype;
        m_K_groups = 0;
        m_K = ic;
        m_N = oc;
        m_M = DNNL_RUNTIME_DIM_VAL;
        if (batch_size > 0) {
            // jit-gemm kernel only support static batch size
            m_M = batch_size;
        }
        if (has_scale) {
            w_scale(ic_group_size);
            if (has_zp) {
                w_zp(ic_group_size);
            }
        }
        m_input_md = memory::desc(memory::dims({m_M, m_K}), act_dtype, memory::format_tag::ab);
        m_output_md = memory::desc(memory::dims({m_M, m_N}), act_dtype, memory::format_tag::ab);
        return *this;
    }

    onednn_matmul& w_scale(int k_group_size) {
        if (m_use_ip) {
            m_sc_dtype = memory::data_type::f32;
            if (k_group_size <= 0) {
                m_K_groups = 1;
            } else {
                OPENVINO_ASSERT((k_group_size % 32) == 0);
                OPENVINO_ASSERT((m_K % k_group_size) == 0);
                m_K_groups = m_K / k_group_size;
            }
            attr.set_scales_dims(DNNL_ARG_WEIGHTS, {m_N, m_K_groups}, m_sc_dtype);
            m_sc_md = memory::desc({m_N, m_K_groups}, m_sc_dtype, memory::format_tag::ba);
        } else {
            m_sc_dtype = memory::data_type::f32;
            if (k_group_size <= 0) {
                m_K_groups = 1;
                // per-OC, no grouping in K dimension
                attr.set_scales(DNNL_ARG_WEIGHTS, (0 << 0) + (1 << 1), {1}, m_sc_dtype);
            } else {
                OPENVINO_ASSERT((k_group_size % 32) == 0);
                OPENVINO_ASSERT((m_K % k_group_size) == 0);
                m_K_groups = m_K / k_group_size;
                attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {k_group_size, 1}, m_sc_dtype);
            }
            m_sc_md = memory::desc({m_K_groups, m_N}, m_sc_dtype, memory::format_tag::ba);
        }
        return *this;
    }

    onednn_matmul& w_zp(int k_group_size) {
        if (m_use_ip) {
            m_zp_dtype = memory::data_type::u8;
            if (k_group_size <= 0) {
                OPENVINO_ASSERT(m_K_groups == 1);
                attr.set_zero_points(DNNL_ARG_WEIGHTS, (0 << 0) + (1 << 1), {1}, m_zp_dtype);
            } else {
                OPENVINO_ASSERT(m_K_groups == (m_K / k_group_size));
                attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {k_group_size, 1}, m_zp_dtype);
            }
            attr.set_zero_points_dims(DNNL_ARG_WEIGHTS, {m_N, m_K_groups}, m_zp_dtype);
            // dtype & shape & layout has to be as following:
            m_zp_md = memory::desc({m_N, m_K_groups}, m_zp_dtype, memory::format_tag::ba);
        } else {
            m_zp_dtype = memory::data_type::s8;
            if (k_group_size <= 0) {
                OPENVINO_ASSERT(m_K_groups == 1);
                attr.set_zero_points(DNNL_ARG_WEIGHTS, (0 << 0) + (1 << 1), {1}, m_zp_dtype);
            } else {
                OPENVINO_ASSERT(m_K_groups == (m_K / k_group_size));
                attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {k_group_size, 1}, m_zp_dtype);
            }
            // dtype & layout must be choosen according to kernel's capability
            m_zp_md = memory::desc({m_K_groups, m_N}, m_zp_dtype, memory::format_tag::ab);
        }
        return *this;
    }

    onednn_matmul& fpmath_f16() {
        attr.set_fpmath_mode(fpmath_mode::f16, true);
        return *this;
    }
    onednn_matmul& post_op_silu() {
        float alpha = 1.0f;
        float beta = 0.0f;
        postops.append_eltwise(algorithm::eltwise_swish, alpha, beta);
        return *this;
    }
    onednn_matmul& post_op_bin_mul(bool per_oc = true) {
        OPENVINO_ASSERT(bin_post_id < 0);
        memory::dim batch_size = m_M;
        if (batch_size == DNNL_RUNTIME_DIM_VAL) {
            batch_size = 1024 * 1024;  // big enough fake static batch
        }

        m_bin_md = memory::desc(memory::dims({batch_size, per_oc ? m_N : 1}), m_a_type, memory::format_tag::ab);
        postops.append_binary(algorithm::binary_mul, m_bin_md);
        bin_post_id = postops.len() - 1;
        return *this;
    }

    onednn_matmul& post_op_sum(float scale = 1.f, int32_t zero_point = 0) {
        postops.append_sum(scale, zero_point, m_a_type);
        return *this;
    }

    void create(const engine& aengine, const memory::desc& exist_wei_md = {}) {
        if (postops.len() > 0) {
            attr.set_post_ops(postops);
        }
        memory::desc src_md = memory::desc(memory::dims({m_M, m_K}), m_a_type, memory::format_tag::ab);
        memory::desc dst_md = memory::desc(memory::dims({m_M, m_N}), m_a_type, memory::format_tag::ab);

        if (m_use_ip) {
            bool use_exist_wei_md = exist_wei_md && exist_wei_md.get_ndims() > 0;
            memory::desc wei_md = use_exist_wei_md
                                      ? exist_wei_md
                                      : memory::desc(memory::dims({m_N, m_K}), m_w_type, memory::format_tag::any);
            auto ip_md = inner_product_forward::primitive_desc(aengine,
                                                               dnnl::prop_kind::forward_inference,
                                                               src_md,
                                                               wei_md,
                                                               dst_md,
                                                               attr);

            m_wei_md = ip_md.weights_desc();
            m_prim = inner_product_forward(ip_md);
        } else {
            // use fixed weight-layout to prevent shape-dependent weight-layout changes
            memory::desc wei_md = memory::desc(memory::dims({m_K, m_N}), m_w_type, memory::format_tag::ba);
            auto matmul_pd = matmul::primitive_desc(aengine, src_md, wei_md, dst_md, attr);
            m_wei_md = matmul_pd.weights_desc();
            m_prim = matmul(matmul_pd);
        }
    }

    void exec(const engine& aengine,
              const stream& strm,
              void* psrc,
              void* pdst,
              memory& weight,
              memory& scale,
              memory& zp,
              memory& bin_mem) const {
        memory src_mem(m_input_md, aengine, psrc);
        memory dst_mem(m_output_md, aengine, pdst);

        std::unordered_map<int, memory> args;
        args.insert({DNNL_ARG_SRC, src_mem});
        args.insert({DNNL_ARG_WEIGHTS, weight});
        args.insert({DNNL_ARG_DST, dst_mem});

        if (scale) {
            args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale});
        }
        if (zp) {
            args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp});
        }
        if (bin_mem) {
            args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(bin_post_id) | DNNL_ARG_SRC_1, bin_mem});
        }
        m_prim.execute(strm, args);
    }
};

struct MOE::Executor : MOE::ExecutorBase {
    int scratch_batch_size = 0;
    memory scratch_src;
    memory scratch_up;
    memory scratch_gate;
    memory scratch_down;
    memory scratch_router;

    std::vector<std::unique_ptr<std::atomic<int>>> expert_ntoks;
    PlainTensor scratch_expert_id;
    PlainTensor scratch_expert_weight;

    onednn_matmul linear_up;
    onednn_matmul linear_gate;
    onednn_matmul linear_down;

    struct nkernels {
        onednn_matmul up;
        onednn_matmul gate;
        onednn_matmul down;
    };

    std::unordered_map<int, nkernels> nkernels_cache;
    memory::data_type m_act_dtype;
    memory::data_type m_weight_dtype;
    int hidden_size = 0;
    int intermediate_size = 0;
    int expert_num = 0;
    int ic_group_size = 0;
    bool m_has_scale = false;
    bool m_has_zp = false;

    static int normalize_batch_size(int batch_size) {
        if (batch_size < 16) {
        } else if (batch_size < 512) {
            batch_size = rnd_up(batch_size, 16);
        } else if (batch_size < 1024) {
            batch_size = rnd_up(batch_size, 32);
        } else {
            batch_size = rnd_up(batch_size, 256);
        }
        return batch_size;
    }

    nkernels& get_nkernels(const dnnl::engine& engine, int batch_size) {
        auto it = nkernels_cache.find(batch_size);
        if (it != nkernels_cache.end()) {
            return it->second;
        }
        nkernels_cache.emplace(batch_size, nkernels());

        nkernels& nks = nkernels_cache[batch_size];

        nks.up.init(m_act_dtype,
                    m_weight_dtype,
                    batch_size,
                    hidden_size,
                    intermediate_size,
                    ic_group_size,
                    m_has_scale,
                    m_has_zp);
        nks.up.create(engine, linear_up.m_wei_md);

        nks.gate.init(m_act_dtype,
                      m_weight_dtype,
                      batch_size,
                      hidden_size,
                      intermediate_size,
                      ic_group_size,
                      m_has_scale,
                      m_has_zp);
        nks.gate.post_op_silu().post_op_bin_mul(true).create(engine, linear_gate.m_wei_md);

        nks.down.init(m_act_dtype,
                      m_weight_dtype,
                      batch_size,
                      intermediate_size,
                      hidden_size,
                      ic_group_size,
                      m_has_scale,
                      m_has_zp);
        nks.down.create(engine, linear_down.m_wei_md);
        return nks;
    }

    Executor(const dnnl::engine& engine,
             memory::data_type act_dtype,
             memory::data_type weight_dtype,
             int hidden_size,
             int intermediate_size,
             int ic_group_size,
             int expert_num,
             bool has_scale,
             bool has_zp)
        : m_act_dtype(act_dtype),
          m_weight_dtype(weight_dtype),
          hidden_size(hidden_size),
          intermediate_size(intermediate_size),
          expert_num(expert_num),
          ic_group_size(ic_group_size),
          m_has_scale(has_scale),
          m_has_zp(has_zp) {
        // onednn not support bf16+f16
        if (m_act_dtype == memory::data_type::bf16 && m_weight_dtype == memory::data_type::f16) {
            m_weight_dtype = memory::data_type::bf16;
        }
        // specialized 2nd-token kernel with more fused ops
        linear_up
            .init(m_act_dtype, m_weight_dtype, 1, hidden_size, intermediate_size, ic_group_size, m_has_scale, m_has_zp);
        linear_gate
            .init(m_act_dtype, m_weight_dtype, 1, hidden_size, intermediate_size, ic_group_size, m_has_scale, m_has_zp);
        linear_down
            .init(m_act_dtype, m_weight_dtype, 1, intermediate_size, hidden_size, ic_group_size, m_has_scale, m_has_zp);
        linear_up.create(engine);
        linear_gate.post_op_silu().post_op_bin_mul(true).create(engine);
        linear_down.post_op_bin_mul(false).post_op_sum().create(engine);

        scratch_src = memory({{scratch_batch_size, hidden_size}, m_act_dtype, memory::format_tag::ab}, engine);
        scratch_up = memory({{scratch_batch_size, intermediate_size}, m_act_dtype, memory::format_tag::ab}, engine);
        scratch_gate = memory({{scratch_batch_size, intermediate_size}, m_act_dtype, memory::format_tag::ab}, engine);
        scratch_down = memory({{scratch_batch_size, hidden_size}, m_act_dtype, memory::format_tag::ab}, engine);
        scratch_router = memory(linear_down.m_bin_md, engine);
        expert_ntoks.resize(expert_num);
        for (auto& p : expert_ntoks) {
            p = std::make_unique<std::atomic<int>>(0);
        }
    }

    void reorder_weights(const dnnl::engine& engine, ExpertWeights* pweight) override {
        stream s(engine);
        auto do_reorder = [&](Weight* pw, onednn_matmul& mm) {
            int oc = pw->oc;
            int ic = pw->ic;
            int n_groups = pw->ic / pw->qg_size;
            memory mem_raw_data({{oc, ic}, pw->raw_data_dtype, memory::format_tag::ab}, engine, pw->data);
            if (pw->mem_data && pw->mem_data.get_desc() == mm.m_wei_md) {
            } else if (mem_raw_data.get_desc() == mm.m_wei_md) {
                pw->mem_data = mem_raw_data;
            } else {
                pw->mem_data = memory(mm.m_wei_md, engine);
                dnnl::reorder(mem_raw_data, pw->mem_data).execute(s, mem_raw_data, pw->mem_data);
            }

            if (pw->scale) {
                memory mem_raw_scale({{oc, n_groups}, pw->raw_scale_dtype, memory::format_tag::ab}, engine, pw->scale);
                if (pw->mem_scale && pw->mem_scale.get_desc() == mm.m_sc_md) {
                } else if (mem_raw_scale.get_desc() == mm.m_sc_md) {
                    pw->mem_scale = mem_raw_scale;
                } else {
                    pw->mem_scale = memory(mm.m_sc_md, engine);
                    dnnl::reorder(mem_raw_scale, pw->mem_scale).execute(s, mem_raw_scale, pw->mem_scale);
                }
            }

            if (pw->zp) {
                if (pw->mem_zp && pw->mem_zp.get_desc() == mm.m_zp_md) {
                    return;
                }

                memory new_src({{pw->oc, n_groups}, memory::data_type::u8, memory::format_tag::ab}, engine);
                if (pw->raw_data_dtype == memory::data_type::u4) {
                    int cnt = n_groups * pw->oc;
                    auto* psrc0 = pw->zp;
                    auto* psrc1 = static_cast<int8_t*>(new_src.get_data_handle());
                    for (int i = 0; i < cnt; i += 2) {
                        auto v = *psrc0++;
                        *psrc1++ = v & 0xF;
                        *psrc1++ = v >> 4;
                    }
                } else {
                    OPENVINO_ASSERT(pw->raw_data_dtype == memory::data_type::u8,
                                    "Expected zp dtype is u8, current:",
                                    DnnlExtensionUtils::DataTypeToElementType(pw->raw_data_dtype));
                    new_src.set_data_handle(pw->zp);
                }
                pw->mem_zp = memory(mm.m_zp_md, engine);
                dnnl::reorder(new_src, pw->mem_zp).execute(s, new_src, pw->mem_zp);
            }
        };
        do_reorder(&pweight->up, linear_up);
        do_reorder(&pweight->gate, linear_gate);
        do_reorder(&pweight->down, linear_down);
        s.wait();
    }

    void execute(const dnnl::stream& strm, MOE* pnode) override {
        auto engine = pnode->getEngine();
        auto input_precisions = pnode->getInputPrecisions();
        auto output_precisions = pnode->getOutputPrecisions();
        auto dims_0 = pnode->getParentEdgeAt(0)->getMemory().getStaticDims();
        auto dims_1 = pnode->getParentEdgeAt(1)->getMemory().getStaticDims();
        auto dims_out = pnode->getChildEdgeAt(0)->getMemory().getStaticDims();

        auto act_dtype_size = memory::data_type_size(m_act_dtype);

        auto* hidden_states = pnode->getSrcDataAtPortAs<int8_t>(0);
        auto* router_weights = pnode->getSrcDataAtPortAs<float>(1);
        auto* final_hidden_states = pnode->getDstDataAtPortAs<int8_t>(0);
        auto n_tokens = dims_0[0];
        int n_states = dims_0[1];

        std::memset(final_hidden_states, 0, n_states * n_tokens * output_precisions[0].bitwidth() / 8);

        auto cmp_greater = [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
            return a.second > b.second;
        };

        // topk + softmax
        for (auto& v : expert_ntoks) {
            v->store(0);
        }
        scratch_expert_id.resize<int32_t>({n_tokens, pnode->m_config.topk});
        scratch_expert_weight.resize<float>({n_tokens, pnode->m_config.topk});
        {
            parallel_for(n_tokens, [&](int i) {
                auto* p_router_weights = router_weights + (i * pnode->m_config.expert_num);
                std::vector<std::pair<int, float>> max_heap(pnode->m_config.expert_num);
                for (size_t k = 0; k < pnode->m_config.expert_num; k++) {
                    max_heap[k].first = k;
                    max_heap[k].second = p_router_weights[k];
                }
                std::nth_element(max_heap.begin(),
                                 max_heap.begin() + pnode->m_config.topk - 1,
                                 max_heap.end(),
                                 cmp_greater);

                // softmax
                float denorm = 0;
                float vmax = max_heap[0].second;
                for (size_t k = 0; k < pnode->m_config.topk; k++) {
                    auto v = std::exp(max_heap[k].second - vmax);
                    denorm += v;
                    max_heap[k].second = v;
                }

                denorm = 1.0f / denorm;
                auto* p_expert_id = scratch_expert_id.ptr<int32_t>(i, 0);
                auto* p_expert_weight = scratch_expert_weight.ptr<float>(i, 0);
                for (size_t k = 0; k < pnode->m_config.topk; k++) {
                    auto expert_id = max_heap[k].first;
                    p_expert_id[k] = expert_id;
                    p_expert_weight[k] = max_heap[k].second * denorm;
                    expert_ntoks[expert_id]->fetch_add(1);
                }
            });
        }

        memory empty_mem;
        if (n_tokens > 1) {
            // gather for expert i
            // allocte scratch memory for gate/up/down
            int max_num_toks = 0;
            for (int i = 0; i < expert_num; i++) {
                auto num_tokens = expert_ntoks[i]->load();
                max_num_toks = std::max(max_num_toks, num_tokens);
            }
            auto max_batch_size = normalize_batch_size(max_num_toks);
            if (scratch_batch_size < max_batch_size) {
                scratch_batch_size = max_batch_size;
                scratch_src = memory({{scratch_batch_size, hidden_size}, m_act_dtype, memory::format_tag::ab}, engine);
                scratch_up =
                    memory({{scratch_batch_size, intermediate_size}, m_act_dtype, memory::format_tag::ab}, engine);
                scratch_gate =
                    memory({{scratch_batch_size, intermediate_size}, m_act_dtype, memory::format_tag::ab}, engine);
                scratch_down = memory({{scratch_batch_size, hidden_size}, m_act_dtype, memory::format_tag::ab}, engine);
            }

            std::vector<int> token_ids(n_tokens);
            std::vector<float> token_weights(n_tokens);

            for (int expert_id = 0; expert_id < expert_num; expert_id++) {
                auto num_tokens = expert_ntoks[expert_id]->load();
                if (num_tokens == 0) {
                    continue;
                }
                // to limit cache size, avoid too many different kernels due to batch-size change
                auto expert_batch_size = normalize_batch_size(num_tokens);
                auto& nks = get_nkernels(engine, expert_batch_size);

                // gather from all tokens
                std::atomic<int> slot(0);
                parallel_for(n_tokens, [&](int i) {
                    auto* p_expert_ids = scratch_expert_id.ptr<int32_t>(i, 0);
                    for (size_t k = 0; k < pnode->m_config.topk; k++) {
                        if (p_expert_ids[k] == expert_id) {
                            auto r_weight = *scratch_expert_weight.ptr<float>(i, k);
                            auto seq = slot.fetch_add(1);
                            auto* dst = reinterpret_cast<int8_t*>(scratch_src.get_data_handle()) +
                                        (seq * n_states * act_dtype_size);
                            // gather hidden_states
                            std::memcpy(dst,
                                        hidden_states + (i * n_states * act_dtype_size),
                                        n_states * act_dtype_size);
                            // gather weights
                            token_weights[seq] = r_weight;
                            token_ids[seq] = i;
                            break;
                        }
                    }
                });

                // call kernels
                auto& up = pnode->m_weights[expert_id].up;
                auto& gate = pnode->m_weights[expert_id].gate;
                auto& down = pnode->m_weights[expert_id].down;

                nks.up.exec(engine,
                            strm,
                            scratch_src.get_data_handle(),
                            scratch_up.get_data_handle(),
                            up.mem_data,
                            up.mem_scale,
                            up.mem_zp,
                            empty_mem);
                nks.gate.exec(engine,
                              strm,
                              scratch_src.get_data_handle(),
                              scratch_gate.get_data_handle(),
                              gate.mem_data,
                              gate.mem_scale,
                              gate.mem_zp,
                              scratch_up);
                nks.down.exec(engine,
                              strm,
                              scratch_gate.get_data_handle(),
                              scratch_down.get_data_handle(),
                              down.mem_data,
                              down.mem_scale,
                              down.mem_zp,
                              empty_mem);

                parallel_for(num_tokens, [&](int seq) {
                    auto i = token_ids[seq];
                    float w = token_weights[seq];
                    switch (m_act_dtype) {
                    case memory::data_type::bf16: {
                        auto* src = reinterpret_cast<ov::bfloat16*>(scratch_down.get_data_handle()) + seq * n_states;
                        auto* dst = reinterpret_cast<ov::bfloat16*>(final_hidden_states) + i * n_states;
                        for (int k = 0; k < n_states; k++) {
                            dst[k] += src[k] * w;
                        }
                    } break;
                    case memory::data_type::f16: {
                        auto* src = reinterpret_cast<ov::float16*>(scratch_down.get_data_handle()) + seq * n_states;
                        auto* dst = reinterpret_cast<ov::float16*>(final_hidden_states) + i * n_states;
                        for (int k = 0; k < n_states; k++) {
                            dst[k] += src[k] * w;
                        }
                    } break;
                    case memory::data_type::f32: {
                        auto* src = reinterpret_cast<float*>(scratch_down.get_data_handle()) + seq * n_states;
                        auto* dst = reinterpret_cast<float*>(final_hidden_states) + i * n_states;
                        for (int k = 0; k < n_states; k++) {
                            dst[k] += src[k] * w;
                        }
                    } break;
                    default:
                        OPENVINO_ASSERT("Unsupport act type in MOE: ", static_cast<int>(m_act_dtype));
                        break;
                    }
                });
            }
        } else {
            for (size_t i = 0; i < n_tokens; i++) {
                auto* router_w = reinterpret_cast<float*>(scratch_router.get_data_handle());
                auto* p_expert_id = scratch_expert_id.ptr<int32_t>(i, 0);
                auto* p_expert_weight = scratch_expert_weight.ptr<float>(i, 0);

                for (size_t k = 0; k < pnode->m_config.topk; k++) {
                    auto expert_id = p_expert_id[k];
                    auto expert_weight = p_expert_weight[k];
                    if (router_w) {
                        switch (m_act_dtype) {
                        case memory::data_type::bf16:
                            (reinterpret_cast<ov::bfloat16*>(router_w))[0] = expert_weight;
                            break;
                        case memory::data_type::f16:
                            (reinterpret_cast<ov::float16*>(router_w))[0] = expert_weight;
                            break;
                        case memory::data_type::f32:
                            router_w[0] = expert_weight;
                            break;
                        default:
                            OPENVINO_ASSERT("Unsupport act type in MOE:", static_cast<int>(m_act_dtype));
                            break;
                        }
                    }
                    auto& up = pnode->m_weights[expert_id].up;
                    auto& gate = pnode->m_weights[expert_id].gate;
                    auto& down = pnode->m_weights[expert_id].down;

                    linear_up.exec(engine,
                                   strm,
                                   hidden_states,
                                   scratch_up.get_data_handle(),
                                   up.mem_data,
                                   up.mem_scale,
                                   up.mem_zp,
                                   empty_mem);
                    linear_gate.exec(engine,
                                     strm,
                                     hidden_states,
                                     scratch_gate.get_data_handle(),
                                     gate.mem_data,
                                     gate.mem_scale,
                                     gate.mem_zp,
                                     scratch_up);
                    linear_down.exec(engine,
                                     strm,
                                     scratch_gate.get_data_handle(),
                                     final_hidden_states,
                                     down.mem_data,
                                     down.mem_scale,
                                     down.mem_zp,
                                     scratch_router);
                }

                router_weights += pnode->m_config.expert_num;
                final_hidden_states += n_states * act_dtype_size;
                hidden_states += n_states * act_dtype_size;
            }
        }
    };
};

bool MOE::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<ov::op::internal::MOE>(op)) {
            errorMessage = "Unknown MOE operation : " + std::string(op->get_type_info().name) + " with name '" +
                           op->get_friendly_name() + "'";
        }
    } catch (...) {
        return false;
    }
    return true;
}

MOE::MOE(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, PassThroughShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    m_moe_op = ov::as_type_ptr<ov::op::internal::MOE>(op);
    CPU_NODE_ASSERT(m_moe_op,
                    "Attempt to create MOE node from an invalid op type: ",
                    op,
                    " with name ",
                    op->get_friendly_name());

    m_config = m_moe_op->get_config();
    extractConsts();
}

void MOE::extractConsts() {
    const auto& consts = m_moe_op->get_consts();
    OPENVINO_ASSERT(m_config.expert_num == consts.size());
    m_weights.resize(consts.size());

    auto extract_consts = [](const std::shared_ptr<ov::op::v0::Constant>& op,
                             const std::shared_ptr<ov::op::v0::Constant>& sc,
                             const std::shared_ptr<ov::op::v0::Constant>& zp,
                             Weight& wei) {
        auto shape = op->get_output_shape(0);
        auto eletype = op->get_output_element_type(0);
        int oc = shape[0];
        OPENVINO_ASSERT(shape.size() == 2 || shape.size() == 3);
        int ic = shape[1], qg_size = shape[1];
        if (shape.size() == 3) {
            qg_size = shape[2];
            ic = shape[2] * shape[1];
        }
        OPENVINO_ASSERT(op, "Weight must not be null");
        wei.data = const_cast<uint8_t*>(op->get_data_ptr<uint8_t>());
        wei.ic = ic;
        wei.oc = oc;
        wei.qg_size = qg_size;
        wei.raw_data_dtype = DnnlExtensionUtils::ElementTypeToDataType(eletype);

        if (sc) {
            wei.scale = const_cast<uint8_t*>(sc->get_data_ptr<uint8_t>());
            wei.raw_scale_dtype = DnnlExtensionUtils::ElementTypeToDataType(sc->get_output_element_type(0));
        }

        if (zp) {
            wei.zp = const_cast<uint8_t*>(zp->get_data_ptr<uint8_t>());
            wei.raw_zp_dtype = DnnlExtensionUtils::ElementTypeToDataType(zp->get_output_element_type(0));
        }
    };

    for (size_t i = 0; i < consts.size(); i++) {
        auto expert_consts = consts[i];
        extract_consts(expert_consts.gates[0], expert_consts.gates[1], expert_consts.gates[2], m_weights[i].gate);
        extract_consts(expert_consts.ups[0], expert_consts.ups[1], expert_consts.ups[2], m_weights[i].up);
        extract_consts(expert_consts.downs[0], expert_consts.downs[1], expert_consts.downs[2], m_weights[i].down);
    }
}

void MOE::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    auto input_precision0 = getOriginalInputPrecisionAtPort(0);
    auto output_precision = getOriginalOutputPrecisionAtPort(0);

    std::vector<PortConfigurator> inPortConfigs;
    std::vector<PortConfigurator> outPortConfigs;

    inPortConfigs.emplace_back(LayoutType::ncsp, input_precision0, getInputShapeAtPort(0), false, -1);  // hidden_states
    inPortConfigs.emplace_back(LayoutType::ncsp,
                               ov::element::f32,
                               getInputShapeAtPort(1),
                               false,
                               -1);  // router-weights
    outPortConfigs.emplace_back(LayoutType::ncsp,
                                output_precision,
                                getOutputShapeAtPort(0),
                                false,
                                -1);  // final_hidden_states

    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);
}

void MOE::prepareParams() {
    auto input_precisions = getInputPrecisions();

    MoEKey key = {DnnlExtensionUtils::ElementTypeToDataType(input_precisions[0], DnnlExtensionUtils::throw_tag()),
                  DnnlExtensionUtils::ElementTypeToDataType(m_config.weight_type),
                  static_cast<int>(m_config.hidden_size),
                  m_weights[0].gate.oc,
                  static_cast<int>(m_config.group_size),
                  m_config.expert_num,
                  m_weights[0].up.scale != nullptr,
                  m_weights[0].up.zp != nullptr};
    auto engine = getEngine();
    auto builder = [&engine](const MoEKey& key) {
        return std::make_shared<Executor>(engine,
                                          key.act_dtype,
                                          key.weight_dtype,
                                          key.hidden_size,
                                          key.intermediate_size,
                                          key.ic_group_size,
                                          key.expert_num,
                                          key.has_scale,
                                          key.has_zp);
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);
    if (!result.first) {
        OPENVINO_THROW("Primitive descriptor was not found for node ", getName(), ".");
    }

    if (m_executor != result.first) {
        m_executor = result.first;
        for (auto& weight : m_weights) {
            m_executor->reorder_weights(getEngine(), &weight);
        }
    }
}

void MOE::execute(const dnnl::stream& strm) {
    m_executor->execute(strm, this);
}

void MOE::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

}  // namespace ov::intel_cpu::node