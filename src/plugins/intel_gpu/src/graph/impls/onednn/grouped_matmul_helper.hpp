// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// Inline dnnl::matmul wrappers + a construction-cache, shared by primitives that
// orchestrate their own sort/gather/gemm/scatter pipelines.

#ifdef ENABLE_ONEDNN_FOR_GPU

#    include <array>
#    include <memory>
#    include <mutex>
#    include <oneapi/dnnl/dnnl.hpp>
#    include <tuple>
#    include <unordered_map>

#    include "impls/onednn/utils.hpp"
#    include "intel_gpu/runtime/itt.hpp"
#    include "intel_gpu/runtime/memory.hpp"
#    include "intel_gpu/runtime/stream.hpp"
#    include "intel_gpu/runtime/utils.hpp"
#    include "openvino/core/except.hpp"

namespace cldnn {
namespace onednn {

struct onednn_matmul {
    dnnl::matmul m_prim;
    dnnl::memory::desc m_wei_md;
    dnnl::memory::data_type m_w_type;
    dnnl::memory::data_type m_a_type;  // activation dtype
    dnnl::memory::dim m_K;
    dnnl::memory::dim m_N;
    dnnl::memory::dim m_M;
    dnnl::memory::dim m_K_groups;

    dnnl::primitive_attr attr;
    dnnl::post_ops postops;

    // zp_dtype == undef means no weight zero-point
    onednn_matmul(dnnl::memory::data_type act_dtype,
                  dnnl::memory::data_type weight_dtype,
                  int batch_size,
                  int ic,
                  int oc,
                  int ic_group_size = -1,
                  dnnl::memory::data_type zp_dtype = dnnl::memory::data_type::undef,
                  bool common_zp = false) {
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
        if (ic_group_size >= 0) {
            w_scale(ic_group_size);
            if (zp_dtype != dnnl::memory::data_type::undef) {
                w_zp(ic_group_size, zp_dtype, common_zp);
            }
            fpmath_f16();
        }
    }

    onednn_matmul& w_scale(int k_group_size) {
        if (k_group_size <= 0) {
            m_K_groups = 1;
            // per-OC, no grouping in K dimension
            attr.set_scales(DNNL_ARG_WEIGHTS, (0 << 0) + (1 << 1), {1}, dnnl::memory::data_type::f16);
        } else {
            OPENVINO_ASSERT((k_group_size % 32) == 0);
            OPENVINO_ASSERT((m_K % k_group_size) == 0);
            m_K_groups = m_K / k_group_size;
            attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {k_group_size, 1}, dnnl::memory::data_type::f16);
        }
        return *this;
    }

    onednn_matmul& w_zp(int k_group_size, dnnl::memory::data_type zp_dtype, bool common_zp = false) {
        if (common_zp) {
            // One zero point for the whole weights tensor, while the scales stay grouped
            attr.set_zero_points(DNNL_ARG_WEIGHTS, 0, {}, zp_dtype);
            return *this;
        }
        if (k_group_size <= 0) {
            OPENVINO_ASSERT(m_K_groups == 1);
            attr.set_zero_points(DNNL_ARG_WEIGHTS, (0 << 0) + (1 << 1), {1}, zp_dtype);
        } else {
            OPENVINO_ASSERT((m_K % k_group_size) == 0);
            m_K_groups = (m_K / k_group_size);
            attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {k_group_size, 1}, zp_dtype);
        }
        return *this;
    }

    onednn_matmul& fpmath_f16() {
        attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
        return *this;
    }
    onednn_matmul& post_op_gate_activation(dnnl::algorithm algo) {
        float alpha = 1.0f;
        float beta = 0.0f;
        postops.append_eltwise(algo, alpha, beta);
        return *this;
    }
    onednn_matmul& post_op_bin_mul(bool per_oc = true) {
        dnnl::memory::dim batch_size = m_M;
        if (batch_size == DNNL_RUNTIME_DIM_VAL) {
            batch_size = 1024 * 1024;  // big enough fake static batch
        }

        dnnl::memory::desc bin_mul_md = dnnl::memory::desc(dnnl::memory::dims({batch_size, per_oc ? m_N : 1}), m_a_type, dnnl::memory::format_tag::ab);
        postops.append_binary(dnnl::algorithm::binary_mul, bin_mul_md);
        return *this;
    }

    onednn_matmul& post_op_sum(float scale = 1.f, int32_t zero_point = 0) {
        postops.append_sum(scale, zero_point, dnnl::memory::data_type::undef);
        return *this;
    }

    void create(dnnl::engine eng) {
        if (postops.len() > 0) {
            attr.set_post_ops(postops);
        }

        dnnl::memory::desc src_md = dnnl::memory::desc(dnnl::memory::dims({m_M, m_K}), m_a_type, dnnl::memory::format_tag::ab);
        dnnl::memory::desc dst_md = dnnl::memory::desc(dnnl::memory::dims({m_M, m_N}), m_a_type, dnnl::memory::format_tag::ab);

        // use fixed weight-layout to prevent shape-dependent weight-layout changes
        dnnl::memory::desc wei_md = dnnl::memory::desc(dnnl::memory::dims({m_K, m_N}), m_w_type, dnnl::memory::format_tag::ba);

        // Create primitive descriptor.
        auto matmul_pd = dnnl::matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr);

        // Pre-packed weights stored as int8_t
        m_wei_md = matmul_pd.weights_desc();

        // Create the primitive.
        m_prim = dnnl::matmul(matmul_pd);
    }

    // this creator is for predefined matmul primitive types
    enum class type {
        none,
        with_bin_mul,
        with_bin_mul_per_row,
        with_bin_mul_per_row_sum,
        with_gate_act,
        with_gate_act_bin_mul,
        with_sigmoid,
        with_bin_mul_sum,
    };
    int bin_post_id = -1;
    bool bin_per_row = false;
    onednn_matmul(dnnl::engine eng,
                  dnnl::memory::data_type act_dtype,
                  dnnl::memory::data_type weight_dtype,
                  int batch,
                  int ic,
                  int oc,
                  int ic_group_size,
                  type t,
                  dnnl::memory::data_type zp_dtype = dnnl::memory::data_type::undef,
                  dnnl::algorithm activation_algo = dnnl::algorithm::eltwise_swish,
                  bool common_zp = false)
        : onednn_matmul(act_dtype, weight_dtype, batch, ic, oc, ic_group_size, zp_dtype, common_zp) {
        if (t == type::with_bin_mul) {
            bin_post_id = 0;
            post_op_bin_mul(true);
        }
        if (t == type::with_bin_mul_sum) {
            bin_post_id = 0;
            post_op_bin_mul(false);
            post_op_sum();
        }
        if (t == type::with_sigmoid) {
            postops.append_eltwise(dnnl::algorithm::eltwise_logistic, 1.0f, 0.0f);
        }
        if (t == type::with_bin_mul_per_row) {
            bin_post_id = 0;
            bin_per_row = true;
            post_op_bin_mul(false);
        }
        if (t == type::with_bin_mul_per_row_sum) {
            bin_post_id = 0;
            bin_per_row = true;
            post_op_bin_mul(false);
            post_op_sum();
        }
        if (t == type::with_gate_act) {
            post_op_gate_activation(activation_algo);
        }
        if (t == type::with_gate_act_bin_mul) {
            bin_post_id = 1;
            post_op_gate_activation(activation_algo);
            post_op_bin_mul(true);
        }

        create(eng);
    }
};

// Stateless functor wrapping a built dnnl primitive; cache returns shared_ptr<const T> for thread safety.
template <typename... TTypes>
class tuple_hasher {
private:
    using Tuple = std::tuple<TTypes...>;
    template <int N>
    size_t hash(Tuple& value) const {
        return 0;
    }
    template <int N, typename THead, typename... TTail>
    size_t hash(Tuple& value) const {
        constexpr int Index = N - sizeof...(TTail) - 1;
        return std::hash<THead>()(std::get<Index>(value)) ^ hash<N, TTail...>(value);
    }

public:
    size_t operator()(Tuple value) const {
        auto hv = hash<sizeof...(TTypes), TTypes...>(value);
        return hv;
    }
};

// Process-wide cache keyed by constructor args; same-shape callers across TUs reuse one instance.
template <class T, typename... CArgs>
std::shared_ptr<const T> make_cacheable(dnnl::engine eng, CArgs... cargs) {
    std::shared_ptr<const T> sptr;
    auto key = std::make_tuple(cargs...);
    static std::unordered_map<decltype(key), std::weak_ptr<const T>, tuple_hasher<CArgs...>> cache;
    static std::mutex mutex;
    std::lock_guard<std::mutex> guard(mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        auto& wptr = it->second;
        sptr = wptr.lock();
        if (!sptr) {
            sptr = std::make_shared<T>(eng, cargs...);
            wptr = sptr;
        }
    } else {
        sptr = std::make_shared<T>(eng, cargs...);
        cache.emplace(std::make_pair(key, std::weak_ptr<const T>(sptr)));
    }
    return sptr;
}

struct onednn_linear {
    std::shared_ptr<const onednn_matmul> mm;
    dnnl::memory weight;
    dnnl::memory scale;
    dnnl::memory zp;
    dnnl::matmul m_prim;
    dnnl::memory::dim m_K;
    dnnl::memory::dim m_N;
    dnnl::memory::dim m_batch;
    dnnl::memory::data_type m_a_type;
    int bin_post_id;

    static onednn_linear create(dnnl::engine eng,
                                dnnl::memory::data_type act_dtype,
                                dnnl::memory::data_type weight_dtype,
                                int batch,
                                int ic,
                                int oc,
                                int ic_group_size,
                                onednn_matmul::type t,
                                dnnl::memory weight,  // external weight
                                dnnl::memory scale,
                                dnnl::memory zp,
                                dnnl::algorithm activation_algo = dnnl::algorithm::eltwise_swish) {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("onednn_linear::create()"));
        // The zp dtype may differ from the weight dtype (u3 weights have u8 zp)
        const auto zp_dtype = zp ? zp.get_desc().get_data_type() : dnnl::memory::data_type::undef;
        const bool common_zp = zp && zp.get_desc().get_dims() == dnnl::memory::dims{1};
        auto mm = make_cacheable<onednn_matmul>(eng, act_dtype, weight_dtype, batch, ic, oc, ic_group_size, t, zp_dtype, activation_algo, common_zp);
        onednn_linear linear;
        linear.mm = mm;
        linear.bin_post_id = mm->bin_post_id;
        linear.m_prim = mm->m_prim;
        linear.m_K = mm->m_K;
        linear.m_N = mm->m_N;
        linear.m_batch = batch;
        linear.m_a_type = mm->m_a_type;
        linear.weight = weight;

        if (scale) {
            // https://uxlfoundation.github.io/oneDNN/page_weights_decompression_matmul_cpp.html
            // Quantization Group size for scales. Must be divisible by 32.
            linear.scale = scale;
            if (zp) {
                linear.zp = zp;
            }
        }
        return linear;
    }

    void forward(dnnl::stream& stream, int m, dnnl::memory src_mem, dnnl::memory dst_mem, dnnl::memory bin_mem) {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("onednn_linear::forward()"));
        dnnl::memory::dim M = m;

        if (m_batch != 0 && m_batch != M) {
            OPENVINO_THROW("onednn_linear::forward(): invalid batch size m_batch=", m_batch, " M=", M);
        }

        std::unordered_map<int, dnnl::memory> args;
        args.insert({DNNL_ARG_SRC, src_mem});
        args.insert({DNNL_ARG_WEIGHTS, weight});
        // args.insert({DNNL_ARG_BIAS, bias_mem});
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
        m_prim.execute(stream, args);
    }
};

// Weights decompression of a grouped matmul [T, K] x W[G, K, N] -> [T, N], W stored as [G, N, K] (acb).
// Scales and zero points are [G, K / group_size, N] (abc), or [G, N] (ab) without K groups.
struct grouped_matmul_quant {
    dnnl::memory::data_type scale_dt = dnnl::memory::data_type::undef;  // undef: uncompressed weights
    dnnl::memory::data_type zp_dt = dnnl::memory::data_type::undef;     // undef: no zero point
    dnnl::memory::dim group_size = 0;                                   // K group size; <= 0 or >= K: per output channel
    bool host_zp = false;                                               // one zero point for all the experts, passed as a host scalar

    bool compressed() const {
        return scale_dt != dnnl::memory::data_type::undef;
    }
    bool has_zp() const {
        return zp_dt != dnnl::memory::data_type::undef;
    }
    bool device_zp() const {
        return has_zp() && !host_zp;
    }
    bool has_k_groups(dnnl::memory::dim K) const {
        return group_size > 0 && group_size < K;
    }
};

// Weights mask bits of the logical [G, K, N] grouped weights
constexpr int grouped_wei_mask_g = 1 << 0;
constexpr int grouped_wei_mask_k = 1 << 1;
constexpr int grouped_wei_mask_n = 1 << 2;

inline void set_grouped_matmul_quant_attr(dnnl::primitive_attr& attr, const grouped_matmul_quant& q, dnnl::memory::dim K) {
    if (!q.compressed()) {
        return;
    }
    const bool k_groups = q.has_k_groups(K);
    const int mask = grouped_wei_mask_g | grouped_wei_mask_n | (k_groups ? grouped_wei_mask_k : 0);
    const dnnl::memory::dims groups = k_groups ? dnnl::memory::dims{q.group_size, 1} : dnnl::memory::dims{};
    if (k_groups) {
        OPENVINO_ASSERT(K % q.group_size == 0, "[GPU] grouped matmul: group size ", q.group_size, " does not divide K=", K);
    }
    attr.set_scales(DNNL_ARG_WEIGHTS, mask, groups, q.scale_dt);
    if (q.host_zp) {
        attr.set_host_zero_point(DNNL_ARG_WEIGHTS, q.zp_dt);
    } else if (q.has_zp()) {
        attr.set_zero_points(DNNL_ARG_WEIGHTS, mask, groups, q.zp_dt);
    }
}

inline dnnl::memory::desc make_grouped_matmul_quant_md(dnnl::memory::dim G,
                                                       dnnl::memory::dim K,
                                                       dnnl::memory::dim N,
                                                       const grouped_matmul_quant& q,
                                                       dnnl::memory::data_type dt) {
    if (q.has_k_groups(K)) {
        return dnnl::memory::desc({G, K / q.group_size, N}, dt, dnnl::memory::format_tag::abc);
    }
    return dnnl::memory::desc({G, N}, dt, dnnl::memory::format_tag::ab);
}

// src [T, K] and dst [T, N] are grouped along axis 0 by s32 end-offsets
inline dnnl::matmul::primitive_desc make_grouped_matmul_pd(const dnnl::engine& engine,
                                                           dnnl::memory::dim total_tokens,
                                                           dnnl::memory::dim K,
                                                           dnnl::memory::dim N,
                                                           dnnl::memory::dim G,
                                                           dnnl::memory::data_type src_dt,
                                                           dnnl::memory::data_type wei_dt,
                                                           dnnl::memory::data_type dst_dt,
                                                           const dnnl::primitive_attr& attr) {
    auto src_md = dnnl::memory::desc::grouped({total_tokens, K}, src_dt, 0, G, dnnl::memory::data_type::s32);
    auto dst_md = dnnl::memory::desc::grouped({total_tokens, N}, dst_dt, 0, G, dnnl::memory::data_type::s32);
    auto wei_md = dnnl::memory::desc({G, K, N}, wei_dt, dnnl::memory::format_tag::acb);
    return dnnl::matmul::primitive_desc(engine, src_md, wei_md, dst_md, attr);
}

// One grouped matmul primitive with its decompression descriptors
struct grouped_matmul_kernel {
    dnnl::matmul::primitive_desc pd;
    dnnl::matmul prim;
    grouped_matmul_quant quant;
    dnnl::memory::desc scale_md;
    dnnl::memory::desc zp_md;
    int bin_po_idx = -1;  // binary post-op index, its src_1 is grouped like dst

    grouped_matmul_kernel() = default;
    grouped_matmul_kernel(const dnnl::engine& engine,
                          dnnl::memory::dim total_tokens,
                          dnnl::memory::dim K,
                          dnnl::memory::dim N,
                          dnnl::memory::dim G,
                          dnnl::memory::data_type act_dt,
                          dnnl::memory::data_type wei_dt,
                          const grouped_matmul_quant& q,
                          const dnnl::post_ops& post_ops = dnnl::post_ops(),
                          int binary_post_op_idx = -1)
        : quant(q),
          bin_po_idx(binary_post_op_idx) {
        dnnl::primitive_attr attr;
        if (q.compressed()) {
            attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
        }
        attr.set_post_ops(post_ops);
        set_grouped_matmul_quant_attr(attr, q, K);
        pd = make_grouped_matmul_pd(engine, total_tokens, K, N, G, act_dt, wei_dt, act_dt, attr);
        prim = dnnl::matmul(pd);
        if (q.compressed()) {
            scale_md = make_grouped_matmul_quant_md(G, K, N, q, q.scale_dt);
        }
        if (q.device_zp()) {
            zp_md = make_grouped_matmul_quant_md(G, K, N, q, q.zp_dt);
        }
    }
};

// Execute args of one grouped matmul. Building them costs several dnnl::memory objects (each one queries the USM
// pointer type) per execute, so they are reused while all the buffer addresses they wrap stay the same.
class grouped_matmul_args {
public:
    struct buffers {
        const cldnn::memory* src = nullptr;
        const cldnn::memory* dst = nullptr;
        const cldnn::memory* offsets = nullptr;
        const cldnn::memory* weights = nullptr;
        const cldnn::memory* scale = nullptr;
        const cldnn::memory* zp = nullptr;       // device zero point
        const cldnn::memory* bin_src = nullptr;  // src_1 of the binary post-op
    };

    // host_zp and hint are host scalars, passed when non-empty
    const std::unordered_map<int, dnnl::memory>& get(const grouped_matmul_kernel& k,
                                                    const buffers& b,
                                                    const dnnl::memory& host_zp = dnnl::memory(),
                                                    const dnnl::memory& hint = dnnl::memory()) {
        auto ptr = [](const cldnn::memory* m) -> const void* {
            return m ? m->buffer_ptr() : nullptr;
        };
        const std::array<const void*, 7> key{ptr(b.src), ptr(b.dst), ptr(b.offsets), ptr(b.weights), ptr(b.scale), ptr(b.zp), ptr(b.bin_src)};
        if (!m_args.empty() && key == m_key) {
            return m_args;
        }
        m_args = {{DNNL_ARG_SRC, b.src->get_onednn_grouped_memory(k.pd.src_desc(), *b.offsets)},
                  {DNNL_ARG_WEIGHTS, b.weights->get_onednn_memory(k.pd.weights_desc())},
                  {DNNL_ARG_DST, b.dst->get_onednn_grouped_memory(k.pd.dst_desc(), *b.offsets)}};
        if (k.quant.compressed()) {
            m_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, b.scale->get_onednn_memory(k.scale_md)});
        }
        if (k.quant.device_zp()) {
            m_args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, b.zp->get_onednn_memory(k.zp_md)});
        } else if (k.quant.host_zp) {
            OPENVINO_ASSERT(host_zp, "[GPU] grouped matmul: host zero point is not set");
            m_args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, host_zp});
        }
        if (b.bin_src) {
            OPENVINO_ASSERT(k.bin_po_idx >= 0, "[GPU] grouped matmul: binary post-op index is not set");
            m_args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(k.bin_po_idx) | DNNL_ARG_SRC_1, b.bin_src->get_onednn_grouped_memory(k.pd.dst_desc(), *b.offsets)});
        }
        if (hint) {
            m_args.insert({DNNL_ARG_HINT_MAX_GROUP_SIZE, hint});
        }
        m_key = key;
        return m_args;
    }

private:
    std::array<const void*, 7> m_key{};
    std::unordered_map<int, dnnl::memory> m_args;
};

// DNNL_ARG_HINT_MAX_GROUP_SIZE: max rows of one group. Read on the host at execute(), so it can be updated in place
// with set_host_scalar_value() between executes.
inline dnnl::memory make_grouped_matmul_hint(int32_t max_group_size = 0) {
    return dnnl::memory(dnnl::memory::desc::host_scalar(dnnl::memory::data_type::s32), max_group_size);
}

// A single zero point shared by all the experts, as a OneDNN host scalar
inline dnnl::memory make_host_scalar_zp(const cldnn::memory::ptr& zp_mem, cldnn::stream& stream) {
    const auto dt = zp_mem->get_layout().data_type;
    OPENVINO_ASSERT(cldnn::one_of(dt, {cldnn::data_types::u8, cldnn::data_types::i8}), "[GPU] unsupported host scalar zero point type ", dt);
    cldnn::mem_lock<uint8_t, cldnn::mem_lock_type::read> lock(zp_mem, stream);
    const auto md = dnnl::memory::desc::host_scalar(convert_data_type(dt));
    return dt == cldnn::data_types::u8 ? dnnl::memory(md, lock[0]) : dnnl::memory(md, static_cast<int8_t>(lock[0]));
}

}  // namespace onednn
}  // namespace cldnn

#endif  // ENABLE_ONEDNN_FOR_GPU
