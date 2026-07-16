// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// Inline dnnl::matmul wrappers + a construction-cache, shared by primitives that
// orchestrate their own sort/gather/gemm/scatter pipelines.

#ifdef ENABLE_ONEDNN_FOR_GPU

#    include <memory>
#    include <mutex>
#    include <oneapi/dnnl/dnnl.hpp>
#    include <tuple>
#    include <unordered_map>

#    include "intel_gpu/runtime/itt.hpp"
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

    onednn_matmul(dnnl::memory::data_type act_dtype,
                  dnnl::memory::data_type weight_dtype,
                  int batch_size,
                  int ic,
                  int oc,
                  int ic_group_size = -1,
                  bool has_zp = true) {
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
            if (has_zp)
                w_zp(ic_group_size);
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

    onednn_matmul& w_zp(int k_group_size) {
        if (k_group_size <= 0) {
            OPENVINO_ASSERT(m_K_groups == 1);
            attr.set_zero_points(DNNL_ARG_WEIGHTS, (0 << 0) + (1 << 1), {1}, m_w_type);
        } else {
            OPENVINO_ASSERT((m_K % k_group_size) == 0);
            m_K_groups = (m_K / k_group_size);
            attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {k_group_size, 1}, m_w_type);
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
        if (batch_size == DNNL_RUNTIME_DIM_VAL)
            batch_size = 1024 * 1024;  // big enough fake static batch

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
                  bool has_zp = true,
                  dnnl::algorithm activation_algo = dnnl::algorithm::eltwise_swish)
        : onednn_matmul(act_dtype, weight_dtype, batch, ic, oc, ic_group_size, has_zp) {
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
        if (t == type::with_gate_act)
            post_op_gate_activation(activation_algo);
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
    typedef std::tuple<TTypes...> Tuple;
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
        bool has_zp = static_cast<bool>(zp);
        auto mm = make_cacheable<onednn_matmul>(eng, act_dtype, weight_dtype, batch, ic, oc, ic_group_size, t, has_zp, activation_algo);
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

        if (!(m_batch == 0 || m_batch == M)) {
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

}  // namespace onednn
}  // namespace cldnn

#endif  // ENABLE_ONEDNN_FOR_GPU
