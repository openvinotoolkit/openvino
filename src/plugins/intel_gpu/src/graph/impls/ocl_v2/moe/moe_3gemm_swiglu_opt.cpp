// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#include "moe_3gemm_gen_micro.hpp"
#include "moe_3gemm_swiglu_opt.hpp"
// clang-format on

#define DEBUG_MOE_LOG 0

#ifdef ENABLE_ONEDNN_FOR_GPU
#    include <initializer_list>
#    include <oneapi/dnnl/dnnl.hpp>
#    include <oneapi/dnnl/dnnl_ocl.hpp>
#    include <sstream>
#    include <string_view>
#    include <tuple>
#    include <utility>

#    include "../primitive_ocl_base.hpp"
#    include "../utils/kernel_generator.hpp"
#    include "common_utils/jitter.hpp"
#    include "debug_helper.hpp"
#    include "intel_gpu/graph/kernel_impl_params.hpp"
#    include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#    include "intel_gpu/runtime/lru_cache.hpp"
#    include "intel_gpu/runtime/stream.hpp"
#    include "intel_gpu/runtime/utils.hpp"
#    include "moe_3gemm_fused_inst.h"
#    include "moe_3gemm_gen_micro.hpp"
#    include "ocl_v2/utils/fused_ops_jitter.hpp"
#    include "ocl_v2/utils/jitter.hpp"
#    include "primitive_inst.h"

namespace ov::intel_gpu::ocl {

namespace {

using namespace ov::intel_gpu::ocl;

dnnl::memory::data_type convert_data_type(cldnn::data_types dt) {
    switch (dt) {
    case cldnn::data_types::f32:
        return dnnl::memory::data_type::f32;
    case cldnn::data_types::f16:
        return dnnl::memory::data_type::f16;
    case cldnn::data_types::i8:
        return dnnl::memory::data_type::s8;
    case cldnn::data_types::u8:
        return dnnl::memory::data_type::u8;
    case cldnn::data_types::i32:
        return dnnl::memory::data_type::s32;
    case cldnn::data_types::i4:
        return dnnl::memory::data_type::s4;
    case cldnn::data_types::u4:
        return dnnl::memory::data_type::u4;
    default:
        throw std::invalid_argument("[clDNN] Unsupported conversion from cldnn to onednn type");
    }
}

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

    onednn_matmul(dnnl::memory::data_type act_dtype, dnnl::memory::data_type weight_dtype, int batch_size, int ic, int oc, int ic_group_size = -1) {
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
            w_scale(ic_group_size).w_zp(ic_group_size).fpmath_f16();
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
    onednn_matmul& post_op_silu() {
        float alpha = 1.0f;
        float beta = 0.0f;
        postops.append_eltwise(dnnl::algorithm::eltwise_swish, alpha, beta);
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
        with_silu,
        with_silu_bin_mul,
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
                  type t)
        : onednn_matmul(act_dtype, weight_dtype, batch, ic, oc, ic_group_size) {
        if (t == type::with_bin_mul) {
            bin_post_id = 0;
            post_op_bin_mul(true);
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
        if (t == type::with_silu)
            post_op_silu();
        if (t == type::with_silu_bin_mul) {
            bin_post_id = 1;
            post_op_silu();
            post_op_bin_mul(true);
        }

        create(eng);
    }
};

// all jit-based/performance-aware function should be a functor/callable because:
//   - it needs to hold reference to kernel (to save build time & resources)
//   - it needs to do other compile time preparation work and hold the relevant
//     runtime-data-struct (to make runtime faster)
// to optimize compile-time-workload itself, the functor instance itself should be
// cached with compile-time parameter as the key.
//
// because it's a functor, which supposed to have no states, so cache-factory should
// always return shared_ptr to constant object, so it won't behave differently when being
// called by different caller, and this also ensure it's multi-threading safe since it
// won't modify it's content.
//
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

// create const object with internal cache with constructor-args as the key
// this helps reduces construction time overhead, and perfectly suitable
// for caching functor/callable.
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
                                dnnl::memory zp) {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("onednn_linear::create()"));
        auto mm = make_cacheable<onednn_matmul>(eng, act_dtype, weight_dtype, batch, ic, oc, ic_group_size, t);
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

class MoE3GemmSwigluSoftMaxTopK : public KernelGenerator {
public:
    MoE3GemmSwigluSoftMaxTopK() : KernelGenerator("moe_3gemm_swiglu_fuse", "softmax_topk") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<moe_3gemm_fused_compressed>();
        jit.make("SOFTMAX_TOPK_ENABLE", 1);
        jit.make("TOP_K", desc->_config.top_k);
        jit.make("VALUE_NUM", desc->_config.num_expert);
        jit.make("MOE_DTYPE", params.get_input_layout(0).data_type == ov::element::f16 ? "half" : "float");
        jit.make("MOE_DTYPE_SIZE", params.get_input_layout(0).data_type == ov::element::f16 ? 2 : 4);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }
};

class MoE3GemmSwigluGather : public KernelGenerator {
public:
    MoE3GemmSwigluGather() : KernelGenerator("moe_3gemm_swiglu_fuse", "gather") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<moe_3gemm_fused_compressed>();
        auto& engine = params.prog->get_engine();
        const auto& info = engine.get_device_info();
        jit.make("GATHER_ENABLE", 1);
        jit.make("HIDDEN_SIZE", desc->_config.hidden_size);
        jit.make("MOE_DTYPE", params.get_input_layout(0).data_type == ov::element::f16 ? "half" : "float");
        jit.make("MOE_DTYPE_SIZE", params.get_input_layout(0).data_type == ov::element::f16 ? 2 : 4);
        jit.make("SUBGROUP_SIZE", info.arch >= gpu_arch::xe2 ? 32 : 16);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }
};

class MoE3GemmSwigluPrefillMaskGen : public KernelGenerator {
public:
    MoE3GemmSwigluPrefillMaskGen() : KernelGenerator("moe_mask_gen", "prefill_mask_gen") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<moe_3gemm_fused_compressed>();
        jit.make("INPUT0_TYPE", "int");   // topk_id
        jit.make("OUTPUT_TYPE", "int");   // tokens_per_expert
        jit.make("OUTPUT1_TYPE", "int");  // experts_info_start_idx
        jit.make("OUTPUT2_TYPE", "int");  // experts_id
        jit.make("OUTPUT3_TYPE", "int");  // tokens_lens_per_expert
        jit.make("OUTPUT4_TYPE", "int");  // num_actual_used_experts

        auto& config = desc->_config;
        jit.make("NUM_EXPERTS_PER_TOKEN", config.top_k);
        jit.make("SET_TOKEN_LEN", 1);
        jit.make("OPTIONAL_SHAPE_INFO_ARG", "");

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }
};

static size_t get_seq_len(cldnn::layout& layout) {
    auto shape = layout.get_shape();
    size_t seq_len = static_cast<size_t>(shape[0]);
    if (shape.size() == 4) {
        seq_len = static_cast<size_t>(shape[0] * shape[1]);
    }
    return seq_len;
}

static size_t get_vec_size(const RuntimeParams& params) {
    const auto& input = params.get_input_layout(0);
    size_t vec_size = 1;
    switch (input.data_type) {
    case ov::element::i8:
    case ov::element::u8:
        vec_size = 16;
        break;
    case ov::element::f16:
        vec_size = 8;
        break;
    case ov::element::f32:
    case ov::element::i32:
        vec_size = 4;
        break;
    case ov::element::i64:
        vec_size = 2;
        break;
    default:
        vec_size = 1;
        break;
    }
    return vec_size;
}

static auto calc_thread_count(RuntimeParams& params, const size_t vector_size, const size_t hidden_size) {
    auto max_wgs = params.get_program().get_engine().get_device_info().max_work_group_size;
    const uint64_t threads_needed = (hidden_size + vector_size - 1) / vector_size;
    size_t local_threads_needed = std::min(threads_needed, max_wgs);
    size_t batches_per_thread = 1;
    size_t unaligned_elements = 0;

    if (threads_needed <= max_wgs) {
        batches_per_thread = 1;
        unaligned_elements = hidden_size % vector_size;
    } else {
        batches_per_thread = (threads_needed + max_wgs - 1) / max_wgs;
        auto new_block_size = batches_per_thread * vector_size;
        unaligned_elements = hidden_size % new_block_size;

        local_threads_needed = hidden_size / new_block_size;
        auto partialblock = (hidden_size % new_block_size != 0) ? 1 : 0;
        local_threads_needed += partialblock;
    }

    return std::tuple{local_threads_needed, batches_per_thread, unaligned_elements};
}
class MoE3GemmSwigluPrefillGather : public KernelGenerator {
public:
    MoE3GemmSwigluPrefillGather() : KernelGenerator("moe_gather_ref", "prefill_gather") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<moe_3gemm_fused_compressed>();
        // auto& engine = params.prog->get_engine();
        // const auto& info = engine.get_device_info();

        auto hidden_size = desc->_config.hidden_size;
        auto block_size = get_vec_size(params);
        auto [local_threads_count, batches_per_thread, unaligned_elements] = calc_thread_count(const_cast<RuntimeParams&>(params), block_size, hidden_size);

        jit.make("HIDDEN_SIZE", hidden_size);
        jit.make("VEC_BLK_SIZE", block_size);
        jit.make("BATCHES_PER_THREAD", batches_per_thread);
        jit.make("UNALIGNED_ELEMENTS", unaligned_elements);

        jit.make("INPUT0_TYPE", "half");
        jit.make("INPUT1_TYPE", "int");
        jit.make("OUTPUT_TYPE", "half");
        jit.make("OPTIONAL_SHAPE_INFO_ARG", "");

        GPU_DEBUG_TRACE_DETAIL << "MoE3GemmSwigluPrefillGather::get_jit_constants():  hidden_size: " << hidden_size << ", block_size: " << block_size
                               << ", local_threads_count: " << local_threads_count << ", batches_per_thread: " << batches_per_thread
                               << ", unaligned_elements: " << unaligned_elements << std::endl;

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }
};

class MoE3GemmSwigluPrefillSwiglu : public KernelGenerator {
public:
    MoE3GemmSwigluPrefillSwiglu() : KernelGenerator("moe_3gemm_swiglu_fuse", "prefill_swiglu") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<moe_3gemm_fused_compressed>();
        auto& engine = params.prog->get_engine();
        const auto& info = engine.get_device_info();

        jit.make("PREFILL_SWIGLU_ENABLE", 1);
        jit.make("SUBGROUP_SIZE", info.arch >= gpu_arch::xe2 ? 32 : 16);
        jit.make("INTERMEDIA_SIZE", desc->_config.inter_size);
        jit.make("MOE_DTYPE", "half");
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }
};

class MoE3GemmSwigluPrefillScatterReduce : public KernelGenerator {
public:
    MoE3GemmSwigluPrefillScatterReduce() : KernelGenerator("moe_scatter_reduction_opt", "moe_scatter_reduction_ref") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<moe_3gemm_fused_compressed>();
        // auto& engine = params.prog->get_engine();
        // const auto& info = engine.get_device_info();

        auto hidden_size = desc->_config.hidden_size;
        auto block_size = 4;
        auto [local_threads_count, batches_per_thread, unaligned_elements] = calc_thread_count(const_cast<RuntimeParams&>(params), block_size, hidden_size);

        jit.make("OPTIONAL_SHAPE_INFO_ARG", "");
        jit.make("ACTIVE_EXPERTS", desc->_config.top_k);
        jit.make("HIDDEN_SIZE", hidden_size);
        jit.make("VEC_BLK_SIZE", 4);
        jit.make("BATCHES_PER_THREAD", batches_per_thread);
        jit.make("SET_ACTUAL_USED_EXPERTS_NUM", 1);

        jit.make("INPUT0_TYPE", "half");  // mlp_down output
        jit.make("INPUT1_TYPE", "int");   // expert indices per token
        jit.make("INPUT2_TYPE", "half");  // experts router weights
        jit.make("INPUT3_TYPE", "int");   // tokens per expert
        jit.make("INPUT4_TYPE", "int");   // expert start offsets
        jit.make("INPUT5_TYPE", "int");   // tokens len for experts
        jit.make("INPUT6_TYPE", "int");   // expert id
        jit.make("OUTPUT_TYPE", "half");  // output

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }
};

class MoE3GemmSwigluScatter : public KernelGenerator {
public:
    MoE3GemmSwigluScatter() : KernelGenerator("moe_3gemm_swiglu_fuse", "index_add") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<moe_3gemm_fused_compressed>();
        jit.make("SCATTER_ENABLE", 1);
        jit.make("HIDDEN_SIZE", desc->_config.hidden_size);
        jit.make("MOE_DTYPE", params.get_input_layout(0).data_type == ov::element::f16 ? "half" : "float");
        jit.make("MOE_DTYPE_SIZE", params.get_input_layout(0).data_type == ov::element::f16 ? 2 : 4);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }
};

// Performance tuning parameters
#    define N_BLOCK      4
#    define SUBGROUP_NUM 8

static void add_common_consts(const RuntimeParams& params, JitConstants& jit) {
    auto desc = params.typed_desc<moe_3gemm_fused_compressed>();
    auto& engine = params.prog->get_engine();
    const auto& info = engine.get_device_info();
    auto gate_up_group_size = desc->_config.group_size;
    auto down_group_size = desc->_config.group_size;
    if (desc->_config.group_size == std::numeric_limits<size_t>::max()) {
        gate_up_group_size = desc->_config.hidden_size;
        down_group_size = desc->_config.inter_size;
    }

    GPU_DEBUG_TRACE_DETAIL << "[DEBUG] moe_3gemm_swiglu_opt: group_size=" << desc->_config.group_size << ", gate_up_group_size=" << gate_up_group_size
                           << ", down_group_size=" << down_group_size << std::endl;
    jit.make("MAX_TOPK", desc->_config.top_k);
    jit.make("EXPERT_NUM", desc->_config.num_expert);
    jit.make("HIDDEN_SIZE", desc->_config.hidden_size);
    jit.make("INTERMEDIATE_SIZE", desc->_config.inter_size);
    jit.make("N_BLOCK", N_BLOCK);
    jit.make("SUBGROUP_SIZE", info.arch >= gpu_arch::xe2 ? 32 : 16);
    jit.make("SUBGROUP_NUM", SUBGROUP_NUM);
    jit.make("GATE_UP_GROUP_SIZE", gate_up_group_size);
    jit.make("DOWN_GROUP_SIZE", down_group_size);
    jit.make("MOE_DTYPE", params.get_input_layout(0).data_type == ov::element::f16 ? "half" : "float");
    jit.make("MOE_DTYPE_SIZE", params.get_input_layout(0).data_type == ov::element::f16 ? 2 : 4);

    ov::element::Type weight_dt = params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_0)).data_type;
    // auto scale_dt = params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::SCALE_0)).data_type;
    // auto zp_dt = params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::ZP_0)).data_type;
    if (weight_dt == ov::element::u4 || weight_dt == ov::element::i4) {
        jit.make("WEIGHT_COMPRESSEION_DT", 0);
        jit.make("MOE_WEI_DT", "uchar");
        jit.make("MOE_SCALE_DT", "half");
        jit.make("MOE_ZP_DT", "uchar");
    } else if (weight_dt == ov::element::u8 || weight_dt == ov::element::i8) {
        jit.make("WEIGHT_COMPRESSEION_DT", 1);
        jit.make("MOE_WEI_DT", "uchar");
        jit.make("MOE_SCALE_DT", "half");
        jit.make("MOE_ZP_DT", "uchar");
    } else if (weight_dt == ov::element::f16) {
        jit.make("WEIGHT_COMPRESSEION_DT", 2);
        jit.make("MOE_WEI_DT", "half");
        jit.make("MOE_SCALE_DT", "half");  // not use
        jit.make("MOE_ZP_DT", "half");     // not use
    }
}

class MoE3GemmSwigluMLPGateUp : public KernelGenerator {
public:
    MoE3GemmSwigluMLPGateUp() : KernelGenerator("moe_3gemm_swiglu_mlp", "gate_up") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        add_common_consts(params, jit);
        jit.make("GATE_UP_ENABLE", 1);

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }
};

class MoE3GemmSwigluMLPDown : public KernelGenerator {
public:
    MoE3GemmSwigluMLPDown() : KernelGenerator("moe_3gemm_swiglu_mlp", "down") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        add_common_consts(params, jit);
        jit.make("DOWN_ENABLE", 1);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }
};

class MoE3GemmSwigluMLPReduce : public KernelGenerator {
public:
    MoE3GemmSwigluMLPReduce() : KernelGenerator("moe_3gemm_swiglu_mlp", "reduce") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        add_common_consts(params, jit);
        jit.make("REDUCE_ENABLE", 1);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }
};

dnnl::memory convert2dnnl(const memory::ptr& ptr, const std::vector<int64_t>& dim, dnnl::memory::format_tag tag, int64_t offset = 0) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("convert2dnnl"));
    return ptr->get_onednn_memory(dnnl::memory::desc(dnnl::memory::dims(dim), convert_data_type(ptr->get_layout().data_type), tag), offset);
}

static bool use_micro_gemm_prefill;
static bool use_gpu_mask_gen_prefill;
class moe_3gemm_swiglu_opt_impl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::MoE3GemmSwigluImpl)
    Stage::Ptr softmax_topk = make_stage<MoE3GemmSwigluSoftMaxTopK>();
    Stage::Ptr gather = make_stage<MoE3GemmSwigluGather>();
    Stage::Ptr scatter = make_stage<MoE3GemmSwigluScatter>();
    Stage::Ptr mlp_gate_up = make_stage<MoE3GemmSwigluMLPGateUp>();
    Stage::Ptr mlp_down = make_stage<MoE3GemmSwigluMLPDown>();
    Stage::Ptr mlp_reduce = make_stage<MoE3GemmSwigluMLPReduce>();

    Stage::Ptr prefill_gather = make_stage<MoE3GemmSwigluPrefillGather>();
    Stage::Ptr micro_gemm_gate = make_stage<MoE3GemmMicroGenerator>(MoE3GemmMicroKernelType::MLP_GATE);
    Stage::Ptr micro_gemm_up = make_stage<MoE3GemmMicroGenerator>(MoE3GemmMicroKernelType::MLP_UP);
    Stage::Ptr micro_gemm_down = make_stage<MoE3GemmMicroGenerator>(MoE3GemmMicroKernelType::MLP_DOWN);
    Stage::Ptr prefill_swiglu = make_stage<MoE3GemmSwigluPrefillSwiglu>();
    Stage::Ptr prefill_scatter_reduce = make_stage<MoE3GemmSwigluPrefillScatterReduce>();
    Stage::Ptr prefill_mask_gen = make_stage<MoE3GemmSwigluPrefillMaskGen>();

    struct dnnl_weights {
        dnnl::memory weight;
        dnnl::memory scale;
        dnnl::memory zp;
        int ic, oc, ic_group_size;
    };

    // expert_mask result in cpu side
    struct expert_mask_cpu {
        std::vector<int8_t> pred_flag;
        // shape: [expert_num, batch_no]
        std::vector<std::vector<int>> batch;
        // shape: [expert_num, topk_no]
        std::vector<std::vector<int>> topk;
    };

    // store expert_mask for gpu kernel
    struct expert_mask_gpu {
        memory::ptr batch;
        memory::ptr topk;
    };

    struct moe_fusion_weights_base_addr {
        memory::ptr weight[3];  // gate/up/down weights, experts fusion
        memory::ptr scale[3];
        memory::ptr zp[3];
        memory::ptr bias[3];
    } moe_fusion_weights;

    struct scratch_buffers {
        // softmax+topk
        memory::ptr topk_id;
        memory::ptr topk_weights;

        // fast single batch: scratch.up = up(x) * silu(gate(x))
        //                    scratch.y = down(scratch.up) * routing_weights
        memory::ptr up;
        memory::ptr y;
        // onednn: scratch.x, scratch.routing_weights = gather(x, ...)
        //         scratch.up = up(scratch.x)
        //         scratch.gate = gate(scratch.x) * scratch.up
        //         scratch.y = down(scratch.gate) * routing_weights
        memory::ptr x;
        memory::ptr routing_weights;
        memory::ptr gate;
        // buffers for batch and topk from cpu, each expert has one
        std::vector<expert_mask_gpu> expert_masks;

        moe_fusion_weights_base_addr moe_fusion_wei_addr;
        memory::ptr input_routing_weights;
        memory::ptr input_router_topk_idx;
    };

    std::vector<std::vector<dnnl_weights>> _dnnl_weights;
    int _hidden_size;
    int _intermediate_size;
    int _gate_up_group_size;
    int _down_group_size;

    moe_3gemm_swiglu_opt_impl() : PrimitiveImplOCL(moe_3gemm_swiglu_opt::get_type_info_static()) {}
    moe_3gemm_swiglu_opt_impl(const program_node& node, const RuntimeParams& params) : moe_3gemm_swiglu_opt_impl() {
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<MoE3GemmRuntimeParams>();
        }
        init(node.as<moe_3gemm_fused_compressed>().get_primitive());

        auto use_micro_gemm_prefill_str = std::getenv("MOE_USE_MICRO_GEMM_PREFILL");
        if (use_micro_gemm_prefill_str) {
            GPU_DEBUG_TRACE_DETAIL << "MOE_USE_MICRO_GEMM_PREFILL = " << use_micro_gemm_prefill_str << std::endl;
            use_micro_gemm_prefill = std::stoi(use_micro_gemm_prefill_str);
        } else {
            // micro_gemm is better than gemm, default to use it
            use_micro_gemm_prefill = true;
        }

        auto use_gpu_mask_gen_prefill_str = std::getenv("MOE_USE_GPU_MASK_PREFILL");
        if (use_gpu_mask_gen_prefill_str) {
            GPU_DEBUG_TRACE_DETAIL << "MOE_USE_GPU_MASK_PREFILL = " << use_gpu_mask_gen_prefill_str << std::endl;
            use_gpu_mask_gen_prefill = std::stoi(use_gpu_mask_gen_prefill_str);
        } else {
            // gpu mask gen kernel performace is worse than cpu mask gen, default is off
            use_gpu_mask_gen_prefill = false;
        }

        auto& engine = params.prog->get_engine();
        const auto& info = engine.get_device_info();
        if (info.arch < gpu_arch::xe2) {
            use_micro_gemm_prefill = false;
            GPU_DEBUG_TRACE_DETAIL << "[DEBUG] moe_3gemm_swiglu_opt_impl(): use_micro_gemm_prefill=" << use_micro_gemm_prefill
                                   << ", arch=" << static_cast<int>(info.arch) << std::endl;
        } else {
            GPU_DEBUG_TRACE_DETAIL << "[DEBUG] moe_3gemm_swiglu_opt_impl(): use_micro_gemm_prefill=" << use_micro_gemm_prefill
                                   << ", arch=" << static_cast<int>(info.arch) << std::endl;
        }

        // Remove this limitation once micro_gemm kernels has supported i8/u8 weights.
        const auto& weight_dt = params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_0)).data_type;
        if (weight_dt != data_types::u4 && use_micro_gemm_prefill) {
            use_micro_gemm_prefill = false;
        }

        // Don't change the order of stages
        add_stage(softmax_topk, params);
        add_stage(gather, params);
        add_stage(scatter, params);
        add_stage(mlp_gate_up, params);
        add_stage(mlp_down, params);
        add_stage(mlp_reduce, params);
        if (use_micro_gemm_prefill) {
            add_stage(prefill_mask_gen, params);
            add_stage(prefill_gather, params);
            add_stage(micro_gemm_gate, params);
            add_stage(micro_gemm_up, params);
            add_stage(prefill_swiglu, params);
            add_stage(micro_gemm_down, params);
            add_stage(prefill_scatter_reduce, params);
        }
    }

    void init(const std::shared_ptr<const moe_3gemm_fused_compressed>& cur_moe) {
        _hidden_size = static_cast<int>(cur_moe->_config.hidden_size);
        _intermediate_size = static_cast<int>(cur_moe->_config.inter_size);
        _gate_up_group_size = static_cast<int>(cur_moe->_config.group_size);
        _down_group_size = static_cast<int>(cur_moe->_config.group_size);

        if (cur_moe->_config.group_size == std::numeric_limits<size_t>::max()) {
            _gate_up_group_size = static_cast<int>(cur_moe->_config.hidden_size);
            _down_group_size = static_cast<int>(cur_moe->_config.inter_size);
        }
        GPU_DEBUG_TRACE_DETAIL << "[DEBUG] moe_3gemm_swiglu_opt prefill: group_size=" << cur_moe->_config.group_size
                               << ", gate_up_group_size=" << _gate_up_group_size << ", down_group_size=" << _down_group_size << std::endl;
    }

    void init_dnnl_weights(const std::shared_ptr<const moe_3gemm_fused_compressed>& cur_moe,
                           cldnn::engine& engine,
                           const struct moe_fusion_weights_base_addr& moe_fusion_wei_addr) {
        if (_dnnl_weights.size() == cur_moe->_config.num_expert)
            return;
        init(cur_moe);

        auto get_bytes_count = [](int64_t size, const cldnn::layout& layout) {
            ov::element::Type dt = layout.data_type;
            switch (layout.data_type) {
            case ov::element::u4:
            case ov::element::i4:
                return size / 2;
                break;
            default:
                return size * static_cast<int64_t>(dt.size());
                break;
            }
        };

        _dnnl_weights.resize(cur_moe->_config.num_expert);
        for (size_t j = 0; j < cur_moe->_config.num_expert; j++) {
            auto& dnnl_weights = _dnnl_weights[j];
            dnnl_weights.resize(3);
            dnnl_weights[0].ic = _hidden_size;
            dnnl_weights[0].ic_group_size = _gate_up_group_size;
            dnnl_weights[0].oc = _intermediate_size;
            dnnl_weights[1].ic = _hidden_size;
            dnnl_weights[1].ic_group_size = _gate_up_group_size;
            dnnl_weights[1].oc = _intermediate_size;
            dnnl_weights[2].ic = _intermediate_size;
            dnnl_weights[2].ic_group_size = _down_group_size;
            dnnl_weights[2].oc = _hidden_size;
            for (int i = 0; i < 3; i++) {
                // weight shape: [ic, oc], type: u4/i8
                int64_t wei_offset = j * get_bytes_count(dnnl_weights[i].ic * dnnl_weights[i].oc, moe_fusion_wei_addr.weight[i]->get_layout());
                dnnl_weights[i].weight =
                    convert2dnnl(moe_fusion_wei_addr.weight[i], {dnnl_weights[i].ic, dnnl_weights[i].oc}, dnnl::memory::format_tag::ba, wei_offset);

                // scale shape: [ic / ic_group_size, oc], type: f16
                int64_t scale_offset =
                    j * get_bytes_count(dnnl_weights[i].ic * dnnl_weights[i].oc / dnnl_weights[i].ic_group_size, moe_fusion_wei_addr.scale[i]->get_layout());
                dnnl_weights[i].scale = convert2dnnl(moe_fusion_wei_addr.scale[i],
                                                     {dnnl_weights[i].ic / dnnl_weights[i].ic_group_size, dnnl_weights[i].oc},
                                                     dnnl::memory::format_tag::ab,
                                                     scale_offset);

                // zp shape: [ic / ic_group_size, oc], type: u4/i8
                int64_t zp_offset =
                    j * get_bytes_count(dnnl_weights[i].ic * dnnl_weights[i].oc / dnnl_weights[i].ic_group_size, moe_fusion_wei_addr.zp[i]->get_layout());
                dnnl_weights[i].zp = convert2dnnl(moe_fusion_wei_addr.zp[i],
                                                  {dnnl_weights[i].ic / dnnl_weights[i].ic_group_size, dnnl_weights[i].oc},
                                                  dnnl::memory::format_tag::ab,
                                                  zp_offset);
            }
        }
    }

    void load(BinaryInputBuffer& ib) override {
        PrimitiveImplOCL::load(ib);
        const kernel_impl_params* impl_params = reinterpret_cast<kernel_impl_params*>(ib.getKernelImplParams());
        init(impl_params->typed_desc<moe_3gemm_fused_compressed>());
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        auto cur_moe = make_deep_copy<moe_3gemm_swiglu_opt_impl>(this);
        cur_moe->_dnnl_weights = _dnnl_weights;
        cur_moe->_hidden_size = _hidden_size;
        cur_moe->_intermediate_size = _intermediate_size;
        cur_moe->_gate_up_group_size = _gate_up_group_size;
        cur_moe->_down_group_size = _down_group_size;
        return cur_moe;
    }

    // Notice: don't change the order of internal buffers, it is defined in MOE3GemmInternalBufferIdx
    std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        auto cur_moe = params.typed_desc<moe_3gemm_fused_compressed>();
        const auto& config = cur_moe->_config;
        size_t max_topk = static_cast<size_t>(config.top_k);
        size_t expert_num = static_cast<size_t>(config.num_expert);
        auto hidden_states_layout = params.input_layouts[0];
        auto token_num = get_seq_len(hidden_states_layout);
        auto data_type = hidden_states_layout.data_type;

        std::vector<BufferDescriptor> internal_buffers;
        // softmax+topk
        layout layout_topk_id(ov::Shape{token_num, max_topk}, data_types::u32, cldnn::format::bfyx);
        layout layout_topk_weights(ov::Shape{token_num, max_topk}, data_type, cldnn::format::bfyx);
        internal_buffers.emplace_back(layout_topk_id, true);       // 0: topk_id
        internal_buffers.emplace_back(layout_topk_weights, true);  // 1: topk_weights

        // To support micro_gemm, prefill need to allocate max_topk * token_num for input data of micro_gemm
        auto max_batch = max_topk * token_num;
        layout layout_gateup_out(ov::Shape{max_batch, static_cast<size_t>(config.inter_size)}, data_type, cldnn::format::bfyx);
        layout layout_down_out(ov::Shape{max_batch, static_cast<size_t>(config.hidden_size)}, data_type, cldnn::format::bfyx);
        internal_buffers.emplace_back(layout_gateup_out, true);  // 2: up output
        internal_buffers.emplace_back(layout_down_out, true);    // 3: down output
        // onednn: scratch.x, scratch.routing_weights = gather(x, ...)
        //         scratch.up = up(scratch.x)
        //         scratch.gate = gate(scratch.x) * scratch.up
        //         scratch.y = down(scratch.gate) * routing_weights
        internal_buffers.emplace_back(layout_down_out, true);  // 4: up/gate input, scratch.x has same layout with down output
        layout routing_layout(ov::Shape{token_num * max_topk}, data_type, cldnn::format::bfyx);
        internal_buffers.emplace_back(routing_layout, true);     // 5: routing_weights
        internal_buffers.emplace_back(layout_gateup_out, true);  // 6: gate output, scratch.gate has same layout with up
        // expert masks for gpu
        layout index_layout(ov::Shape{expert_num, token_num}, ov::element::i32, cldnn::format::bfyx);
        internal_buffers.emplace_back(index_layout, true);  // 7: expert_mask_batch
        internal_buffers.emplace_back(index_layout, true);  // 8: expert_mask_topk

        GPU_DEBUG_TRACE_DETAIL << "[DEBUG] get_internal_buffer_descs(): use_micro_gemm_prefill=" << use_micro_gemm_prefill << std::endl;
        // for micro_gemm
        if (use_micro_gemm_prefill && token_num > 1) {
            layout layout_micro_gemm(ov::Shape{expert_num, token_num}, ov::element::i32, cldnn::format::bfyx);
            internal_buffers.emplace_back(layout_micro_gemm, true);  // 9: experts_ids for each activated expert
            internal_buffers.emplace_back(layout_micro_gemm, true);  // 10: token start offset idx (input gather tokens) for each activated expert
            internal_buffers.emplace_back(layout_micro_gemm, true);  // 11: token len (input gather tokens) for each activated expert
            layout layout_token_idx(ov::Shape{token_num * max_topk}, ov::element::i32, cldnn::format::bfyx);
            internal_buffers.emplace_back(layout_token_idx, true);  // 12: token idx per expert
            layout layout_actual_used_expert_num(ov::Shape{1}, ov::element::i32, cldnn::format::bfyx);
            internal_buffers.emplace_back(layout_actual_used_expert_num, false);  // 13: actual_used_expert_num
        }
        return internal_buffers;
    }

    void prepare_internal_buffers(typed_primitive_inst<moe_3gemm_fused_compressed>& instance, scratch_buffers& scratch, size_t token_num) {
        const auto& intermediates_memories = instance.get_intermediates_memories();
        auto& engine = instance.get_network().get_engine();
        scratch.topk_id = intermediates_memories[MOE_INTERNAL_BUFFER_TOPK_IDX];
        scratch.topk_weights = intermediates_memories[MOE_INTERNAL_BUFFER_TOPK_WEIGHTS];
        scratch.up = intermediates_memories[MOE_INTERNAL_BUFFER_UP_OUTPUT];
        scratch.y = intermediates_memories[MOE_INTERNAL_BUFFER_DOWN_OUTPUT];
        if (token_num > 1) {
            scratch.x = intermediates_memories[MOE_INTERNAL_BUFFER_GATE_UP_INPUT];
            scratch.routing_weights = intermediates_memories[MOE_INTERNAL_BUFFER_ROUTING_WEIGHTS];
            scratch.gate = intermediates_memories[MOE_INTERNAL_BUFFER_GATE_OUTPUT];
            const auto& config = instance.get_typed_desc<moe_3gemm_fused_compressed>()->_config;
            int expert_num = static_cast<int>(config.num_expert);
            scratch.expert_masks.resize(expert_num);
            for (int i = 0; i < expert_num; i++) {
                auto mask_layout = cldnn::layout({static_cast<int>(token_num)}, cldnn::data_types::i32, cldnn::format::get_default_format(1));
                scratch.expert_masks[i].batch =
                    engine.create_subbuffer(*intermediates_memories[MOE_INTERNAL_BUFFER_EXPERT_MASK_BATCH], mask_layout, i * token_num * sizeof(int32_t));
                scratch.expert_masks[i].topk =
                    engine.create_subbuffer(*intermediates_memories[MOE_INTERNAL_BUFFER_EXPERT_MASK_TOPK], mask_layout, i * token_num * sizeof(int32_t));
            }
        }

        // gate
        scratch.moe_fusion_wei_addr.weight[0] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_0));
        scratch.moe_fusion_wei_addr.scale[0] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::SCALE_0));
        scratch.moe_fusion_wei_addr.zp[0] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::ZP_0));

        // up
        scratch.moe_fusion_wei_addr.weight[1] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_1));
        scratch.moe_fusion_wei_addr.scale[1] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::SCALE_1));
        scratch.moe_fusion_wei_addr.zp[1] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::ZP_1));

        // down
        scratch.moe_fusion_wei_addr.weight[2] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_2));
        scratch.moe_fusion_wei_addr.scale[2] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::SCALE_2));
        scratch.moe_fusion_wei_addr.zp[2] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::ZP_2));
    }

    void get_expert_mask_from_gpu(const MOE3GemmFusedCompressed::Config& config, memory::ptr mem, stream& stream, expert_mask_cpu& expert_mask) {
        // shape: [token_num, topk]
        auto layout = mem->get_layout();
        const auto& shape = layout.get_shape();

        int max_expert_num = static_cast<int>(config.num_expert);
        int max_topk = static_cast<int>(config.top_k);
        int max_tokens = static_cast<int>(shape[0]);

        expert_mask.pred_flag.resize(max_expert_num, 0);
        expert_mask.batch.resize(max_expert_num, {});
        expert_mask.topk.resize(max_expert_num, {});

        if (layout.data_padding) {
            OPENVINO_THROW("get_expert_mask_from_memory not support padding");
        }

        GPU_DEBUG_TRACE_DETAIL << "[DEBUG] get_expert_mask_from_gpu: layout=" << layout.to_short_string() << ", max_expert_num=" << max_expert_num
                               << ", max_topk=" << max_topk << ", max_tokens=" << max_tokens << std::endl;
        std::vector<int32_t> buf(max_topk * max_tokens);
        mem->copy_to(stream, buf.data(), 0, 0, buf.size() * sizeof(int32_t), true);

        for (int b = 0; b < max_tokens; b++) {
            auto* tok_p = &buf[b * max_topk];
            for (int t = 0; t < max_topk; t++) {
                auto expert_no = tok_p[t];
                if (expert_no >= max_expert_num) {
                    OPENVINO_THROW("expert_no ", expert_no, " exceed max_expert_num ", max_expert_num);
                }

                expert_mask.batch[expert_no].push_back(b);
                expert_mask.topk[expert_no].push_back(t + b * max_topk);
                expert_mask.pred_flag[expert_no] = 1;
            }
        }
        {
            // check if the result is ok
            int count = 0;
            for (int no = 0; no < max_expert_num; no++) {
                count += static_cast<int>(expert_mask.batch[no].size());
            }
            OPENVINO_ASSERT(count == max_topk * max_tokens,
                            "With max_expert_num=",
                            max_expert_num,
                            ",max_topk=",
                            max_topk,
                            ",max_tokens=",
                            max_tokens,
                            " should have ",
                            max_topk * max_tokens,
                            " tokens, but current is ",
                            count,
                            ". layout=",
                            layout);
        }
    }

    void copy_expert_mask_to_gpu(stream& stream, const expert_mask_cpu& expert_mask, size_t expert_no, expert_mask_gpu& expert_mask_mem) {
        auto size = expert_mask.batch[expert_no].size() * sizeof(int);
        expert_mask_mem.batch->copy_from(stream, expert_mask.batch[expert_no].data(), 0, 0, size, true);
        expert_mask_mem.topk->copy_from(stream, expert_mask.topk[expert_no].data(), 0, 0, size, true);
    }

    cldnn::event::ptr execute_stage(const std::vector<cldnn::event::ptr>& events,
                                    cldnn::primitive_inst& instance,
                                    Stage& stage,
                                    std::vector<memory::ptr> inputs,
                                    std::vector<memory::ptr> outputs,
                                    const std::vector<size_t>& global,
                                    const std::vector<size_t>& local,
                                    bool needs_completion_event = false,
                                    std::vector<int> scalar_inputs = {}) const {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("moe_3gemm_swiglu_opt_impl::execute_stage"));
        cldnn::stream& stream = instance.get_network().get_stream();
        cldnn::kernel_arguments_data args;
        cldnn::kernel_arguments_desc desc;

        GPU_DEBUG_TRACE_DETAIL << "moe::execute_stage: " << stage.kernel->get_id() << std::endl;
        for (uint32_t i = 0; i < inputs.size(); i++) {
            desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, i});
            args.inputs.push_back(inputs[i]);
            GPU_DEBUG_TRACE_DETAIL << "\tinput[" << i << "]: " << inputs[i]->get_layout().to_short_string() << std::endl;
        }

        for (uint32_t i = 0; i < outputs.size(); i++) {
            desc.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, i});
            args.outputs.push_back(outputs[i]);
            GPU_DEBUG_TRACE_DETAIL << "\toutput[" << i << "]: " << outputs[i]->get_layout().to_short_string() << std::endl;
        }

        cldnn::scalars_desc scalar_desc;
        if (!scalar_inputs.empty()) {
            scalar_desc.resize(scalar_inputs.size());
            for (uint32_t i = 0; i < scalar_inputs.size(); i++) {
                desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, i});
                scalar_desc[i].t = ScalarDescriptor::Types::INT32;
                scalar_desc[i].v.s32 = scalar_inputs[i];
            }
            args.scalars = &scalar_desc;
            GPU_DEBUG_TRACE_DETAIL << "\tscalar_inputs: ";
            for (const auto& scalar : scalar_inputs) {
                GPU_DEBUG_TRACE_DETAIL << scalar << " ";
            }
            GPU_DEBUG_TRACE_DETAIL << std::endl;
        }

        stream.set_arguments(*stage.kernel, desc, args);
        desc.workGroups.global = global;
        desc.workGroups.local = local;

        if (global.size() == 2) {
            GPU_DEBUG_TRACE_DETAIL << "\tgws = {" << global[0] << ", " << global[1] << "}" << std::endl;
            GPU_DEBUG_TRACE_DETAIL << "\tlws = {" << local[0] << ", " << local[1] << "}" << std::endl;
        } else if (global.size() == 3) {
            GPU_DEBUG_TRACE_DETAIL << "\tgws = {" << global[0] << ", " << global[1] << ", " << global[2] << "}" << std::endl;
            GPU_DEBUG_TRACE_DETAIL << "\tlws = {" << local[0] << ", " << local[1] << ", " << local[2] << "}" << std::endl;
        }

        return stream.enqueue_kernel(*stage.kernel, desc, {}, events, needs_completion_event);
    }

    auto get_input_info(typed_primitive_inst<moe_3gemm_fused_compressed>& instance, int idx) {
        auto mem = instance.input_memory_ptr(idx);
        auto dep = instance.dependencies()[idx];
        auto layout = dep.first->get_impl_params()->get_output_layout(dep.second);
        return std::make_tuple(mem, layout);
    }

    bool print_wei = false;
    cldnn::event::ptr exec_single_token(const std::vector<cldnn::event::ptr>& events,
                                        typed_primitive_inst<moe_3gemm_fused_compressed>& instance,
                                        scratch_buffers& scratch) {
        auto cur_moe = instance.get_typed_desc<moe_3gemm_fused_compressed>();
        int max_topk = static_cast<int>(cur_moe->_config.top_k);

        auto final_hidden_states_mem_ptr = instance.output_memory_ptr(0);
        auto batch_mem_ptr = scratch.topk_id;
        auto [hidden_states_mem_ptr, hidden_states_layout] = get_input_info(instance, static_cast<size_t>(MOE3GemmInputIndex::HIDDEN_STATES));
        auto routing_mem_ptr = scratch.topk_weights;

        _hidden_size = static_cast<int>(cur_moe->_config.hidden_size);
        _intermediate_size = static_cast<int>(cur_moe->_config.inter_size);

        const size_t subgroup_size = instance.get_impl_params()->get_device_info().arch >= gpu_arch::xe2 ? 32 : 16;
        const size_t max_work_group_size = instance.get_impl_params()->get_device_info().max_work_group_size;

        // gate
        const auto& mlp_gate_wei_mem = scratch.moe_fusion_wei_addr.weight[0];
        const auto& mlp_gate_scale_mem = scratch.moe_fusion_wei_addr.scale[0];
        const auto& mlp_gate_zp_mem = scratch.moe_fusion_wei_addr.zp[0];

        // up
        const auto& mlp_up_wei_mem = scratch.moe_fusion_wei_addr.weight[1];
        const auto& mlp_up_scale_mem = scratch.moe_fusion_wei_addr.scale[1];
        const auto& mlp_up_zp_mem = scratch.moe_fusion_wei_addr.zp[1];

        // down
        const auto& mlp_down_wei_mem = scratch.moe_fusion_wei_addr.weight[2];
        const auto& mlp_down_scale_mem = scratch.moe_fusion_wei_addr.scale[2];
        const auto& mlp_down_zp_mem = scratch.moe_fusion_wei_addr.zp[2];
        event::ptr ret;

        {
            // scratch.up = up(x) * silu(gate(x))
            auto ret_event = execute_stage(
                events,
                instance,
                *mlp_gate_up,
                {batch_mem_ptr, mlp_gate_wei_mem, mlp_gate_scale_mem, mlp_gate_zp_mem, mlp_up_wei_mem, mlp_up_scale_mem, mlp_up_zp_mem, hidden_states_mem_ptr},
                {scratch.up},
                {static_cast<size_t>(max_topk), subgroup_size, static_cast<size_t>(_intermediate_size / N_BLOCK)},
                {1, subgroup_size, SUBGROUP_NUM});

            // scratch.y = down(scratch.up) * weight[expert_no]
            ret_event = execute_stage({ret_event},
                                      instance,
                                      *mlp_down,
                                      {batch_mem_ptr, mlp_down_wei_mem, mlp_down_scale_mem, mlp_down_zp_mem, scratch.up, routing_mem_ptr},
                                      {scratch.y},
                                      {static_cast<size_t>(max_topk), subgroup_size, static_cast<size_t>(_hidden_size / N_BLOCK)},
                                      {1, subgroup_size, SUBGROUP_NUM});

            // final = sum(scratch.y)
            ret = execute_stage({ret_event},
                                instance,
                                *mlp_reduce,
                                {scratch.y},
                                {final_hidden_states_mem_ptr},
                                {static_cast<size_t>(1), static_cast<size_t>(_hidden_size)},
                                {1, std::min(max_work_group_size, size_t{1024})},
                                instance.needs_completion_event());
        }
        return ret;
    }

    cldnn::event::ptr exec_prefill_micro_gemm(const std::vector<cldnn::event::ptr>& events,
                                              typed_primitive_inst<moe_3gemm_fused_compressed>& instance,
                                              scratch_buffers& scratch,
                                              const bool use_gpu_mask_gen) {
        auto cur_moe = instance.get_typed_desc<moe_3gemm_fused_compressed>();
        int max_topk = static_cast<int>(cur_moe->_config.top_k);
        const auto& config = cur_moe->_config;

        auto final_hidden_states_mem_ptr = instance.output_memory_ptr(0);
        // [batch, max_topk]
        auto batch_mem_ptr = scratch.topk_id;
        auto [hidden_states_mem_ptr, hidden_states_layout] = get_input_info(instance, static_cast<size_t>(MOE3GemmInputIndex::HIDDEN_STATES));
        auto routing_mem_ptr = scratch.topk_weights;
        auto input_layout = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::HIDDEN_STATES))->get_layout();
        auto token_num = get_seq_len(input_layout);

        _hidden_size = static_cast<int>(cur_moe->_config.hidden_size);
        _intermediate_size = static_cast<int>(cur_moe->_config.inter_size);

        auto rtp = static_cast<MoE3GemmRuntimeParams*>(m_rt_params.get());
        const size_t subgroup_size = instance.get_impl_params()->get_device_info().arch >= gpu_arch::xe2 ? 32 : 16;

        event::ptr ret_event;
        const auto& intermediates_memories = instance.get_intermediates_memories();
        auto& stream = instance.get_network().get_stream();
        auto num_total_experts = static_cast<int>(cur_moe->_config.num_expert);
        int num_actually_used_experts = 0;

        // step 1: generate 4 mask data for following kernel execution
        // input: topk output, [token_len, expert_topk]
        // output:
        //   mask 0: token idx per expert, flat array of length token_len * expert_topk
        //             (experts are laid out consecutively; use experts_info_start_idx + tokens_lens_per_expert to slice)
        //   mask 1: token start offset idx in mask 0 for each activated expert, shape = [activated_expert_num]
        //   mask 2: token len for each activated expert, shape = [activated_expert_num]
        //   mask 3: expert id, shape = [activated_expert_num]
        //   mask 4: actual activated expert num, shape = [1]
        if (use_gpu_mask_gen) {
            auto token_size = token_num;
            ret_event = execute_stage(events,
                                      instance,
                                      *prefill_mask_gen,
                                      {batch_mem_ptr},
                                      {intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_IDX_PER_EXPERT],
                                       intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_START_OFFSET_PER_EXPERT],
                                       intermediates_memories[MOE_INTERNAL_BUFFER_ACTIVATED_EXPERT_IDS],
                                       intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_LEN_PER_ACTIVATED_EXPERT],
                                       intermediates_memories[MOE_INTERNAL_BUFFER_ACTUAL_USED_EXPERT_NUM]},
                                      {static_cast<size_t>(num_total_experts), 1, 1},
                                      {static_cast<size_t>(num_total_experts), 1, 1},
                                      false,
                                      {static_cast<int>(token_size)});

            // num_actually_used_experts is needed for micro_gem wgs, need sync
            ret_event->wait();
            cldnn::mem_lock<int32_t, mem_lock_type::read> num_actual_experts_lock(intermediates_memories[MOE_INTERNAL_BUFFER_ACTUAL_USED_EXPERT_NUM], stream);
            rtp->num_actually_used_experts = num_actual_experts_lock[0];
#    if DEBUG_MOE_LOG
            GPU_DEBUG_TRACE_DETAIL << "Step 1: mask gen by gpu, num_actually_used_experts = " << rtp->num_actually_used_experts << std::endl;
#    endif
        } else {
            ret_event = events.empty() ? nullptr : events[0];
            expert_mask_cpu expert_mask_cpu;
            get_expert_mask_from_gpu(config, batch_mem_ptr, stream, expert_mask_cpu);

            auto token_size = token_num;
            auto max_topk = static_cast<int>(cur_moe->_config.top_k);
            std::vector<int32_t> tokens_per_expert_cpu(token_size * max_topk, -1);
            std::vector<int32_t> tokens_lens_per_expert_cpu(num_total_experts, -1);
            std::vector<int32_t> experts_info_start_idx_cpu(num_total_experts, -1);
            std::vector<int32_t> experts_id_cpu(num_total_experts, -1);

            int tokens_per_expert_iter = 0;
            int experts_id_iter = 0;
            for (int expert_idx = 0; expert_idx < num_total_experts; expert_idx++) {
                if (!expert_mask_cpu.batch[expert_idx].empty()) {
                    experts_info_start_idx_cpu[experts_id_iter] = tokens_per_expert_iter;
                    experts_id_cpu[experts_id_iter] = expert_idx;
                    tokens_lens_per_expert_cpu[experts_id_iter++] = static_cast<int32_t>(expert_mask_cpu.batch[expert_idx].size());
                    num_actually_used_experts++;
                    for (auto t : expert_mask_cpu.batch[expert_idx]) {
                        tokens_per_expert_cpu[tokens_per_expert_iter++] = t;
                    }
                }
            }
            rtp->num_actually_used_experts = num_actually_used_experts;

            intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_IDX_PER_EXPERT]
                ->copy_from(stream, tokens_per_expert_cpu.data(), 0, 0, tokens_per_expert_cpu.size() * sizeof(int32_t), true);
            intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_START_OFFSET_PER_EXPERT]
                ->copy_from(stream, experts_info_start_idx_cpu.data(), 0, 0, num_actually_used_experts * sizeof(int32_t), true);
            intermediates_memories[MOE_INTERNAL_BUFFER_ACTIVATED_EXPERT_IDS]
                ->copy_from(stream, experts_id_cpu.data(), 0, 0, num_actually_used_experts * sizeof(int32_t), true);
            intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_LEN_PER_ACTIVATED_EXPERT]
                ->copy_from(stream, tokens_lens_per_expert_cpu.data(), 0, 0, num_actually_used_experts * sizeof(int32_t), true);

            intermediates_memories[MOE_INTERNAL_BUFFER_ACTUAL_USED_EXPERT_NUM]->copy_from(stream, &num_actually_used_experts, 0, 0, sizeof(int32_t), true);

#    if DEBUG_MOE_LOG
            {
                GPU_DEBUG_TRACE_DETAIL << "\nstep 1: prefill_mask num_actually_used_experts=" << num_actually_used_experts << std::endl;
                GPU_DEBUG_TRACE_DETAIL << "expert_id[" << num_actually_used_experts << "]: = ";
                for (int i = 0; i < num_actually_used_experts; i++) {
                    GPU_DEBUG_TRACE_DETAIL << experts_id_cpu[i] << ", ";
                }
                GPU_DEBUG_TRACE_DETAIL << std::endl;
                GPU_DEBUG_TRACE_DETAIL << "experts_info_start_idx[" << num_actually_used_experts << "]: = ";
                for (int i = 0; i < num_actually_used_experts; i++) {
                    GPU_DEBUG_TRACE_DETAIL << experts_info_start_idx_cpu[i] << ", ";
                }
                GPU_DEBUG_TRACE_DETAIL << std::endl;
                GPU_DEBUG_TRACE_DETAIL << "tokens_len_per_expert[" << num_actually_used_experts << "]: = ";
                for (int i = 0; i < num_actually_used_experts; i++) {
                    GPU_DEBUG_TRACE_DETAIL << tokens_lens_per_expert_cpu[i] << ", ";
                }
                GPU_DEBUG_TRACE_DETAIL << std::endl;
                GPU_DEBUG_TRACE_DETAIL << "tokens_per_expert[" << num_actually_used_experts << "]:" << std::endl;
                int token_idx = 0;
                for (int i = 0; i < num_actually_used_experts; i++) {
                    GPU_DEBUG_TRACE_DETAIL << "\texpert[" << i << "]: = ";
                    for (int j = 0; j < tokens_lens_per_expert_cpu[i]; j++) {
                        GPU_DEBUG_TRACE_DETAIL << tokens_per_expert_cpu[token_idx + j] << ", ";
                    }
                    token_idx += tokens_lens_per_expert_cpu[i];
                    GPU_DEBUG_TRACE_DETAIL << std::endl;
                }
                GPU_DEBUG_TRACE_DETAIL << std::endl;
            }
#    endif
        }

        // step 2: generate gather input tokens
        //  input
        //      0: input tensor, shape = [token_len, hidden_size]
        //      1: token idx per expert, static shape = [token_num * topK_num]
        //  output
        //      0: gathered token: shape = [token_len * expert_topK, hidden_size]
        {
            auto hidden_size = _hidden_size;
            auto block_size = get_vec_size(*instance.get_impl_params());
            auto [local_threads_count, batches_per_thread, unaligned_elements] =
                calc_thread_count(const_cast<RuntimeParams&>(*instance.get_impl_params()), block_size, hidden_size);
            auto token_per_expert = intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_IDX_PER_EXPERT]->get_layout().get_shape()[0];

#    if DEBUG_MOE_LOG
            GPU_DEBUG_TRACE_DETAIL << "\nstep 2: prefill_gather local_threads_count=" << local_threads_count << ", batches_per_thread=" << batches_per_thread
                                   << ", unaligned_elements=" << unaligned_elements << ", token_per_expert=" << token_per_expert
                                   << ", block_size = " << block_size << std::endl;
#    endif
            ret_event = execute_stage({ret_event},
                                      instance,
                                      *prefill_gather,
                                      {instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::HIDDEN_STATES)),
                                       intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_IDX_PER_EXPERT]},
                                      {scratch.x},
                                      {static_cast<size_t>(token_per_expert * local_threads_count), 1, 1},
                                      {static_cast<size_t>(local_threads_count), 1, 1});
        }

        // step 3: moe_gemm for up and gate
        //  input
        //      0: gathered token, shape = [token_len * expert_topK, hidden_size]
        //      1: moe weights
        //      2: expert id, dynamic shape = [activated_expert_num]
        //      3: token start offset idx (input gather tokens) for each activated expert, dynamic shape = [activated_expert_num]
        //      4: token len (input gather tokens) for each activated expert, dynamic shape = [activated_expert_num]
        //      5: m = itermedia_size
        //      6: k = hidden_size
        //      7: wei_scale
        //      8: wei_zp
        //  output:
        //      0: up/gate output, shape = [token_len * expert_topK, hidden_size]
        {
#    if DEBUG_MOE_LOG
            GPU_DEBUG_TRACE_DETAIL << "\nstep 3: moe_gemm for up and gate" << std::endl;
#    endif
            ret_event = PrimitiveImplOCL::execute_stage({ret_event}, instance, micro_gemm_up);
            ret_event = PrimitiveImplOCL::execute_stage({ret_event}, instance, micro_gemm_gate);
        }

        // step 4: post proc - gate_up = silu(gate)*up, silu(x)=x*sigmod(x)=x*(1+exp(-x))
        //  input
        //      0: up  [token_len * expert_topK, hidden_size]
        //      1: gate  [token_len * expert_topK, hidden_size]
        // output
        //      0: gate_up  [token_len * expert_topK, hidden_size]
        {
            auto token_size = token_num * max_topk;
#    if DEBUG_MOE_LOG
            GPU_DEBUG_TRACE_DETAIL << "\nstep 4: prefill_swiglu token_size=" << token_size << ", hidden_size=" << _intermediate_size << std::endl;
#    endif
            ret_event = execute_stage({ret_event},
                                      instance,
                                      *prefill_swiglu,
                                      {intermediates_memories[MOE_INTERNAL_BUFFER_UP_OUTPUT], intermediates_memories[MOE_INTERNAL_BUFFER_GATE_OUTPUT]},
                                      {intermediates_memories[MOE_INTERNAL_BUFFER_GATE_OUTPUT]},
                                      {static_cast<size_t>(token_size), static_cast<size_t>(_intermediate_size), 1},
                                      {1, subgroup_size, 1});
        }

        // step 5: moe_gemm for down
        //  input
        //      0: gate_up, shape = [token_len * expert_topK, hidden_size]
        //      1: moe weights
        //      2: expert id, dynamic shape = [activated_expert_num]
        //      3: token start offset idx (input gather tokens) for each activated expert, dynamic shape = [activated_expert_num]
        //      4: token len (input gather tokens) for each activated expert, dynamic shape = [activated_expert_num]
        //      5: m = itermedia_size
        //      6: k = hidden_size
        //      7: wei_scale
        //      8: wei_zp
        //  output:
        //      0: down output, shape = [token_len * expert_topK, hidden_size]
        {
#    if DEBUG_MOE_LOG
            GPU_DEBUG_TRACE_DETAIL << "\nstep 5: moe_gemm for down" << std::endl;
#    endif
            ret_event = PrimitiveImplOCL::execute_stage({ret_event}, instance, micro_gemm_down);
        }

        // step 6: scatter and reduce
        // input:
        //      0: down output, shape = [token_len * expert_topK, hidden_size]
        //      1: experts_per_token, shape = [token_len, expert_topK]
        //      2: expert_weights, shape = [expert_num]
        //      3: tokens_per_expert, shape = [expert_num, ?] = [token_len * expert_topK]
        //      4: experts_start_offset, shape = [activated_expert_num]
        //      5: tokens_len_per_expert,dynamic shape = [activated_expert_num]
        //      6: expert id, dynamic shape = [activated_expert_num]
        // output:
        //      0: final hidden states, shape = [token_len, hidden_size]
        {
            auto token_size = token_num;
            auto [local_threads_count, batches_per_thread, _] = calc_thread_count(const_cast<RuntimeParams&>(*instance.get_impl_params()), 4, _hidden_size);

#    if DEBUG_MOE_LOG
            GPU_DEBUG_TRACE_DETAIL << "\nstep 6: prefill_scatter_reduce token_size=" << token_size << ", local_threads_count=" << local_threads_count
                                   << ", num_actually_used_experts = " << num_actually_used_experts << std::endl;
#    endif

            ret_event = execute_stage({ret_event},
                                      instance,
                                      *prefill_scatter_reduce,
                                      {intermediates_memories[MOE_INTERNAL_BUFFER_DOWN_OUTPUT],
                                       batch_mem_ptr,
                                       routing_mem_ptr,
                                       intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_IDX_PER_EXPERT],
                                       intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_START_OFFSET_PER_EXPERT],
                                       intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_LEN_PER_ACTIVATED_EXPERT],
                                       intermediates_memories[MOE_INTERNAL_BUFFER_ACTIVATED_EXPERT_IDS],
                                       intermediates_memories[MOE_INTERNAL_BUFFER_ACTUAL_USED_EXPERT_NUM]},
                                      {final_hidden_states_mem_ptr},
                                      {static_cast<size_t>(token_size * local_threads_count), 1, 1},
                                      {local_threads_count, 1, 1},
                                      true /*instance.needs_completion_event()*/);
        }

        return ret_event;
    }

    void update_rt_params(const primitive_inst& instance) override {
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<MoE3GemmRuntimeParams>();
        }
        update_stages_flags(instance);
    }

    void update(primitive_inst& inst, const kernel_impl_params& impl_params) override {
        PrimitiveImplOCL::update(inst, impl_params);
        inst.update_shape_info_tensor(impl_params);
        update_rt_params(inst);
    }

    struct onednn_kernel {
        onednn_linear up;
        onednn_linear gate;
        onednn_linear down;
    };
    struct PairHash {
        template <class T1, class T2>
        size_t operator()(const std::pair<T1, T2>& p) const {
            // Combine hash values of the pair elements
            return std::hash<T1>()(p.first) ^ std::hash<T2>()(p.second);
        }
    };

    using lru_cache_hash = LruCache<std::pair<int, int>, std::shared_ptr<onednn_kernel>, PairHash>;
    lru_cache_hash _kernels = lru_cache_hash(1024);
    onednn_kernel& get_kernel(int n_token, int expert_no, typed_primitive_inst<moe_3gemm_fused_compressed>& instance) {
        auto key = std::make_pair(n_token, expert_no);
        if (_kernels.has(key)) {
            return *_kernels.get(key);
        }

        auto& cur_net = instance.get_network();
        auto& stream = cur_net.get_stream();
        auto& dnn_stream = stream.get_onednn_stream();
        auto hidden_states_layout_dt =
            convert_data_type(instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::HIDDEN_STATES))->get_layout().data_type);

        auto& dnnl_weights = _dnnl_weights[expert_no];
        auto kernel = std::make_shared<onednn_kernel>();

        // gate
        auto gate_weight_layout_dt = convert_data_type(instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_0))->get_layout().data_type);
        kernel->gate = onednn_linear::create(dnn_stream.get_engine(),
                                             hidden_states_layout_dt,
                                             gate_weight_layout_dt,
                                             n_token,
                                             dnnl_weights[0].ic,
                                             dnnl_weights[0].oc,
                                             dnnl_weights[0].ic_group_size,
                                             onednn_matmul::type::with_silu_bin_mul,
                                             dnnl_weights[0].weight,
                                             dnnl_weights[0].scale,
                                             dnnl_weights[0].zp);

        // up
        auto up_weight_layout_dt = convert_data_type(instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_1))->get_layout().data_type);
        kernel->up = onednn_linear::create(dnn_stream.get_engine(),
                                           hidden_states_layout_dt,
                                           up_weight_layout_dt,
                                           n_token,
                                           dnnl_weights[1].ic,
                                           dnnl_weights[1].oc,
                                           dnnl_weights[1].ic_group_size,
                                           onednn_matmul::type::none,
                                           dnnl_weights[1].weight,
                                           dnnl_weights[1].scale,
                                           dnnl_weights[1].zp);

        // down
        auto down_weight_layout_dt = convert_data_type(instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_2))->get_layout().data_type);
        kernel->down = onednn_linear::create(dnn_stream.get_engine(),
                                             hidden_states_layout_dt,
                                             down_weight_layout_dt,
                                             n_token,
                                             dnnl_weights[2].ic,
                                             dnnl_weights[2].oc,
                                             dnnl_weights[2].ic_group_size,
                                             onednn_matmul::type::with_bin_mul_per_row,
                                             dnnl_weights[2].weight,
                                             dnnl_weights[2].scale,
                                             dnnl_weights[2].zp);
        _kernels.add(key, kernel);
        return *_kernels.get(key);
    }

    //  inputs 0 is hidden_states, inputs 1 is router_logits[num_tokens, NUM_EXPERTS=128]
    //  extra step Softmax_TopK is fused to give topk-id & router_weights
    //
    //     scratch.topk_id, scratch.full_router_weights = Softmax_TopK(router_logits)
    //
    //  generate expert_mask from topk-id
    //        expert_mask.batch[i][j] : j'th token index for i'th expert
    //        expert_mask.topk[i][j] : topk-output offset for j'th token for i'th expert, used to get weights
    //        expert_mask.pred_flag[i]: bool, if expert i can be skipped
    //
    //     scratch.x, scratch.routing_weights = gather(hidden_states, scratch.full_router_weights, expert_mask.batch, expert_mask.topk)
    //     scratch.y = MLP(scratch.x, .gate/up/down) * scratch.routing_weights
    //     scatter(final_hidden, scratch.y, expert_mask.batch)
    //
    cldnn::event::ptr exec_prefill_onednn(const std::vector<cldnn::event::ptr>& events,
                                          cldnn::stream& stream,
                                          typed_primitive_inst<moe_3gemm_fused_compressed>& instance,
                                          scratch_buffers& scratch) {
        auto cur_moe = instance.get_typed_desc<moe_3gemm_fused_compressed>();
        const auto& config = cur_moe->_config;
        auto& dnn_stream = stream.get_onednn_stream();
        cldnn::event::ptr result_event = nullptr;

        auto [hidden_states_mem_ptr, hidden_states_layout] = get_input_info(instance, static_cast<size_t>(MOE3GemmInputIndex::HIDDEN_STATES));
        auto& engine = instance.get_network().get_engine();
        init_dnnl_weights(cur_moe, engine, scratch.moe_fusion_wei_addr);

        auto routing_mem_ptr = scratch.topk_weights;
        auto final_hidden_states_mem_ptr = instance.output_memory_ptr(0);
        auto get_best_lws = [](size_t hidden_size) {
            const size_t candidate[] = {128, 64, 32, 16, 8};
            for (size_t i = 0; i < sizeof(candidate) / sizeof(size_t); i++) {
                if (hidden_size % candidate[i] == 0) {
                    return candidate[i];
                }
            }
            OPENVINO_THROW("hidden_size=", hidden_size, " is not divisible by any of ", sizeof(candidate) / sizeof(size_t), " candidates");
        };
        auto lws_size = get_best_lws(_hidden_size);
        int max_topk = static_cast<int>(config.top_k);

        // [batch, max_topk]
        auto topk_id_mem = scratch.topk_id;
        expert_mask_cpu expert_mask;
        get_expert_mask_from_gpu(config, topk_id_mem, stream, expert_mask);

        for (size_t expert_no = 0; expert_no < config.num_expert; expert_no++) {
            if (expert_no >= expert_mask.pred_flag.size()) {
                OPENVINO_THROW("expert_no=", expert_no, " is out of bounds");
            }
            auto can_skip_subgraph = !expert_mask.pred_flag[expert_no];
            if (can_skip_subgraph) {
                continue;
            }
            auto& dnnl_weights = _dnnl_weights[expert_no];

            // expert_mask
            expert_mask_gpu& expert_mask_mem = scratch.expert_masks[expert_no];
            copy_expert_mask_to_gpu(stream, expert_mask, expert_no, expert_mask_mem);

            auto n_token = static_cast<int>(expert_mask.batch[expert_no].size());
            onednn_kernel& kernel = get_kernel(n_token, static_cast<int>(expert_no), instance);

            // gather
            result_event = execute_stage({result_event},
                                         instance,
                                         *gather,
                                         {hidden_states_mem_ptr, routing_mem_ptr, expert_mask_mem.batch, expert_mask_mem.topk},
                                         {scratch.x, scratch.routing_weights},
                                         {static_cast<size_t>(n_token), static_cast<size_t>(_hidden_size)},
                                         {1, lws_size},
                                         instance.needs_completion_event());

            // up
            kernel.up.forward(dnn_stream,
                              n_token,
                              convert2dnnl(scratch.x, {static_cast<int>(n_token), dnnl_weights[1].ic}, dnnl::memory::format_tag::ab),
                              convert2dnnl(scratch.up, {static_cast<int>(n_token), _intermediate_size}, dnnl::memory::format_tag::ab),
                              dnnl::memory());

            // gate
            kernel.gate.forward(dnn_stream,
                                n_token,
                                convert2dnnl(scratch.x, {static_cast<int>(n_token), dnnl_weights[0].ic}, dnnl::memory::format_tag::ab),
                                convert2dnnl(scratch.gate, {static_cast<int>(n_token), _intermediate_size}, dnnl::memory::format_tag::ab),
                                convert2dnnl(scratch.up, {static_cast<int>(n_token), _intermediate_size}, dnnl::memory::format_tag::ab));

            // down
            kernel.down.forward(dnn_stream,
                                n_token,
                                convert2dnnl(scratch.gate, {static_cast<int>(n_token), _intermediate_size}, dnnl::memory::format_tag::ab),
                                convert2dnnl(scratch.y, {static_cast<int>(n_token), _hidden_size}, dnnl::memory::format_tag::ab),
                                convert2dnnl(scratch.routing_weights, {static_cast<int>(n_token * max_topk)}, dnnl::memory::format_tag::a));

            // index_add
            result_event = execute_stage({result_event},
                                         instance,
                                         *scatter,
                                         {scratch.y, expert_mask_mem.batch},
                                         {final_hidden_states_mem_ptr},
                                         {static_cast<size_t>(n_token), static_cast<size_t>(_hidden_size)},
                                         {1, lws_size},
                                         true /*instance.needs_completion_event()*/);
        }

        return result_event;
    }

    cldnn::event::ptr execute(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& ins) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("moe_3gemm_swiglu_opt_impl::execute"));
        auto& instance = reinterpret_cast<typed_primitive_inst<moe_3gemm_fused_compressed>&>(ins);
        auto cur_moe = instance.get_typed_desc<moe_3gemm_fused_compressed>();
        const auto& config = cur_moe->_config;
        auto& cur_net = instance.get_network();
        auto& stream = cur_net.get_stream();
        cldnn::event::ptr ret_env = nullptr;

        auto [hidden_states_mem_ptr, hidden_states_layout] = get_input_info(instance, static_cast<size_t>(MOE3GemmInputIndex::HIDDEN_STATES));
        size_t token_num = get_seq_len(hidden_states_layout);
        scratch_buffers scratch;
        prepare_internal_buffers(instance, scratch, token_num);

        // softmax+topk
        auto lws_size = config.num_expert;
        auto topk_event = execute_stage(events,
                                        instance,
                                        *softmax_topk,
                                        {instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::ROUTING_WEIGHTS))},
                                        {scratch.topk_id, scratch.topk_weights},
                                        {static_cast<size_t>(token_num), lws_size},
                                        {1, lws_size},
                                        instance.needs_completion_event());

        // Single token is a special case, we don't need to do gather/scatter,
        // and we can apply optimal kernels against memory bound to improve performance.
        if (token_num == 1) {
            return exec_single_token({topk_event}, instance, scratch);
        }

        // onednn path will accumulate to the output
        if (!use_micro_gemm_prefill) {
            auto final_hidden_states_mem_ptr = instance.output_memory_ptr(0);
            final_hidden_states_mem_ptr->fill(stream, false);
        }
        const bool use_gpu_mask_gen = use_gpu_mask_gen_prefill;
        if (!use_gpu_mask_gen) {
            // Wait for topk is ready
            topk_event->wait();
        }

        GPU_DEBUG_TRACE_DETAIL << "\nMoE3GemmFusedCompressed exec(): token_num=" << token_num << ", max_topk=" << static_cast<int>(config.top_k)
                               << ", use_micro_gemm_prefill=" << use_micro_gemm_prefill << std::endl;
        update_rt_params(instance);
        if (use_micro_gemm_prefill) {
            ret_env = exec_prefill_micro_gemm({topk_event}, instance, scratch, use_gpu_mask_gen);
        } else {
            ret_env = exec_prefill_onednn({topk_event}, stream, instance, scratch);
        }
        // Wait for the final event to be ready
        // ret_env->wait();
        return ret_env;
    }
};

}  // namespace

std::unique_ptr<primitive_impl> moe_3gemm_swiglu_opt::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<moe_3gemm_fused_compressed>());
    return std::make_unique<moe_3gemm_swiglu_opt_impl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::moe_3gemm_fused_compressed)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::moe_3gemm_swiglu_opt_impl)

#else

namespace ov::intel_gpu::ocl {

std::unique_ptr<primitive_impl> moe_3gemm_swiglu_opt::create_impl(const program_node& node, const RuntimeParams& params) const {
    OPENVINO_THROW("moe_3gemm_swiglu_opt depends on onednn.");
}

}  // namespace ov::intel_gpu::ocl

#endif
