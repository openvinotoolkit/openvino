// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_opt.hpp"

#ifdef ENABLE_ONEDNN_FOR_GPU
#    include <initializer_list>
#    include <oneapi/dnnl/dnnl.hpp>
#    include <oneapi/dnnl/dnnl_ocl.hpp>
#    include <sstream>
#    include <string_view>
#    include <tuple>
#    include <utility>

#    include "cm/utils/kernel_generator.hpp"
#    include "common_utils/jitter.hpp"
#    include "debug_helper.hpp"
#    include "intel_gpu/graph/kernel_impl_params.hpp"
#    include "intel_gpu/primitives/moe.hpp"
#    include "intel_gpu/runtime/lru_cache.hpp"
#    include "intel_gpu/runtime/stream.hpp"
#    include "intel_gpu/runtime/utils.hpp"
#    include "moe_inst.h"
#    include "ocl_v2/utils/fused_ops_jitter.hpp"
#    include "ocl_v2/utils/jitter.hpp"
#    include "primitive_inst.h"
#    include "primitive_ocl_base.hpp"
#    include "utils/kernel_generator.hpp"

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
            OPENVINO_ASSERT(m_K_groups = (m_K / k_group_size));
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
            // ECOUT("make_cacheable re-constructed: ", typeid(T).name(), "(", cargs..., ")");
            wptr = sptr;
        }
    } else {
        sptr = std::make_shared<T>(eng, cargs...);
        // ECOUT("make_cacheable constructed: ", typeid(T).name(), "(", cargs..., ")");
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
            auto wei_scale_md = dnnl::memory::desc(dnnl::memory::dims({mm->m_K_groups, mm->m_N}), dnnl::memory::data_type::f16, dnnl::memory::format_tag::ab);
            linear.scale = scale;  // dnnl::ocl_interop::make_memory(wei_scale_md, linear.m_engine, dnnl::ocl_interop::memory_kind::usm, scale);
            if (zp) {
                auto wei_zp_md = dnnl::memory::desc(dnnl::memory::dims({mm->m_K_groups, mm->m_N}), mm->m_w_type, dnnl::memory::format_tag::ab);
                linear.zp = zp;  // dnnl::ocl_interop::make_memory(wei_zp_md, linear.m_engine, dnnl::ocl_interop::memory_kind::usm, zp);
            }
        }
        return linear;
    }

    void forward(dnnl::stream& stream, int m, dnnl::memory src_mem, dnnl::memory dst_mem, dnnl::memory bin_mem) {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("onednn_linear::forward()"));
        dnnl::memory::dim M = m;

        OPENVINO_ASSERT(m_batch == 0 || m_batch == M, "m_batch=", m_batch, " M=", M);

        dnnl::memory::desc rt_src_md = dnnl::memory::desc(dnnl::memory::dims({M, m_K}), m_a_type, dnnl::memory::format_tag::ab);
        dnnl::memory::desc rt_dst_md = dnnl::memory::desc(dnnl::memory::dims({M, m_N}), m_a_type, dnnl::memory::format_tag::ab);
        dnnl::memory::desc rt_bin_md;
        if (mm->bin_per_row) {
            rt_bin_md = dnnl::memory::desc(dnnl::memory::dims({M, 1}), m_a_type, dnnl::memory::format_tag::ab);
        } else {
            rt_bin_md = dnnl::memory::desc(dnnl::memory::dims({M, m_N}), m_a_type, dnnl::memory::format_tag::ab);
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
            // auto bin_mem = dnnl::ocl_interop::make_memory(rt_bin_md, m_engine, dnnl::ocl_interop::memory_kind::usm, (void *)(bin_input));
            args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(bin_post_id) | DNNL_ARG_SRC_1, bin_mem});
        }
        m_prim.execute(stream, args);
    }
};

class MOEOptSoftMaxTopK : public KernelGenerator {
public:
    MOEOptSoftMaxTopK() : KernelGenerator("moe_opt", "softmax_topk") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<moe>();
        jit.make("SOFTMAX_TOPK_ENABLE", 1);
        jit.make("TOP_K", desc->_config.topk);
        jit.make("VALUE_NUM", desc->_config.expert_num);
        jit.make("TYPE", params.get_input_layout(0).data_type == ov::element::f16 ? "half" : "float");
        jit.make("TYPE_SIZE", params.get_input_layout(0).data_type == ov::element::f16 ? 2 : 4);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{nullptr};
    }
};

class MOEOptGather : public KernelGenerator {
public:
    MOEOptGather() : KernelGenerator("moe_opt", "gather") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<moe>();
        jit.make("GATHER_ENABLE", 1);
        jit.make("HIDDEN_SIZE", desc->_config.hidden_size);
        jit.make("TYPE", params.get_input_layout(0).data_type == ov::element::f16 ? "half" : "float");
        jit.make("TYPE_SIZE", params.get_input_layout(0).data_type == ov::element::f16 ? 2 : 4);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{nullptr};
    }
};

class MOEOptScatter : public KernelGenerator {
public:
    MOEOptScatter() : KernelGenerator("moe_opt", "index_add") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<moe>();
        jit.make("SCATTER_ENABLE", 1);
        jit.make("HIDDEN_SIZE", desc->_config.hidden_size);
        jit.make("TYPE", params.get_input_layout(0).data_type == ov::element::f16 ? "half" : "float");
        jit.make("TYPE_SIZE", params.get_input_layout(0).data_type == ov::element::f16 ? 2 : 4);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{nullptr};
    }
};

#    define N_BLOCK      4
#    define SUBGROUP_NUM 8

static void add_common_consts(const RuntimeParams& params, JitConstants& jit) {
    auto desc = params.typed_desc<moe>();
    auto& engine = params.prog->get_engine();
    const auto& info = engine.get_device_info();
    jit.make("MAX_TOPK", desc->_config.topk);
    jit.make("EXPERT_NUM", desc->_config.expert_num);
    jit.make("HIDDEN_SIZE", desc->_config.hidden_size);
    jit.make("INTERMEDIATE_SIZE", desc->_config.intermediate_size);
    jit.make("N_BLOCK", N_BLOCK);
    jit.make("SUBGROUP_SIZE", info.arch >= gpu_arch::xe2 ? 32 : 16);
    jit.make("SUBGROUP_NUM", SUBGROUP_NUM);
    jit.make("GROUP_SIZE", desc->_config.group_size);
    jit.make("TYPE", params.get_input_layout(0).data_type == ov::element::f16 ? "half" : "float");
    jit.make("TYPE_SIZE", params.get_input_layout(0).data_type == ov::element::f16 ? 2 : 4);
}

class MOEOptMLPGateUp : public KernelGenerator {
public:
    MOEOptMLPGateUp() : KernelGenerator("moe_mlp", "gate_up") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<moe>();
        add_common_consts(params, jit);
        jit.make("GATE_UP_ENABLE", 1);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{nullptr};
    }
};

class MOEOptMLPDown : public KernelGenerator {
public:
    MOEOptMLPDown() : KernelGenerator("moe_mlp", "down") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<moe>();
        add_common_consts(params, jit);
        jit.make("DOWN_ENABLE", 1);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{nullptr};
    }
};

class MOEOptMLPReduce : public KernelGenerator {
public:
    MOEOptMLPReduce() : KernelGenerator("moe_mlp", "reduce") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<moe>();
        add_common_consts(params, jit);
        jit.make("REDUCE_ENABLE", 1);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{nullptr};
    }
};

dnnl::memory convert2dnnl(const memory::ptr& ptr, const std::vector<int64_t>& dim, dnnl::memory::format_tag tag, int offset = 0) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("convert2dnnl"));
    return ptr->get_onednn_memory(dnnl::memory::desc(dnnl::memory::dims(dim), convert_data_type(ptr->get_layout().data_type), tag), offset);
}

class MOEOptImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::MOEOptImpl)
    Stage::Ptr softmax_topk = make_stage<MOEOptSoftMaxTopK>();
    Stage::Ptr gather = make_stage<MOEOptGather>();
    Stage::Ptr scatter = make_stage<MOEOptScatter>();
    Stage::Ptr mlp_gate_up = make_stage<MOEOptMLPGateUp>();
    Stage::Ptr mlp_down = make_stage<MOEOptMLPDown>();
    Stage::Ptr mlp_reduce = make_stage<MOEOptMLPReduce>();

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
    };

    std::vector<std::vector<dnnl_weights>> _dnnl_weights;
    int _hidden_size;
    int _intermediate_size;
    int _group_size;

    MOEOptImpl() : PrimitiveImplOCL(MOEOpt::get_type_info_static()) {}
    MOEOptImpl(const program_node& node, const RuntimeParams& params) : MOEOptImpl() {
        init(node.as<moe>().get_primitive());

        add_stage(softmax_topk, params);
        add_stage(gather, params);
        add_stage(scatter, params);
        add_stage(mlp_gate_up, params);
        add_stage(mlp_down, params);
        add_stage(mlp_reduce, params);
    }

    void init(const std::shared_ptr<const moe>& cur_moe) {
        const auto& moe_mlp_params = cur_moe->_mlp_params;
        _dnnl_weights.resize(moe_mlp_params.size());
        _hidden_size = static_cast<int>(cur_moe->_config.hidden_size);
        _intermediate_size = static_cast<int>(cur_moe->_config.intermediate_size);
        _group_size = static_cast<int>(cur_moe->_config.group_size);

        for (size_t j = 0; j < moe_mlp_params.size(); j++) {
            const auto& mlp_params = moe_mlp_params[j];
            auto& dnnl_weights = _dnnl_weights[j];
            dnnl_weights.resize(3);
            dnnl_weights[0].ic = _hidden_size;
            dnnl_weights[0].ic_group_size = _group_size;
            dnnl_weights[0].oc = _intermediate_size;
            dnnl_weights[1].ic = _hidden_size;
            dnnl_weights[1].ic_group_size = _group_size;
            dnnl_weights[1].oc = _intermediate_size;
            dnnl_weights[2].ic = _intermediate_size;
            dnnl_weights[2].ic_group_size = _group_size;
            dnnl_weights[2].oc = _hidden_size;
            for (int i = 0; i < 3; i++) {
                if (mlp_params.param[i].scale) {
                    dnnl_weights[i].scale = convert2dnnl(mlp_params.param[i].scale,
                                                         {dnnl_weights[i].ic / dnnl_weights[i].ic_group_size, dnnl_weights[i].oc},
                                                         dnnl::memory::format_tag::ab);
                }
                if (mlp_params.param[i].zp) {
                    dnnl_weights[i].zp = convert2dnnl(mlp_params.param[i].zp,
                                                      {dnnl_weights[i].ic / dnnl_weights[i].ic_group_size, dnnl_weights[i].oc},
                                                      dnnl::memory::format_tag::ab);
                }
                if (mlp_params.param[i].weight) {
                    dnnl_weights[i].weight = convert2dnnl(mlp_params.param[i].weight, {dnnl_weights[i].ic, dnnl_weights[i].oc}, dnnl::memory::format_tag::ba);
                }
            }
        }
    }

    void load(BinaryInputBuffer& ib) override {
        PrimitiveImplOCL::load(ib);
        const kernel_impl_params* impl_params = reinterpret_cast<kernel_impl_params*>(ib.getKernelImplParams());
        init(impl_params->typed_desc<moe>());
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        auto cur_moe = make_deep_copy<MOEOptImpl>(this);
        cur_moe->_dnnl_weights = _dnnl_weights;
        cur_moe->_hidden_size = _hidden_size;
        cur_moe->_intermediate_size = _intermediate_size;
        cur_moe->_group_size = _group_size;
        return cur_moe;
    }

    std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        auto cur_moe = params.typed_desc<moe>();
        const auto& config = cur_moe->_config;
        int max_topk = static_cast<int>(config.topk);
        int expert_num = static_cast<int>(config.expert_num);

        auto hidden_states_layout = params.input_layouts[0];
        auto batch = static_cast<int>(hidden_states_layout.get_shape()[0]);
        auto data_type = hidden_states_layout.data_type;

        std::vector<BufferDescriptor> internal_buffers;
        // softmax+topk
        layout layout_topk_id(ov::PartialShape{batch, max_topk}, data_types::u32, cldnn::format::bfyx);
        layout layout_topk_weights(ov::PartialShape{batch, max_topk}, data_type, cldnn::format::bfyx);
        internal_buffers.emplace_back(layout_topk_id, true);       // topk_id
        internal_buffers.emplace_back(layout_topk_weights, true);  // topk_weights
        // fast single batch: scratch.up = up(x) * silu(gate(x)); scratch.y = down(scratch.up) * weight[expert_no]
        layout layout_gateup_out(ov::PartialShape{batch, static_cast<int>(config.intermediate_size)}, data_type, cldnn::format::bfyx);
        layout layout_down_out(ov::PartialShape{batch, static_cast<int>(config.hidden_size)}, data_type, cldnn::format::bfyx);
        internal_buffers.emplace_back(layout_gateup_out, true);  // up
        internal_buffers.emplace_back(layout_down_out, true);    // y
        // onednn: scratch.x, scratch.routing_weights = gather(x, ...)
        //         scratch.up = up(scratch.x)
        //         scratch.gate = gate(scratch.x) * scratch.up
        //         scratch.y = down(scratch.gate) * routing_weights
        internal_buffers.emplace_back(layout_down_out, true);  // x, scratch.x has same layout with down output
        layout routing_layout(ov::PartialShape{batch * max_topk}, data_type, cldnn::format::bfyx);
        internal_buffers.emplace_back(layout_down_out, true);    // routing_weights
        internal_buffers.emplace_back(layout_gateup_out, true);  // gate, scratch.gate has same layout with up
        // expert masks for gpu
        layout index_layout(ov::PartialShape{batch}, ov::element::i32, cldnn::format::bfyx);
        for (int i = 0; i < expert_num; i++) {
            internal_buffers.emplace_back(index_layout, true);  // batch
            internal_buffers.emplace_back(index_layout, true);  // topk
        }

        return internal_buffers;
    }

    void prepare_internal_buffers(typed_primitive_inst<moe>& instance, scratch_buffers& scratch, bool is_single_batch) {
        const auto& intermediates_memories = instance.get_intermediates_memories();
        scratch.topk_id = intermediates_memories[0];
        scratch.topk_weights = intermediates_memories[1];
        scratch.up = intermediates_memories[2];
        scratch.y = intermediates_memories[3];
        if (!is_single_batch) {
            scratch.x = intermediates_memories[4];
            scratch.routing_weights = intermediates_memories[5];
            scratch.gate = intermediates_memories[6];
            const auto& config = instance.get_typed_desc<moe>()->_config;
            int expert_num = static_cast<int>(config.expert_num);
            scratch.expert_masks.resize(expert_num);
            for (int i = 0; i < expert_num; i++) {
                scratch.expert_masks[i].batch = intermediates_memories[7 + 2 * i + 0];
                scratch.expert_masks[i].topk = intermediates_memories[7 + 2 * i + 1];
            }
        }
    }

    void get_expert_mask_from_gpu(const MOE::Config& config, memory::ptr mem, stream& stream, expert_mask_cpu& expert_mask) {
        // shape: [batch, topk]
        auto layout = mem->get_layout();
        const auto& shape = layout.get_shape();

        int max_expert_num = static_cast<int>(config.expert_num), max_topk = static_cast<int>(config.topk), max_tokens = static_cast<int>(shape[0]);

        expert_mask.pred_flag.resize(max_expert_num, 0);
        expert_mask.batch.resize(max_expert_num, {});
        expert_mask.topk.resize(max_expert_num, {});

        OPENVINO_ASSERT(!layout.data_padding, "get_expert_mask_from_memory not support padding");

        std::vector<int32_t> buf(max_topk * max_tokens);
        mem->copy_to(stream, buf.data(), 0, 0, buf.size() * sizeof(int32_t), true);

        for (int b = 0; b < max_tokens; b++) {
            auto* tok_p = &buf[b * max_topk];
            for (int t = 0; t < max_topk; t++) {
                auto expert_no = tok_p[t];
                OPENVINO_ASSERT(expert_no < max_expert_num);
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

        {
            mem_lock<int32_t, mem_lock_type::write> lock_data{expert_mask_mem.batch, stream};
            memcpy(lock_data.data(), expert_mask.batch[expert_no].data(), size);
        }
        {
            mem_lock<int32_t, mem_lock_type::write> lock_data{expert_mask_mem.topk, stream};
            memcpy(lock_data.data(), expert_mask.topk[expert_no].data(), size);
        }
    }

    cldnn::event::ptr execute_stage(const std::vector<cldnn::event::ptr>& events,
                                    cldnn::primitive_inst& instance,
                                    Stage& stage,
                                    std::vector<memory::ptr> inputs,
                                    std::vector<memory::ptr> outputs,
                                    const std::vector<size_t>& global,
                                    const std::vector<size_t>& local,
                                    bool needs_completion_event = false) const {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("MOEOptImpl::execute_stage"));
        cldnn::stream& stream = instance.get_network().get_stream();
        cldnn::kernel_arguments_data args;
        cldnn::kernel_arguments_desc desc;
        for (uint32_t i = 0; i < inputs.size(); i++) {
            desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, i});
            args.inputs.push_back(inputs[i]);
        }

        for (uint32_t i = 0; i < outputs.size(); i++) {
            desc.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, i});
            args.outputs.push_back(outputs[i]);
        }

        stream.set_arguments(*stage.kernel, desc, args);
        desc.workGroups.global = global;
        desc.workGroups.local = local;

        return stream.enqueue_kernel(*stage.kernel, desc, {}, events, needs_completion_event);
    }

    auto get_input_info(typed_primitive_inst<moe>& instance, int idx) {
        auto mem = instance.input_memory_ptr(idx);
        auto dep = instance.dependencies()[idx];
        auto layout = dep.first->get_impl_params()->get_output_layout(dep.second);
        return std::make_tuple(mem, layout);
    }

    cldnn::event::ptr exec_single_batch(typed_primitive_inst<moe>& instance, scratch_buffers& scratch) {
        auto cur_moe = instance.get_typed_desc<moe>();
        int max_topk = static_cast<int>(cur_moe->_config.topk);

        auto final_hidden_states_mem_ptr = instance.output_memory_ptr(0);
        auto batch_mem_ptr = scratch.topk_id;
        auto [hidden_states_mem_ptr, hidden_states_layout] = get_input_info(instance, 0);
        auto routing_mem_ptr = scratch.topk_weights;

        _hidden_size = static_cast<int>(cur_moe->_config.hidden_size);
        _intermediate_size = static_cast<int>(cur_moe->_config.intermediate_size);

        const size_t subgroup_size = instance.get_impl_params()->get_device_info().arch >= gpu_arch::xe2 ? 32 : 16;
        const size_t max_work_group_size = instance.get_impl_params()->get_device_info().max_work_group_size;
        const auto& mlp_weight_mem = cur_moe->_mlp_weights_mem;
        event::ptr ret;

        {
            // scratch.up = up(x) * silu(gate(x))
            execute_stage({},
                          instance,
                          *mlp_gate_up,
                          {batch_mem_ptr, mlp_weight_mem.weights_base, mlp_weight_mem.weights_offset, hidden_states_mem_ptr},
                          {scratch.up},
                          {static_cast<size_t>(max_topk), subgroup_size, static_cast<size_t>(_intermediate_size / N_BLOCK)},
                          {1, subgroup_size, SUBGROUP_NUM});

            // scratch.y = down(scratch.up) * weight[expert_no]
            execute_stage({},
                          instance,
                          *mlp_down,
                          {batch_mem_ptr, mlp_weight_mem.weights_base, mlp_weight_mem.weights_offset, scratch.up, routing_mem_ptr},
                          {scratch.y},
                          {static_cast<size_t>(max_topk), subgroup_size, static_cast<size_t>(_hidden_size / N_BLOCK)},
                          {1, subgroup_size, SUBGROUP_NUM});

            // final = sum(scratch.y)
            ret = execute_stage({},
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
    onednn_kernel& get_kernel(int n_token, int expert_no, typed_primitive_inst<moe>& instance) {
        auto key = std::make_pair(n_token, expert_no);
        if (_kernels.has(key)) {
            return *_kernels.get(key);
        }

        auto& cur_net = instance.get_network();
        auto& stream = cur_net.get_stream();
        auto cur_moe = instance.get_typed_desc<moe>();
        const auto& moe_mlp_params = cur_moe->_mlp_params;
        const auto& mlp_params = moe_mlp_params[expert_no];
        auto& dnn_stream = stream.get_onednn_stream();
        auto hidden_states_layout_dt = convert_data_type(instance.input_memory_ptr(0)->get_layout().data_type);
        auto& dnnl_weights = _dnnl_weights[expert_no];
        auto kernel = std::make_shared<onednn_kernel>();
        // up
        auto up_weight_layout = mlp_params.param[1].weight->get_layout();
        kernel->up = onednn_linear::create(dnn_stream.get_engine(),
                                           hidden_states_layout_dt,
                                           convert_data_type(up_weight_layout.data_type),
                                           n_token,
                                           dnnl_weights[1].ic,
                                           dnnl_weights[1].oc,
                                           dnnl_weights[1].ic_group_size,
                                           onednn_matmul::type::none,
                                           dnnl_weights[1].weight,
                                           dnnl_weights[1].scale,
                                           dnnl_weights[1].zp);

        // gate
        auto gate_weight_layout = mlp_params.param[0].weight->get_layout();
        kernel->gate = onednn_linear::create(dnn_stream.get_engine(),
                                             hidden_states_layout_dt,
                                             convert_data_type(gate_weight_layout.data_type),
                                             n_token,
                                             dnnl_weights[0].ic,
                                             dnnl_weights[0].oc,
                                             dnnl_weights[0].ic_group_size,
                                             onednn_matmul::type::with_silu_bin_mul,
                                             dnnl_weights[0].weight,
                                             dnnl_weights[0].scale,
                                             dnnl_weights[0].zp);

        // down
        auto down_weight_layout = mlp_params.param[2].weight->get_layout();
        kernel->down = onednn_linear::create(dnn_stream.get_engine(),
                                             hidden_states_layout_dt,
                                             convert_data_type(down_weight_layout.data_type),
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
    //
    //     scratch.x, scratch.routing_weights = gather(hidden_states, scratch.full_router_weights, expert_mask.batch, expert_mask.topk)
    //     scratch.y = MLP(scratch.x, .gate/up/down) * scratch.routing_weights
    //     scatter(final_hidden, scratch.y, expert_mask.batch)
    //
    cldnn::event::ptr execute(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& ins) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("MOEOptImpl::execute"));
        auto& instance = reinterpret_cast<typed_primitive_inst<moe>&>(ins);
        auto cur_moe = instance.get_typed_desc<moe>();
        const auto& config = cur_moe->_config;
        int max_topk = static_cast<int>(config.topk);
        auto& cur_net = instance.get_network();
        auto& stream = cur_net.get_stream();

        auto [hidden_states_mem_ptr, hidden_states_layout] = get_input_info(instance, 0);
        auto batch = static_cast<int>(hidden_states_layout.get_shape()[0]);

        scratch_buffers scratch;
        prepare_internal_buffers(instance, scratch, batch == 1);

        // softmax+topk
        auto lws_size = cur_moe->_config.expert_num;
        auto topk_event = execute_stage(events,
                                        instance,
                                        *softmax_topk,
                                        {instance.input_memory_ptr(1)},
                                        {scratch.topk_id, scratch.topk_weights},
                                        {static_cast<size_t>(batch), lws_size},
                                        {1, lws_size});

        // Single batch is a special case, we don't need to do gather/scatter,
        // and we can apply optimal kernels against memory bound to improve performance.
        // It is very important for MoE's second token performance.
        if (batch == 1) {
            return exec_single_batch(instance, scratch);
        }

        auto final_hidden_states_mem_ptr = instance.output_memory_ptr(0);
        auto final_hidden_states_layout = instance.get_output_layout(0);

        // onednn path will accumulate to the output
        final_hidden_states_mem_ptr->fill(stream, false);

        // Wait for topk is ready
        topk_event->wait();
        // [batch, max_topk]
        auto topk_id_mem = scratch.topk_id;
        expert_mask_cpu expert_mask;
        get_expert_mask_from_gpu(config, topk_id_mem, stream, expert_mask);

        auto& dnn_stream = stream.get_onednn_stream();
        cldnn::event::ptr result_event;

        auto routing_mem_ptr = scratch.topk_weights;
        auto get_best_lws = [](size_t hidden_size) {
            const size_t candidate[] = {128, 64, 32, 16, 8};
            for (size_t i = 0; i < sizeof(candidate) / sizeof(size_t); i++) {
                if (hidden_size % candidate[i] == 0) {
                    return candidate[i];
                }
            }
            OPENVINO_ASSERT(false, "hidden_size=", hidden_size, " is not divisible by any of ", sizeof(candidate) / sizeof(size_t), " candidates");
        };
        lws_size = get_best_lws(_hidden_size);

        OPENVINO_ASSERT(batch != 1, "batch size shouldn't be 1 for this path!");
        for (size_t expert_no = 0; expert_no < config.expert_num; expert_no++) {
            OPENVINO_ASSERT(expert_no < expert_mask.pred_flag.size());
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
            memory::ptr& x = scratch.x;

            // gather
            execute_stage(events,
                          instance,
                          *gather,
                          {hidden_states_mem_ptr, routing_mem_ptr, expert_mask_mem.batch, expert_mask_mem.topk},
                          {x, scratch.routing_weights},
                          {static_cast<size_t>(n_token), static_cast<size_t>(_hidden_size)},
                          {1, lws_size});

            // up
            kernel.up.forward(dnn_stream,
                              n_token,
                              convert2dnnl(x, {static_cast<int>(n_token), dnnl_weights[1].ic}, dnnl::memory::format_tag::ab),
                              convert2dnnl(scratch.up, {static_cast<int>(n_token), _intermediate_size}, dnnl::memory::format_tag::ab),
                              dnnl::memory());

            // gate
            kernel.gate.forward(dnn_stream,
                                n_token,
                                convert2dnnl(x, {static_cast<int>(n_token), dnnl_weights[0].ic}, dnnl::memory::format_tag::ab),
                                convert2dnnl(scratch.gate, {static_cast<int>(n_token), _intermediate_size}, dnnl::memory::format_tag::ab),
                                convert2dnnl(scratch.up, {static_cast<int>(n_token), _intermediate_size}, dnnl::memory::format_tag::ab));

            // down
            kernel.down.forward(dnn_stream,
                                n_token,
                                convert2dnnl(scratch.gate, {static_cast<int>(n_token), _intermediate_size}, dnnl::memory::format_tag::ab),
                                convert2dnnl(scratch.y, {static_cast<int>(n_token), _hidden_size}, dnnl::memory::format_tag::ab),
                                convert2dnnl(scratch.routing_weights, {n_token * max_topk}, dnnl::memory::format_tag::a));
            // index_add
            result_event = execute_stage(events,
                                         instance,
                                         *scatter,
                                         {scratch.y, expert_mask_mem.batch},
                                         {final_hidden_states_mem_ptr},
                                         {static_cast<size_t>(n_token), static_cast<size_t>(_hidden_size)},
                                         {1, lws_size},
                                         instance.needs_completion_event());
        }

        return result_event;
    }
};

}  // namespace

std::unique_ptr<primitive_impl> MOEOpt::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<moe>());
    return std::make_unique<MOEOptImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::moe)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::MOEOptImpl)

#else

namespace ov::intel_gpu::ocl {

std::unique_ptr<primitive_impl> MOEOpt::create_impl(const program_node& node, const RuntimeParams& params) const {
    OPENVINO_THROW("MOEOpt depends on onednn.");
}

}  // namespace ov::intel_gpu::ocl

#endif