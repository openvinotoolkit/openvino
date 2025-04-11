// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "moe_expert_opt.hpp"

#include <initializer_list>
#include <string_view>

#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/moe_expert.hpp"
#include "ocl_v2/utils/fused_ops_jitter.hpp"
#include "ocl_v2/utils/jitter.hpp"
#include "primitive_inst.h"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_ocl.hpp>
#include <tuple>
#include "intel_gpu/runtime/utils.hpp"
#include "moe_expert_inst.h"

namespace ov::intel_gpu::ocl {

namespace {

using namespace ov::intel_gpu::ocl;

dnnl::memory::data_type convert_data_type(cldnn::data_types dt) {
    switch (dt) {
        case cldnn::data_types::f32: return dnnl::memory::data_type::f32;
        case cldnn::data_types::f16: return dnnl::memory::data_type::f16;
        case cldnn::data_types::i8: return dnnl::memory::data_type::s8;
        case cldnn::data_types::u8: return dnnl::memory::data_type::u8;
        case cldnn::data_types::i32: return dnnl::memory::data_type::s32;
        case cldnn::data_types::i4: return dnnl::memory::data_type::s4;
        case cldnn::data_types::u4: return dnnl::memory::data_type::u4;
        default: throw std::invalid_argument("[clDNN] Unsupported conversion from cldnn to onednn type");
    }
}

struct onednn_matmul {
    dnnl::matmul m_prim;
    dnnl::memory::desc m_wei_md;
    dnnl::memory::data_type m_w_type;
    dnnl::memory::data_type m_a_type; // activation dtype
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
            batch_size = 1024*1024; // big enough fake static batch

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
        //memory::desc wei_md = memory::desc(memory::dims({m_K, m_N}), m_w_type, memory::format_tag::any);

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
    onednn_matmul(dnnl::engine eng, dnnl::memory::data_type act_dtype, dnnl::memory::data_type weight_dtype, int batch, int ic, int oc, int ic_group_size, type t) : onednn_matmul(act_dtype, weight_dtype, batch, ic, oc, ic_group_size) {
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
// to optimze compile-time-workload itself, the functor instance itself should be
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

    static onednn_linear create(
              dnnl::engine eng,
              dnnl::memory::data_type act_dtype, dnnl::memory::data_type weight_dtype, int batch, int ic, int oc, int ic_group_size, onednn_matmul::type t,
              dnnl::memory weight, // external weight
              dnnl::memory scale,
              dnnl::memory zp) {
        auto mm = make_cacheable<onednn_matmul>(eng, act_dtype, weight_dtype, batch, ic, oc, ic_group_size, t);
        onednn_linear linear;
        linear.mm = mm;
        linear.bin_post_id = mm->bin_post_id;
        linear.m_prim = mm->m_prim;
        linear.m_K = mm->m_K;
        linear.m_N = mm->m_N;
        linear.m_batch = batch;
        linear.m_a_type = mm->m_a_type;

        // assume raw weights are nn.Linear
        // dnnl::memory::desc raw_wei_md = dnnl::memory::desc(dnnl::memory::dims({mm->m_K, mm->m_N}), dtype, dnnl::memory::format_tag::ba);

        // if (raw_wei_md != mm->m_wei_md) {
        //    OPENVINO_ASSERT(0);
            /*
            linear.weight = memory(mm->m_wei_md, mm->m_engine);
            std::cout << ">>>>>>>>>>>>>>>>>> weight layout changed : reorder is called (seems to be not working)" << std::endl;
            auto src_wei_mem = dnnl::ocl_interop::make_memory(
                                        raw_wei_md,
                                        mm->m_engine,
                                        ocl_interop::memory_kind::usm,
                                        static_cast<void*>(data));
            reorder cvt(src_wei_mem, linear.weight);
            cvt.execute(linear.m_stream, src_wei_mem, linear.weight);
            linear.m_stream.wait();
            */
        //} else {
            linear.weight = weight; // dnnl::ocl_interop::make_memory(raw_wei_md, linear.m_engine, dnnl::ocl_interop::memory_kind::usm, static_cast<void*>(data));
        //}

        if (scale) {
            // https://uxlfoundation.github.io/oneDNN/page_weights_decompression_matmul_cpp.html
            // Quantization Group size for scales. Must be divisible by 32.
            auto wei_scale_md = dnnl::memory::desc(dnnl::memory::dims({mm->m_K_groups, mm->m_N}),
                                             dnnl::memory::data_type::f16,
                                             dnnl::memory::format_tag::ab);
            linear.scale = scale; //dnnl::ocl_interop::make_memory(wei_scale_md, linear.m_engine, dnnl::ocl_interop::memory_kind::usm, scale);
            if (zp) {
                auto wei_zp_md = dnnl::memory::desc(dnnl::memory::dims({mm->m_K_groups, mm->m_N}),
                                              mm->m_w_type,
                                              dnnl::memory::format_tag::ab);
                linear.zp = zp; // dnnl::ocl_interop::make_memory(wei_zp_md, linear.m_engine, dnnl::ocl_interop::memory_kind::usm, zp);
            }
        }
        return linear;
    }

    void forward(dnnl::stream& stream, int m, dnnl::memory src_mem, dnnl::memory dst_mem, dnnl::memory bin_mem/*void* a, void* c, void* bin_input*/) {
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
        // auto src_mem = dnnl::ocl_interop::make_memory(rt_src_md, m_engine, dnnl::ocl_interop::memory_kind::usm, (void *)(a));
        //auto weights_mem = dnnl::ocl_interop::make_memory(m_weights_md, m_engine, ocl_interop::memory_kind::usm, (void *)(w));
        // auto dst_mem = dnnl::ocl_interop::make_memory(rt_dst_md, m_engine, dnnl::ocl_interop::memory_kind::usm, (void *)(c));
        //auto bias_mem = memory(bias_md, m_engine);

        std::unordered_map<int, dnnl::memory> args;
        args.insert({DNNL_ARG_SRC, src_mem});
        args.insert({DNNL_ARG_WEIGHTS, weight});
        //args.insert({DNNL_ARG_BIAS, bias_mem});
        args.insert({DNNL_ARG_DST, dst_mem});

        if (scale) {
            args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale});
        }
        if (zp) {
            args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp});
        }
        if (bin_mem) {
            //auto bin_mem = dnnl::ocl_interop::make_memory(rt_bin_md, m_engine, dnnl::ocl_interop::memory_kind::usm, (void *)(bin_input));
            args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(bin_post_id) | DNNL_ARG_SRC_1, bin_mem});
        }
        m_prim.execute(stream, args);
    }
};

class MoeExpertOptGather : public KernelGenerator {
public:
    MoeExpertOptGather() : KernelGenerator("moe_expert_opt", "gather") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<moe_expert>();
        jit.make("GATHER_ENABLE", 1);
        jit.make("HIDDEN_SIZE", desc->get_hidden_size());
        jit.make("TYPE", params.get_input_layout(0).data_type == ov::element::f16 ? "half" : "float");
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;

            // auto max_wgs = params.get_program().get_engine().get_device_info().max_work_group_size;

            wgs.global[0] = wgs.local[0];
            wgs.global[1] = wgs.local[1];
        }};
    }
};

class MoeExpertOptScatter : public KernelGenerator {
public:
    MoeExpertOptScatter() : KernelGenerator("moe_expert_opt", "index_add") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<moe_expert>();
        jit.make("SCATTER_ENABLE", 1);
        jit.make("HIDDEN_SIZE", desc->get_hidden_size());
        jit.make("TYPE", params.get_input_layout(0).data_type == ov::element::f16 ? "half" : "float");
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        }};
    }
};

dnnl::memory convert2dnnl(const memory::ptr& ptr, const std::vector<int64_t>& dim, dnnl::memory::format_tag tag, int offset = 0) {
    return ptr->get_onednn_memory(dnnl::memory::desc(dnnl::memory::dims(dim), convert_data_type(ptr->get_layout().data_type), tag), offset);
}

class MoeExpertOptImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::MoeExpertOptImpl)
    Stage::Ptr gather = make_stage<MoeExpertOptGather>();
    Stage::Ptr scatter = make_stage<MoeExpertOptScatter>();

    struct dnnl_weights {
        dnnl::memory weight;
        dnnl::memory scale;
        dnnl::memory zp;
        int ic, oc, ic_group_size;
    };
    dnnl_weights _dnnl_weights[3];

    MoeExpertOptImpl() : PrimitiveImplOCL(MoeExpertOpt::get_type_info_static()) {}
    MoeExpertOptImpl(const program_node& node, const RuntimeParams& params) : MoeExpertOptImpl() {
        node.get_program().get_engine().create_onednn_engine(node.get_program().get_config());
        const auto& moe = node.as<moe_expert>();
        const moe_expert::mlp_params& mlp_params = moe.get_mlp_params();
        for (int i = 0; i < 3; i++) {
            auto layout = mlp_params.param[i].weight->get_layout();
            const auto& shape = layout.get_shape();
            _dnnl_weights[i].oc = static_cast<int>(shape[0]);
            if (shape.size() == 3) {
                _dnnl_weights[i].ic = static_cast<int>(shape[1] * shape[2]);
                _dnnl_weights[i].ic_group_size = static_cast<int>(shape[2]);
                if (mlp_params.param[i].zp) {
                    auto zp_shape = mlp_params.param[i].zp->get_layout().get_shape();
                    OPENVINO_ASSERT(zp_shape.size() == 3);
                    OPENVINO_ASSERT(zp_shape[0] == shape[1] && zp_shape[1] == shape[0] && zp_shape[2] == 1);
                }
                if (mlp_params.param[i].scale) {
                    auto scale_shape = mlp_params.param[i].scale->get_layout().get_shape();
                    OPENVINO_ASSERT(scale_shape.size() == 3);
                    OPENVINO_ASSERT(scale_shape[0] == shape[1] && scale_shape[1] == shape[0] && scale_shape[2] == 1);
                }
            } else {
                OPENVINO_ASSERT(shape.size() == 2);
                _dnnl_weights[i].ic = static_cast<int>(shape[1]);
                _dnnl_weights[i].ic_group_size = static_cast<int>(shape[1]);
                if (mlp_params.param[i].zp) {
                    auto zp_shape = mlp_params.param[i].zp->get_layout().get_shape();
                    OPENVINO_ASSERT(zp_shape.size() == 2);
                    OPENVINO_ASSERT(zp_shape[0] == shape[0]);
                }
                if (mlp_params.param[i].scale) {
                    auto scale_shape = mlp_params.param[i].scale->get_layout().get_shape();
                    OPENVINO_ASSERT(scale_shape.size() == 2);
                    OPENVINO_ASSERT(scale_shape[0] == shape[0]);
                }
            }
            if (mlp_params.param[i].scale) {
                _dnnl_weights[i].scale = convert2dnnl(mlp_params.param[i].scale,
                    {_dnnl_weights[i].ic / _dnnl_weights[i].ic_group_size, _dnnl_weights[i].oc},
                    dnnl::memory::format_tag::ab);
            }
            if (mlp_params.param[i].zp) {
                _dnnl_weights[i].zp = convert2dnnl(mlp_params.param[i].zp,
                    {_dnnl_weights[i].ic / _dnnl_weights[i].ic_group_size, _dnnl_weights[i].oc},
                    dnnl::memory::format_tag::ab);
            }
            if (mlp_params.param[i].weight) {
                _dnnl_weights[i].weight = convert2dnnl(mlp_params.param[i].weight, 
                    {_dnnl_weights[i].ic, _dnnl_weights[i].oc},
                    dnnl::memory::format_tag::ba);
            }
        }

        add_stage(gather, params);
        add_stage(scatter, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        auto moe = make_deep_copy<MoeExpertOptImpl>(this);
        moe->_dnnl_weights[0] = _dnnl_weights[0];
        moe->_dnnl_weights[1] = _dnnl_weights[1];
        moe->_dnnl_weights[2] = _dnnl_weights[2];
        return moe;
    }

    bool is_cpu() const override {
        return false;
    }

    // [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
    //     auto desc = params.typed_desc<moe_expert>();
    //     const auto& shape = params.output_layouts[0].get_shape();
    //     auto buf = BufferDescriptor{shape[0] * shape[1], ov::element::f32};
    //     return {buf, buf};
    // }

    cldnn::event::ptr execute_stage(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance, Stage& stage, std::vector<memory::ptr> inputs,
        std::vector<memory::ptr> outputs, const std::vector<size_t>& global, const std::vector<size_t>& local) const {
        cldnn::stream& stream = instance.get_network().get_stream();
        bool needs_completion_event = instance.needs_completion_event();
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

        // scalars_desc scalars_desc;
        // for (auto i : scalars) {
        //     scalar_desc desc;
        //     desc.t = scalar_desc::Types::INT32;
        //     desc.v.s32 = i;
        //     scalars_desc.push_back(desc);
        // }
        // args.scalars = &scalars_desc;
        stream.set_arguments(*stage.kernel, desc, args);
        desc.workGroups.global = global;
        desc.workGroups.local = local;

        return stream.enqueue_kernel(*stage.kernel, desc, {}, events, needs_completion_event);
    }

    cldnn::event::ptr execute(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& ins) override {
        auto& instance = reinterpret_cast<typed_primitive_inst<moe_expert>&>(ins);
        auto expert_no = instance.get_config().expert_no;
        int max_topk = static_cast<int>(instance.get_config().topk);
        auto& cur_net = instance.get_network();
        auto& stream = cur_net.get_stream();
        auto moe = instance.get_typed_desc<moe_expert>();

        instance.update_output_layout();
        instance.update_output_memory(expert_no == 0);

        if (!cur_net.has_scratch<expert_mask_scratch>(expert_mask_scratch_key)) {
            cur_net.set_scratch<expert_mask_scratch>(expert_mask_scratch_key, {});
        }
        expert_mask_scratch& expert_mask = cur_net.get_scratch<expert_mask_scratch>(expert_mask_scratch_key);
        if (expert_mask.execed_count++ % instance.get_config().expert_num == 0) {
            // Wait for moe_expert statement event only, and pass all other events to sub-network directly
            // auto dep_event = instance.pred_inst()->get_impl_params()->out_event;
            // if (dep_event)
            //     dep_event->wait();
            // TODO: wait dep only
            for (auto&& event: events)
                event->wait();
            auto dep = instance.dependencies()[1];
            auto layout = dep.first->get_impl_params()->get_output_layout(dep.second);
            moe_expert_inst::get_expert_mask_from_memory(instance.pred_memory_ptr(), layout, stream, expert_mask);
            {
                const auto& shape = layout.get_shape();
                int max_expert_num = static_cast<int>(shape[0]), max_topk = static_cast<int>(shape[1]), max_tokens = static_cast<int>(shape[2]);
                int count = 0;
                for (int no = 0; no < max_expert_num; no++) {
                    count += static_cast<int>(expert_mask.batch[no].size());
                }
                OPENVINO_ASSERT(count == max_topk * max_tokens, "With max_expert_num=", max_expert_num, ",max_topk=", max_topk, ",exec count=",
                    expert_mask.execed_count, ",max_tokens=", max_tokens, " should have ", max_topk * max_tokens, " tokens, but current is ",
                    count, ". layout=", layout);
            }
        }

        OPENVINO_ASSERT(expert_no < expert_mask.pred_flag.size());
        auto can_skip_subgraph = !expert_mask.pred_flag[expert_no];
        if (can_skip_subgraph) {
            if (instance.needs_completion_event())
                return stream.enqueue_marker({});

            return {};
        }

        auto key = expert_mask_mem_scratch_key + std::to_string(expert_no);
        if (!cur_net.has_scratch<expert_mask_mem_scratch>(key)) {
            cur_net.set_scratch<expert_mask_mem_scratch>(key, {});
        }
        auto& expert_mask_mem = cur_net.get_scratch<expert_mask_mem_scratch>(key);
        instance.copy_expert_mask_to_gpu(stream, expert_mask, expert_no, expert_mask_mem);

        auto dnn_stream = stream.get_onednn_stream();
        auto input_info = [&] (int idx) {
            auto mem = instance.input_memory_ptr(idx);
            auto dep = instance.dependencies()[idx];
            auto layout = dep.first->get_impl_params()->get_output_layout(dep.second);
            return std::make_tuple(mem, layout);
        };
        auto final_hidden_states_mem_ptr = instance.output_memory_ptr(0);
        auto final_hidden_states_layout = instance.get_output_layout(0);
        auto [expert_mask_mem_ptr, expert_mask_layout] = input_info(1);
        auto [hidden_states_mem_ptr, hidden_states_layout] = input_info(2);
        auto [routing_mem_ptr, routing_layout] = input_info(3);

        const moe_expert::mlp_params& mlp_params = moe->_mlp_params;
        auto batch = static_cast<int>(hidden_states_layout.get_shape()[1]);
        auto n_token = static_cast<int>(expert_mask.batch[expert_no].size());
        auto hidden_size = static_cast<int>(hidden_states_layout.get_shape()[2]);
        auto intermediate_size = static_cast<int>(mlp_params.param[0].weight->get_layout().get_shape()[0]);
        if (!cur_net.has_scratch<expert_mask_tmp_scratch>(expert_mask_tmp_scratch_key)) {
            cur_net.set_scratch<expert_mask_tmp_scratch>(expert_mask_tmp_scratch_key, {});
        }
        expert_mask_tmp_scratch& scratch = cur_net.get_scratch<expert_mask_tmp_scratch>(expert_mask_tmp_scratch_key);
        instance.get_tmp_memory(hidden_states_layout.data_type, n_token, hidden_size, intermediate_size, max_topk, scratch);
        memory::ptr x;
        if (batch == 1) {
            x = hidden_states_mem_ptr;
        } else {
            x = scratch.x;
            // gather
            execute_stage(events, instance, *gather, {hidden_states_mem_ptr, routing_mem_ptr, expert_mask_mem.batch, expert_mask_mem.topk},
                {x, scratch.routing_weights}, {static_cast<size_t>(n_token), static_cast<size_t>(hidden_size)}, {1, 1});
        }
        {
            // up
            auto up_weight_layout = mlp_params.param[1].weight->get_layout();

            auto linear = onednn_linear::create(dnn_stream.get_engine(),
                convert_data_type(hidden_states_layout.data_type),
                convert_data_type(up_weight_layout.data_type),
                n_token,
                _dnnl_weights[1].ic,
                _dnnl_weights[1].oc,
                _dnnl_weights[1].ic_group_size,
                onednn_matmul::type::none,
                _dnnl_weights[1].weight,
                _dnnl_weights[1].scale,
                _dnnl_weights[1].zp);
            linear.forward(dnn_stream, n_token,
                convert2dnnl(x, {static_cast<int>(n_token), _dnnl_weights[1].ic}, dnnl::memory::format_tag::ab),
                convert2dnnl(scratch.up, {static_cast<int>(n_token), intermediate_size}, dnnl::memory::format_tag::ab),
                dnnl::memory());
        }
        {
            // gate
            auto gate_weight_layout = mlp_params.param[0].weight->get_layout();
            auto linear = onednn_linear::create(dnn_stream.get_engine(),
                convert_data_type(hidden_states_layout.data_type),
                convert_data_type(gate_weight_layout.data_type),
                n_token,
                _dnnl_weights[0].ic,
                _dnnl_weights[0].oc,
                _dnnl_weights[0].ic_group_size,
                onednn_matmul::type::with_silu_bin_mul,
                _dnnl_weights[0].weight,
                _dnnl_weights[0].scale,
                _dnnl_weights[0].zp);
            linear.forward(dnn_stream, n_token,
                convert2dnnl(x, {static_cast<int>(n_token), _dnnl_weights[0].ic}, dnnl::memory::format_tag::ab),
                convert2dnnl(scratch.gate, {static_cast<int>(n_token), intermediate_size}, dnnl::memory::format_tag::ab),
                convert2dnnl(scratch.up, {static_cast<int>(n_token), intermediate_size}, dnnl::memory::format_tag::ab));
        }
        // down
        auto down_weight_layout = mlp_params.param[2].weight->get_layout();
        auto routing_num = routing_layout.get_dim(0);
        cldnn::event::ptr result_event;
        if (batch == 1) {
            auto linear = onednn_linear::create(dnn_stream.get_engine(),
                convert_data_type(hidden_states_layout.data_type),
                convert_data_type(down_weight_layout.data_type),
                batch,
                _dnnl_weights[2].ic,
                _dnnl_weights[2].oc,
                _dnnl_weights[2].ic_group_size,
                onednn_matmul::type::with_bin_mul_per_row_sum,
                _dnnl_weights[2].weight,
                _dnnl_weights[2].scale,
                _dnnl_weights[2].zp);
            linear.forward(dnn_stream, batch,
                convert2dnnl(scratch.gate, {static_cast<int>(batch), intermediate_size}, dnnl::memory::format_tag::ab),
                convert2dnnl(final_hidden_states_mem_ptr, {static_cast<int>(batch), hidden_size}, dnnl::memory::format_tag::ab),
                convert2dnnl(routing_mem_ptr, {routing_num}, dnnl::memory::format_tag::a,
                    expert_mask.topk[expert_no][0] * static_cast<int>(ov::element::Type(routing_layout.data_type).size())));
            if (instance.needs_completion_event())
                result_event = stream.enqueue_marker({});
        } else {
            auto linear = onednn_linear::create(dnn_stream.get_engine(),
                convert_data_type(hidden_states_layout.data_type),
                convert_data_type(down_weight_layout.data_type),
                n_token,
                _dnnl_weights[2].ic,
                _dnnl_weights[2].oc,
                _dnnl_weights[2].ic_group_size,
                onednn_matmul::type::with_bin_mul_per_row,
                _dnnl_weights[2].weight,
                _dnnl_weights[2].scale,
                _dnnl_weights[2].zp);
            linear.forward(dnn_stream, n_token,
                convert2dnnl(scratch.gate, {static_cast<int>(n_token), intermediate_size}, dnnl::memory::format_tag::ab),
                convert2dnnl(scratch.y, {static_cast<int>(n_token), hidden_size}, dnnl::memory::format_tag::ab),
                convert2dnnl(scratch.routing_weights, {n_token * max_topk}, dnnl::memory::format_tag::a));
            // index add
            result_event = execute_stage(events, instance, *scatter, {scratch.y, expert_mask_mem.batch},
                {final_hidden_states_mem_ptr}, {static_cast<size_t>(n_token), static_cast<size_t>(hidden_size)}, {1, 1});
        }

        return result_event;
    }
};

}  // namespace

std::unique_ptr<primitive_impl> MoeExpertOpt::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<moe_expert>());
    return std::make_unique<MoeExpertOptImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::MoeExpertOptImpl)
