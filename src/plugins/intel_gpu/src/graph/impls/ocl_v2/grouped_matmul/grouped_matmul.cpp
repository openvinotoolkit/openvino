// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../primitive_ocl_base.hpp"
#include "grouped_matmul_impl.hpp"
#include "grouped_matmul_inst.h"

#ifdef ENABLE_ONEDNN_FOR_GPU
#    include <memory>
#    include <oneapi/dnnl/dnnl.hpp>
#    include <unordered_map>

#    include "intel_gpu/runtime/lru_cache.hpp"
#endif

namespace ov::intel_gpu::ocl {
namespace {
#ifdef ENABLE_ONEDNN_FOR_GPU

// Shared with gather_matmul; duplicated here to avoid a cross-subdir dependency.
inline dnnl::memory::data_type convert_data_type(cldnn::data_types dt) {
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
        throw std::invalid_argument("[GPU] grouped_matmul: unsupported cldnn->onednn type conversion");
    }
}
#endif

class GroupedMatmulOCLImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::GroupedMatmulOCLImpl)

    explicit GroupedMatmulOCLImpl() : PrimitiveImplOCL(GroupedMatmulImpl::get_type_info_static()) {}
    explicit GroupedMatmulOCLImpl(const RuntimeParams& /*impl_param*/) : GroupedMatmulOCLImpl() {}

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<GroupedMatmulOCLImpl>(this);
    }

    // No OCL kernel stages — execution is purely via oneDNN. Skip kernel-cache
    // lookups that PrimitiveImplOCL::init_kernels() would otherwise trigger.
    void init_kernels(const cldnn::kernels_cache& /*kernels_cache*/, const RuntimeParams& /*params*/) override {}

    std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& /*params*/) const override {
        // No internal buffers needed: tokens are already partitioned by expert (no sort/scatter).
        return {};
    }

    [[nodiscard]] event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
#ifdef ENABLE_ONEDNN_FOR_GPU
        // Wait for all input events before dispatching to oneDNN.
        for (const auto& e : events) {
            if (e)
                e->wait();
        }

        const auto& impl_params = *instance.get_impl_params();
        auto desc = impl_params.typed_desc<grouped_matmul>();

        const bool is_3d = !desc->has_offsets;

        if (is_3d) {
            return execute_batched_3d(instance);
        } else {
            return execute_grouped_2d(instance);
        }
#else
        OPENVINO_THROW("grouped_matmul is only supported on systolic platforms with oneDNN.");
        return nullptr;
#endif
    }

private:
#ifdef ENABLE_ONEDNN_FOR_GPU
    struct DnnlKernel {
        dnnl::matmul::primitive_desc pd;
        dnnl::matmul prim;
    };

    // Key: {G, M, N, K} packed as a 64-bit hash for the 3D×3D case, or total_tokens for 2D×3D.
    cldnn::LruCache<int64_t, std::shared_ptr<DnnlKernel>> _kernel_cache{32};

    // ---- 3D×3D: batched dnnl::matmul A:[G,M,K] × B:[G,N,K] → out:[G,M,N] ----
    event::ptr execute_batched_3d(primitive_inst& instance) {
        const auto& impl_params = *instance.get_impl_params();
        const auto& input_layout = impl_params.input_layouts[grouped_matmul::InputIdx::INPUT];
        const auto& weight_layout = impl_params.input_layouts[grouped_matmul::InputIdx::WEIGHT];
        const auto& output_layout = impl_params.output_layouts[0];

        const auto& a_shape = input_layout.get_shape();  // [..., G, M, K]
        const auto& b_shape = weight_layout.get_shape();  // [..., G, N, K]

        OPENVINO_ASSERT(a_shape.size() >= 3 && b_shape.size() >= 2,
                        "[GPU] grouped_matmul: 3D×3D case requires A rank >= 3, B rank >= 2");

        const dnnl::memory::dim G = static_cast<dnnl::memory::dim>(a_shape[a_shape.size() - 3]);
        const dnnl::memory::dim M = static_cast<dnnl::memory::dim>(a_shape[a_shape.size() - 2]);
        const dnnl::memory::dim K = static_cast<dnnl::memory::dim>(a_shape[a_shape.size() - 1]);
        const dnnl::memory::dim N = static_cast<dnnl::memory::dim>(b_shape[b_shape.size() - 2]);

        const int64_t cache_key = static_cast<int64_t>(G) * 1000003LL +
                                  static_cast<int64_t>(M) * 100003LL +
                                  static_cast<int64_t>(N) * 1009LL +
                                  static_cast<int64_t>(K);

        if (!_kernel_cache.has(cache_key)) {
            auto& engine = instance.get_network().get_engine();
            auto& dnnl_engine = engine.get_onednn_engine();

            auto a_dt = convert_data_type(input_layout.data_type);
            auto w_dt = convert_data_type(weight_layout.data_type);
            auto out_dt = convert_data_type(output_layout.data_type);

            dnnl::primitive_attr attr;
            attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);

            // A: [G, M, K] — standard row-major batched
            auto src_md = dnnl::memory::desc({G, M, K}, a_dt, dnnl::memory::format_tag::abc);
            // B stored as [G, N, K] (pre-transposed); logical for matmul is [G, K, N] → acb
            auto w_md = dnnl::memory::desc({G, K, N}, w_dt, dnnl::memory::format_tag::acb);
            // Out: [G, M, N]
            auto dst_md = dnnl::memory::desc({G, M, N}, out_dt, dnnl::memory::format_tag::abc);

            auto gk = std::make_shared<DnnlKernel>();
            gk->pd = dnnl::matmul::primitive_desc(dnnl_engine, src_md, w_md, dst_md, attr);
            gk->prim = dnnl::matmul(gk->pd);
            _kernel_cache.add(cache_key, gk);
        }

        auto& gk = *_kernel_cache.get(cache_key);
        auto& net = instance.get_network();
        auto& stream = net.get_stream();
        auto& dnn_stream = stream.get_onednn_stream();

        auto& input_mem = *instance.input_memory_ptr(grouped_matmul::InputIdx::INPUT);
        auto& weight_mem = *instance.input_memory_ptr(grouped_matmul::InputIdx::WEIGHT);
        auto& output_mem = *instance.output_memory_ptr(0);

        std::unordered_map<int, dnnl::memory> args{
            {DNNL_ARG_SRC, input_mem.get_onednn_memory(gk.pd.src_desc())},
            {DNNL_ARG_WEIGHTS, weight_mem.get_onednn_memory(gk.pd.weights_desc())},
            {DNNL_ARG_DST, output_mem.get_onednn_memory(gk.pd.dst_desc())},
        };

        gk.prim.execute(dnn_stream, args);
        dnn_stream.wait();

        return stream.create_user_event(true);
    }

    // ---- 2D×3D: grouped dnnl::matmul A:[T,K] × B:[G,N,K] → out:[T,N] ----
    // Offsets input[2]:[G] is an i32 array of cumulative end-offsets consumed by
    // dnnl::memory::desc::grouped(), identical to how gather_matmul uses EXPERT_OFFSETS.
    event::ptr execute_grouped_2d(primitive_inst& instance) {
        const auto& impl_params = *instance.get_impl_params();
        const auto& input_layout = impl_params.input_layouts[grouped_matmul::InputIdx::INPUT];
        const auto& weight_layout = impl_params.input_layouts[grouped_matmul::InputIdx::WEIGHT];
        const auto& output_layout = impl_params.output_layouts[0];

        const auto& a_shape = input_layout.get_shape();  // [..., T, K]
        const auto& b_shape = weight_layout.get_shape();  // [..., G, N, K]

        OPENVINO_ASSERT(a_shape.size() >= 2 && b_shape.size() >= 3,
                        "[GPU] grouped_matmul: 2D×3D case requires A rank >= 2, B rank >= 3");

        const dnnl::memory::dim T = static_cast<dnnl::memory::dim>(a_shape[a_shape.size() - 2]);
        const dnnl::memory::dim K = static_cast<dnnl::memory::dim>(a_shape[a_shape.size() - 1]);
        const dnnl::memory::dim G = static_cast<dnnl::memory::dim>(b_shape[b_shape.size() - 3]);
        const dnnl::memory::dim N = static_cast<dnnl::memory::dim>(b_shape[b_shape.size() - 2]);

        // Cache key encodes all dimensions that affect the primitive_desc.
        const int64_t cache_key = T * 1000003LL + K * 100003LL + G * 1009LL + N;

        if (!_kernel_cache.has(cache_key)) {
            auto& engine = instance.get_network().get_engine();
            auto& dnnl_engine = engine.get_onednn_engine();

            auto a_dt = convert_data_type(input_layout.data_type);
            auto w_dt = convert_data_type(weight_layout.data_type);
            auto out_dt = convert_data_type(output_layout.data_type);

            dnnl::primitive_attr attr;
            attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);

            // Grouped src/dst: T rows split by G groups via cumulative i32 end-offsets.
            auto src_md = dnnl::memory::desc::grouped(dnnl::memory::dims{T, K}, a_dt, 0, G, dnnl::memory::data_type::s32);
            auto dst_md = dnnl::memory::desc::grouped(dnnl::memory::dims{T, N}, out_dt, 0, G, dnnl::memory::data_type::s32);
            // B stored as [G, N, K] (pre-transposed); logical [G, K, N] → acb
            auto w_md = dnnl::memory::desc(dnnl::memory::dims{G, K, N}, w_dt, dnnl::memory::format_tag::acb);

            auto gk = std::make_shared<DnnlKernel>();
            gk->pd = dnnl::matmul::primitive_desc(dnnl_engine, src_md, w_md, dst_md, attr);
            gk->prim = dnnl::matmul(gk->pd);
            _kernel_cache.add(cache_key, gk);
        }

        auto& gk = *_kernel_cache.get(cache_key);
        auto& net = instance.get_network();
        auto& stream = net.get_stream();
        auto& dnn_stream = stream.get_onednn_stream();

        auto& input_mem = *instance.input_memory_ptr(grouped_matmul::InputIdx::INPUT);
        auto& weight_mem = *instance.input_memory_ptr(grouped_matmul::InputIdx::WEIGHT);
        auto& offsets_mem = *instance.input_memory_ptr(grouped_matmul::InputIdx::OFFSETS);
        auto& output_mem = *instance.output_memory_ptr(0);

        auto src_dnn = input_mem.get_onednn_grouped_memory(gk.pd.src_desc(), offsets_mem);
        auto dst_dnn = output_mem.get_onednn_grouped_memory(gk.pd.dst_desc(), offsets_mem);
        auto w_dnn = weight_mem.get_onednn_memory(gk.pd.weights_desc());

        std::unordered_map<int, dnnl::memory> args{
            {DNNL_ARG_SRC, src_dnn},
            {DNNL_ARG_WEIGHTS, w_dnn},
            {DNNL_ARG_DST, dst_dnn},
        };

        gk.prim.execute(dnn_stream, args);
        dnn_stream.wait();

        return stream.create_user_event(true);
    }
#endif
};
}  // namespace

std::unique_ptr<primitive_impl> GroupedMatmulImpl::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<grouped_matmul>());
    return std::make_unique<GroupedMatmulOCLImpl>(params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::grouped_matmul)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::GroupedMatmulOCLImpl)
