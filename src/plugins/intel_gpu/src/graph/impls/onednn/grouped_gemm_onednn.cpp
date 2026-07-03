// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grouped_gemm_onednn.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "primitive_onednn_base.h"
#include "utils.hpp"

#include <oneapi/dnnl/dnnl.hpp>

#include <memory>

namespace cldnn {
namespace onednn {

// Executes the GroupedMatMul-17 2D x 3D case via dnnl::matmul with grouped
// memory encoding. Layout:
//   mat_a   : [total_tokens, K]     -> dnnl grouped src, variable dim = 0
//   mat_b   : [G, N, K]             -> dnnl weights, format_tag::acb  (logical [G, K, N])
//   offsets : [G] s32               -> attached to src/dst as buffer 1
//   out     : [total_tokens, N]     -> dnnl grouped dst, variable dim = 0
struct grouped_gemm_onednn : typed_primitive_onednn_impl<grouped_matmul> {
    using parent = typed_primitive_onednn_impl<grouped_matmul>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::grouped_gemm_onednn)

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<grouped_gemm_onednn>(*this);
    }

    std::unordered_map<int, dnnl::memory> get_arguments(grouped_matmul_inst& instance) const override {
        std::unordered_map<int, dnnl::memory> args;

        auto& input = instance.input_memory(grouped_matmul::GroupedMatmulInputIdx::INPUT);
        auto& weights = instance.input_memory(grouped_matmul::GroupedMatmulInputIdx::WEIGHT);
        auto& offsets = instance.input_memory(grouped_matmul::GroupedMatmulInputIdx::OFFSETS);
        auto& output = instance.output_memory(0);

        args.insert({DNNL_ARG_SRC, input.get_onednn_grouped_memory(_pd.src_desc(0), offsets)});
        args.insert({DNNL_ARG_WEIGHTS, weights.get_onednn_memory(_pd.weights_desc(0), 0)});
        args.insert({DNNL_ARG_DST, output.get_onednn_grouped_memory(_pd.dst_desc(0), offsets)});

        const auto& prim = instance.get_impl_params()->typed_desc<grouped_matmul>();
        if (prim->compressed_weights) {
            // scale/zp are attached as extra dependencies after the 3 regular inputs.
            size_t idx = grouped_matmul::GroupedMatmulInputIdx::OFFSETS + 1;

            OPENVINO_ASSERT(prim->decompression_scale.is_valid(),
                            "[GPU] Compressed grouped_matmul requires a decompression scale");
            auto scale_mem = instance.dep_memory_ptr(idx++);
            const auto& scale_shape = scale_mem->get_layout().get_shape();
            const auto& wei_shape = instance.get_input_layout(grouped_matmul::GroupedMatmulInputIdx::WEIGHT).get_shape();
            OPENVINO_ASSERT(wei_shape.size() == 3, "[GPU] grouped_matmul weights must be rank 3");
            const dnnl::memory::dim G = static_cast<dnnl::memory::dim>(wei_shape[0]);
            const dnnl::memory::dim N = static_cast<dnnl::memory::dim>(wei_shape[1]);
            const dnnl::memory::dim K = static_cast<dnnl::memory::dim>(wei_shape[2]);

            dnnl::memory::dims scale_dims;
            dnnl::memory::format_tag scale_fmt;
            if (scale_shape.size() == 2) {
                // Per-OC: scale physical is [G, N]. oneDNN weights are logical [G, K, N], so we bind
                // scale as [G, N] with format ab and mask over dims 0 and 2 of the weights tensor.
                scale_dims = dnnl::memory::dims{G, N};
                scale_fmt = dnnl::memory::format_tag::ab;
            } else {
                // Grouped: scale physical is [G, N, K/gs]; onednn expects logical [G, K/gs, N] fmt abc.
                OPENVINO_ASSERT(scale_shape.size() == 3,
                                "[GPU] Unexpected decompression scale rank ", scale_shape.size());
                const dnnl::memory::dim num_groups = static_cast<dnnl::memory::dim>(scale_shape[2]);
                OPENVINO_ASSERT(num_groups > 0 && K % num_groups == 0,
                                "[GPU] grouped_matmul scale groups (", num_groups,
                                ") must evenly divide K (", K, ")");
                scale_dims = dnnl::memory::dims{G, num_groups, N};
                scale_fmt = dnnl::memory::format_tag::abc;
            }
            dnnl::memory::desc scale_md(scale_dims, convert_data_type(scale_mem->get_layout().data_type), scale_fmt);
            args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale_mem->get_onednn_memory(scale_md, 0)});

            if (prim->decompression_zero_point.is_valid() && !prim->decompression_zero_point_scalar.has_value()) {
                auto zp_mem = instance.dep_memory_ptr(idx++);
                const auto& zp_shape = zp_mem->get_layout().get_shape();
                OPENVINO_ASSERT(zp_shape == scale_shape,
                                "[GPU] grouped_matmul zero-point shape ", zp_shape,
                                " must match scale shape ", scale_shape);
                dnnl::memory::desc zp_md(scale_dims, convert_data_type(zp_mem->get_layout().data_type), scale_fmt);
                args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp_mem->get_onednn_memory(zp_md, 0)});
            }
        }

        return args;
    }

    static std::shared_ptr<dnnl::matmul::primitive_desc> get_primitive_descriptor(
        const kernel_impl_params& impl_params,
        const dnnl::primitive_attr& attr = dnnl::primitive_attr()) {
        auto& engine = impl_params.prog->get_engine();

        const auto& in_layout = impl_params.get_input_layout(grouped_matmul::GroupedMatmulInputIdx::INPUT);
        const auto& wei_layout = impl_params.get_input_layout(grouped_matmul::GroupedMatmulInputIdx::WEIGHT);
        const auto& out_layout = impl_params.get_output_layout();

        const auto& in_shape = in_layout.get_shape();
        const auto& wei_shape = wei_layout.get_shape();

        OPENVINO_ASSERT(in_shape.size() == 2,
                        "grouped_gemm_onednn expects mat_a rank 2, got ", in_shape.size());
        OPENVINO_ASSERT(wei_shape.size() == 3,
                        "grouped_gemm_onednn expects mat_b rank 3, got ", wei_shape.size());

        const dnnl::memory::dim total_tokens = static_cast<dnnl::memory::dim>(in_shape[0]);
        const dnnl::memory::dim K = static_cast<dnnl::memory::dim>(in_shape[1]);
        const dnnl::memory::dim num_experts = static_cast<dnnl::memory::dim>(wei_shape[0]);
        const dnnl::memory::dim N = static_cast<dnnl::memory::dim>(wei_shape[1]);
        OPENVINO_ASSERT(static_cast<dnnl::memory::dim>(wei_shape[2]) == K,
                        "grouped_gemm_onednn mat_b last dim (", wei_shape[2],
                        ") must match mat_a last dim (", K, ")");

        // dnnl grouped matmul: src=[T,K] with variable dim 0, weights logical=[G,K,N] stored acb (=> physical [G,N,K]).
        auto src_md = dnnl::memory::desc::grouped(dnnl::memory::dims{total_tokens, K},
                                                  convert_data_type(in_layout.data_type),
                                                  /*variable_dim_idx=*/0,
                                                  num_experts);
        auto dst_md = dnnl::memory::desc::grouped(dnnl::memory::dims{total_tokens, N},
                                                  convert_data_type(out_layout.data_type),
                                                  /*variable_dim_idx=*/0,
                                                  num_experts);
        auto wei_md = dnnl::memory::desc(dnnl::memory::dims{num_experts, K, N},
                                         convert_data_type(wei_layout.data_type),
                                         dnnl::memory::format_tag::acb);

        return std::make_shared<dnnl::matmul::primitive_desc>(engine.get_onednn_engine(),
                                                              src_md,
                                                              wei_md,
                                                              dst_md,
                                                              attr);
    }

public:
    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
    }

    static std::unique_ptr<primitive_impl> create(const grouped_matmul_node& arg,
                                                  const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        auto attr = impl_params.attrs_onednn;

        const auto& prim = impl_params.typed_desc<grouped_matmul>();
        if (prim->compressed_weights) {
            // scale is the first extra dependency (after src/wei/offsets); zp (if any) is next.
            size_t idx = grouped_matmul::GroupedMatmulInputIdx::OFFSETS + 1;

            const auto& scale_layout = arg.get_dependency(idx++).get_output_layout();
            const auto scale_dt = convert_data_type(scale_layout.data_type);
            const auto& scale_shape = scale_layout.get_shape();
            const auto& wei_shape = impl_params.get_input_layout(grouped_matmul::GroupedMatmulInputIdx::WEIGHT).get_shape();
            OPENVINO_ASSERT(wei_shape.size() == 3, "[GPU] grouped_matmul weights must be rank 3");
            const int64_t K = static_cast<int64_t>(wei_shape[2]);

            // Mask 5 = bit0(G) | bit2(N) for per-OC. Mask 7 adds bit1(K) for grouped-K.
            if (scale_shape.size() == 2) {
                attr->set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 2), {}, scale_dt);
            } else {
                OPENVINO_ASSERT(scale_shape.size() == 3,
                                "[GPU] Unexpected decompression scale rank ", scale_shape.size());
                const int64_t num_groups = static_cast<int64_t>(scale_shape[2]);
                OPENVINO_ASSERT(num_groups > 0 && K % num_groups == 0,
                                "[GPU] grouped_matmul scale groups (", num_groups,
                                ") must evenly divide K (", K, ")");
                const dnnl::memory::dim group_size = K / num_groups;
                attr->set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1) | (1 << 2), {group_size, 1}, scale_dt);
            }

            if (prim->decompression_zero_point.is_valid() && !prim->decompression_zero_point_scalar.has_value()) {
                const auto& zp_layout = arg.get_dependency(idx++).get_output_layout();
                const auto zp_dt = convert_data_type(zp_layout.data_type);
                if (scale_shape.size() == 2) {
                    attr->set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 2), {}, zp_dt);
                } else {
                    const int64_t num_groups = static_cast<int64_t>(scale_shape[2]);
                    const dnnl::memory::dim group_size = K / num_groups;
                    attr->set_zero_points(DNNL_ARG_WEIGHTS,
                                          (1 << 0) | (1 << 1) | (1 << 2),
                                          {group_size, 1},
                                          zp_dt);
                }
            }
        }

        auto prim_desc = get_primitive_descriptor(impl_params, *attr);
        return std::make_unique<grouped_gemm_onednn>(engine, config, attr, *prim_desc);
    }
};

std::unique_ptr<primitive_impl> GroupedMatmulImplementationManager::create_impl(const program_node& node,
                                                                                const kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<grouped_matmul>());
    return onednn::grouped_gemm_onednn::create(static_cast<const grouped_matmul_node&>(node), params);
}

}  // namespace onednn
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::grouped_matmul)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::grouped_gemm_onednn)
