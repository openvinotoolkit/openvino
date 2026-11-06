// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grouped_gemm_onednn.hpp"
#include "grouped_matmul_helper.hpp"
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

    // Weights decompression from the scale/zp layouts: scale [G, N] (per output channel) or [G, K / group_size, N]
    static grouped_matmul_quant get_quant(const kernel_impl_params& impl_params) {
        grouped_matmul_quant q;
        const auto& prim = impl_params.typed_desc<grouped_matmul>();
        if (!prim->compressed_weights) {
            return q;
        }
        OPENVINO_ASSERT(prim->decompression_scale.is_valid(), "[GPU] Compressed grouped_matmul requires a decompression scale");
        const auto& wei_shape = impl_params.get_input_layout(grouped_matmul::GroupedMatmulInputIdx::WEIGHT).get_shape();
        OPENVINO_ASSERT(wei_shape.size() == 3, "[GPU] grouped_matmul weights must be rank 3");
        const auto K = static_cast<dnnl::memory::dim>(wei_shape[2]);

        // scale/zp are attached as extra dependencies after the 3 regular inputs.
        size_t idx = grouped_matmul::GroupedMatmulInputIdx::OFFSETS + 1;
        const auto& scale_layout = impl_params.get_input_layout(idx++);
        const auto& scale_shape = scale_layout.get_shape();
        q.scale_dt = convert_data_type(scale_layout.data_type);
        if (scale_shape.size() == 3) {
            const auto num_groups = static_cast<dnnl::memory::dim>(scale_shape[2]);
            OPENVINO_ASSERT(num_groups > 0 && K % num_groups == 0,
                            "[GPU] grouped_matmul scale groups (", num_groups, ") must evenly divide K (", K, ")");
            q.group_size = K / num_groups;
        } else {
            OPENVINO_ASSERT(scale_shape.size() == 2, "[GPU] grouped_matmul scale must be rank 2 or 3, got ", scale_shape.size());
        }

        if (prim->decompression_zero_point.is_valid()) {
            const auto& zp_layout = impl_params.get_input_layout(idx++);
            OPENVINO_ASSERT(zp_layout.count() > 1, "[GPU] grouped_matmul zero-point must be rank > 0");
            OPENVINO_ASSERT(zp_layout.get_shape() == scale_shape,
                            "[GPU] OneDNN requires that grouped_matmul zero-point shape ", zp_layout.get_shape(),
                            " must match scale shape ", scale_shape);
            q.zp_dt = convert_data_type(zp_layout.data_type);
        }
        return q;
    }

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
        if (prim->rows_per_token > 0) {
            const auto rows = instance.get_input_layout(grouped_matmul::GroupedMatmulInputIdx::INPUT).get_shape()[0];
            args.insert({DNNL_ARG_HINT_MAX_GROUP_SIZE, make_grouped_matmul_hint(static_cast<int32_t>(rows / prim->rows_per_token))});
        }
        if (prim->compressed_weights) {
            const auto q = get_quant(*instance.get_impl_params());
            const auto& wei_shape = instance.get_input_layout(grouped_matmul::GroupedMatmulInputIdx::WEIGHT).get_shape();
            const auto G = static_cast<dnnl::memory::dim>(wei_shape[0]);
            const auto N = static_cast<dnnl::memory::dim>(wei_shape[1]);
            const auto K = static_cast<dnnl::memory::dim>(wei_shape[2]);

            size_t idx = grouped_matmul::GroupedMatmulInputIdx::OFFSETS + 1;
            auto scale_mem = instance.dep_memory_ptr(idx++);
            args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale_mem->get_onednn_memory(make_grouped_matmul_quant_md(G, K, N, q, q.scale_dt), 0)});
            if (q.has_zp()) {
                auto zp_mem = instance.dep_memory_ptr(idx++);
                args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp_mem->get_onednn_memory(make_grouped_matmul_quant_md(G, K, N, q, q.zp_dt), 0)});
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
                        "K must match between weight and input: ", wei_shape[2], " - ", K);

        return std::make_shared<dnnl::matmul::primitive_desc>(make_grouped_matmul_pd(engine.get_onednn_engine(),
                                                                                     total_tokens,
                                                                                     K,
                                                                                     N,
                                                                                     num_experts,
                                                                                     convert_data_type(in_layout.data_type),
                                                                                     convert_data_type(wei_layout.data_type),
                                                                                     convert_data_type(out_layout.data_type),
                                                                                     attr));
    }

public:

    static std::unique_ptr<primitive_impl> create(const grouped_matmul_node& /* arg */,
                                                  const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        const auto& config = impl_params.prog->get_config();
        auto attr = impl_params.attrs_onednn;

        const auto& prim = impl_params.typed_desc<grouped_matmul>();
        if (prim->compressed_weights) {
            // The micro-gemm grouped matmul kernel requires fpmath_mode with apply_to_int
            // enabled whenever the activations are floating point but the weights use a
            // narrower integer type (weights-only compression / WOQ). Without this, the
            // oneDNN GPU implementations reject the problem as an unsupported
            // src/weights datatype combination.
            const auto input_dt = impl_params.get_input_layout(grouped_matmul::GroupedMatmulInputIdx::INPUT).data_type;
            const auto weights_dt = impl_params.get_input_layout(grouped_matmul::GroupedMatmulInputIdx::WEIGHT).data_type;
            const bool src_is_integral = cldnn::one_of(input_dt, {data_types::i8, data_types::u8});
            const bool weights_is_integral = cldnn::one_of(weights_dt, {data_types::i8, data_types::u8, data_types::i4, data_types::u4});
            if (!src_is_integral && weights_is_integral) {
                attr->set_fpmath_mode(dnnl::fpmath_mode::f16, true);
            }

            const auto& wei_shape = impl_params.get_input_layout(grouped_matmul::GroupedMatmulInputIdx::WEIGHT).get_shape();
            set_grouped_matmul_quant_attr(*attr, get_quant(impl_params), static_cast<dnnl::memory::dim>(wei_shape[2]));
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
