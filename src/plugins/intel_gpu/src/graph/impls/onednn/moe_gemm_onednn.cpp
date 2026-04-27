// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_gemm_onednn.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "primitive_onednn_base.h"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <memory>

namespace cldnn {
namespace onednn {

struct moe_gemm_onednn : typed_primitive_onednn_impl<moe_gemm> {
    using parent = typed_primitive_onednn_impl<moe_gemm>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::moe_gemm_onednn)

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<moe_gemm_onednn>(*this);
    }

    std::unordered_map<int, dnnl::memory> get_arguments(moe_gemm_inst& instance) const override {
        std::unordered_map<int, dnnl::memory> args;
        auto moe_cfg = MoEGemmImplementationManager::get_moe_cfg(*instance.get_impl_params());

        {
            auto& input = instance.input_memory(moe_gemm::MoEGemmInputIdx::INPUT);
            auto& offsets = instance.input_memory(moe_gemm::MoEGemmInputIdx::INPUT_OFFSET_PER_EXPERT);
            dnnl::memory input_mem = input.get_onednn_grouped_memory(_pd.src_desc(0), offsets);
            args.insert({DNNL_ARG_SRC, input_mem});
        }

        {
            auto& output = instance.output_memory(0);
            auto& offsets = instance.input_memory(moe_gemm::MoEGemmInputIdx::INPUT_OFFSET_PER_EXPERT);
            dnnl::memory output_mem = output.get_onednn_grouped_memory(_pd.dst_desc(0), offsets);
            args.insert({DNNL_ARG_DST, output_mem});
        }

        {
            auto& weights = instance.input_memory(moe_gemm::MoEGemmInputIdx::WEIGHT);
            dnnl::memory weights_mem = weights.get_onednn_memory(_pd.weights_desc(0), 0);
            args.insert({DNNL_ARG_WEIGHTS, weights_mem});
        }

        if (moe_cfg.is_weight_quantized) {
            // cldnn logical shape is [E, N, G]. onednn's set_scales expects scale dims to match
            // weight dim order [E, K, N] -> [E, G, N]. After prepare_quantization's byfx reorder
            // physical bytes are [E, G, N, 1]; declare dims swapped to {E, G, N} and use abc so
            // onednn's walk order matches physical.
            auto& wei_scales = instance.input_memory(moe_cfg.weight_scale_idx);
            auto wei_scales_shape = wei_scales.get_layout().get_shape();
            dnnl::memory::dim d0 = wei_scales_shape[0];
            dnnl::memory::dim d1 = wei_scales_shape[1];
            dnnl::memory::dim d2 = wei_scales_shape[2];
            // Cross-check: weight_group_size in moe_cfg must agree with the actual scale
            // tensor's group dim. Catches drift between MoEGemmImplementationManager::get_moe_cfg
            // (which derives weight_group_size from scale_shape[2] at compile time) and the
            // memory passed at runtime.
            const auto& weight_layout = instance.get_input_layout(moe_gemm::MoEGemmInputIdx::WEIGHT);
            const auto& w_shape = weight_layout.get_shape();
            const dnnl::memory::dim K = (w_shape.size() == 4) ? w_shape[2] * w_shape[3] : w_shape[2];
            const dnnl::memory::dim runtime_num_groups = (moe_cfg.weight_group_size == -1)
                ? 1
                : (K / moe_cfg.weight_group_size);
            const dnnl::memory::dim scale_num_groups = (wei_scales_shape.size() >= 3) ? d2 : 1;
            OPENVINO_ASSERT(scale_num_groups == runtime_num_groups,
                            "moe_gemm scale shape ", wei_scales_shape, " implies num_groups=",
                            scale_num_groups, " but moe_cfg.weight_group_size=", moe_cfg.weight_group_size,
                            " (K=", K, ") implies ", runtime_num_groups);
            dnnl::memory::dims wei_scales_dims = (moe_cfg.weight_group_size == -1)
                ? dnnl::memory::dims{d0, d1}
                : dnnl::memory::dims{d0, d2, d1};
            dnnl::memory::format_tag wei_scales_fmt = (moe_cfg.weight_group_size == -1)
                ? dnnl::memory::format_tag::ab
                : dnnl::memory::format_tag::abc;
            dnnl::memory::desc wei_scales_md(
                wei_scales_dims, convert_data_type(wei_scales.get_layout().data_type), wei_scales_fmt);
            dnnl::memory wei_scales_mem = wei_scales.get_onednn_memory(wei_scales_md, 0);
            args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scales_mem});

            if (!moe_cfg.is_weight_symmetric_quantized) {
                auto& wei_zp = instance.input_memory(moe_cfg.weight_zp_idx);
                const auto& zp_shape = wei_zp.get_layout().get_shape();
                OPENVINO_ASSERT(zp_shape == wei_scales_shape,
                                "moe_gemm scale shape ", wei_scales_shape, " does not match zp shape ", zp_shape);
                dnnl::memory::desc wei_zp_md(
                    wei_scales_dims, convert_data_type(wei_zp.get_layout().data_type), wei_scales_fmt);
                dnnl::memory wei_zp_mem = wei_zp.get_onednn_memory(wei_zp_md, 0);
                args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, wei_zp_mem});
            }
        }

        if (moe_cfg.has_bias) {
            auto& bias = instance.input_memory(moe_gemm::MoEGemmInputIdx::BIAS);
            dnnl::memory bias_mem = bias.get_onednn_memory(_pd.weights_desc(1), 0);
            args.insert({DNNL_ARG_BIAS, bias_mem});
        }

        return args;
    }

    static std::shared_ptr<dnnl::matmul::primitive_desc>
        get_moe_gemm_primitive_descriptor(const kernel_impl_params& impl_params,
                                          const dnnl::primitive_attr& attr = dnnl::primitive_attr()) {
        auto& engine = impl_params.prog->get_engine();
        auto prim = impl_params.typed_desc<moe_gemm>();
        auto moe_cfg = MoEGemmImplementationManager::get_moe_cfg(impl_params);

        auto input_layout = impl_params.get_input_layout(moe_gemm::MoEGemmInputIdx::INPUT);
        auto weights_layout = impl_params.get_input_layout(moe_gemm::MoEGemmInputIdx::WEIGHT);
        auto output_layout = impl_params.get_output_layout();

        dnnl::memory::dim total_tokens = prim->has_batch_dim ? input_layout.get_shape()[1] : input_layout.get_shape()[0];
        const auto& experts_weight_shape = weights_layout.get_shape();
        dnnl::memory::dim N = experts_weight_shape[1];
        dnnl::memory::dim K = experts_weight_shape.size() == 4 ? experts_weight_shape[2] * experts_weight_shape[3] : experts_weight_shape[2];
        dnnl::memory::dim num_experts = experts_weight_shape[0];

        dnnl::memory::dims input_dims = {total_tokens, K};
        dnnl::memory::dims weights_dims = {num_experts, K, N};
        dnnl::memory::dims output_dims = {total_tokens, N};

        auto input_md = dnnl::memory::desc::grouped(
                input_dims, convert_data_type(input_layout.data_type), 0, num_experts);
        auto output_md = dnnl::memory::desc::grouped(
                output_dims, convert_data_type(output_layout.data_type), 0, num_experts);
        auto weights_md = dnnl::memory::desc(
                weights_dims, convert_data_type(weights_layout.data_type), dnnl::memory::format_tag::acb);

        if (moe_cfg.has_bias) {
            auto bias_layout = impl_params.get_input_layout(moe_gemm::MoEGemmInputIdx::BIAS);
            auto bias_md = dnnl::memory::desc({num_experts, N}, convert_data_type(bias_layout.data_type), dnnl::memory::format_tag::ab);

            return std::make_shared<dnnl::matmul::primitive_desc>(
                engine.get_onednn_engine(),
                input_md,
                weights_md,
                bias_md,
                output_md,
                attr);
        } else {
            return std::make_shared<dnnl::matmul::primitive_desc>(
                engine.get_onednn_engine(),
                input_md,
                weights_md,
                output_md,
                attr);
        }
    }

public:
    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
    }

    static std::unique_ptr<primitive_impl> create(const moe_gemm_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        auto attr = impl_params.attrs_onednn;
        auto prim = impl_params.typed_desc<moe_gemm>();
        auto moe_cfg = MoEGemmImplementationManager::get_moe_cfg(impl_params);

        if (moe_cfg.is_weight_quantized) {
            if (moe_cfg.weight_group_size == -1) {
                attr->set_scales(DNNL_ARG_WEIGHTS,
                                 (1 << 0) | (1 << 2),
                                 {},
                                 convert_data_type(impl_params.get_input_layout(moe_cfg.weight_scale_idx).data_type));
            } else {
                attr->set_scales(DNNL_ARG_WEIGHTS,
                                 (1 << 0) | (1 << 1) | (1 << 2),
                                 {moe_cfg.weight_group_size, 1},
                                 convert_data_type(impl_params.get_input_layout(moe_cfg.weight_scale_idx).data_type));
            }

            if (!moe_cfg.is_weight_symmetric_quantized) {
                if (moe_cfg.weight_group_size == -1) {
                    attr->set_zero_points(DNNL_ARG_WEIGHTS,
                                         (1 << 0) | (1 << 2),
                                         {},
                                         convert_data_type(impl_params.get_input_layout(moe_cfg.weight_zp_idx).data_type));
                } else {
                    attr->set_zero_points(DNNL_ARG_WEIGHTS,
                                          (1 << 0) | (1 << 1) | (1 << 2),
                                          {moe_cfg.weight_group_size, 1},
                                          convert_data_type(impl_params.get_input_layout(moe_cfg.weight_zp_idx).data_type));
                }
            }
        }

        auto prim_desc = get_moe_gemm_primitive_descriptor(impl_params, *attr);

        return std::make_unique<moe_gemm_onednn>(engine, config, attr, *prim_desc);
    }
};

std::unique_ptr<primitive_impl> MoEGemmImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const  {
    OPENVINO_ASSERT(node.is_type<moe_gemm>());
    return onednn::moe_gemm_onednn::create(static_cast<const moe_gemm_node&>(node), params);
}

}  // namespace onednn
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::moe_gemm)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::moe_gemm_onednn)
