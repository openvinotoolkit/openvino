// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_gemm_onednn.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "primitive_onednn_base.h"

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_ocl.hpp>

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
        auto& engine = instance.get_network().get_engine();
        auto& onednn_engine = engine.get_onednn_engine();
        auto moe_cfg = MoEGemmImplementationManager::get_moe_cfg(*instance.get_impl_params());

        {
            auto& input = instance.input_memory(moe_gemm::MoEGemmInputIdx::INPUT);
            auto& offsets = instance.input_memory(moe_gemm::MoEGemmInputIdx::INPUT_OFFSET_PER_EXPERT);
            OPENVINO_ASSERT(input.get_allocation_type() >= allocation_type::usm_host, "[GPU] oneDNN MOE GEMM implementation supports only USM input memory");
            dnnl::memory input_mem = dnnl::ocl_interop::make_memory(_pd.src_desc(0), onednn_engine, dnnl::ocl_interop::memory_kind::usm,
                {reinterpret_cast<uint8_t*>(input.buffer_ptr()), reinterpret_cast<uint8_t*>(offsets.buffer_ptr())});

            args.insert({DNNL_ARG_SRC, input_mem});
        }

        {
            auto& output = instance.output_memory(0);
            auto& offsets = instance.input_memory(moe_gemm::MoEGemmInputIdx::INPUT_OFFSET_PER_EXPERT);
            OPENVINO_ASSERT(output.get_allocation_type() >= allocation_type::usm_host, "[GPU] oneDNN MOE GEMM implementation supports only USM output memory");
            dnnl::memory output_mem = dnnl::ocl_interop::make_memory(_pd.dst_desc(0), onednn_engine, dnnl::ocl_interop::memory_kind::usm,
                {reinterpret_cast<uint8_t*>(output.buffer_ptr()), reinterpret_cast<uint8_t*>(offsets.buffer_ptr())});

            args.insert({DNNL_ARG_DST, output_mem});
        }

        {
            auto& weights = instance.input_memory(moe_gemm::MoEGemmInputIdx::WEIGHT);
            dnnl::memory weights_mem = dnnl::ocl_interop::make_memory(_pd.weights_desc(0), onednn_engine, dnnl::ocl_interop::memory_kind::usm,
                reinterpret_cast<uint8_t*>(weights.buffer_ptr()));
            args.insert({DNNL_ARG_WEIGHTS, weights_mem});
        }

        if (moe_cfg.is_weight_quantized) {
            auto& wei_scales = instance.input_memory(moe_cfg.weight_scale_idx);
            auto wei_scales_shape = wei_scales.get_layout().get_shape();
            dnnl::memory::dim d0 = wei_scales_shape[0];
            dnnl::memory::dim d1 = wei_scales_shape[1];
            dnnl::memory::dim d2 = wei_scales_shape[2];
            dnnl::memory::desc wei_scales_md(
                    {d0, d1, d2}, dnnl::memory::data_type::f16, dnnl::memory::format_tag::abc);
            dnnl::memory wei_scales_mem = dnnl::ocl_interop::make_memory(wei_scales_md, onednn_engine, dnnl::ocl_interop::memory_kind::usm,
                reinterpret_cast<uint8_t*>(wei_scales.buffer_ptr()));
            args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scales_mem});
        }

        if (moe_cfg.has_bias) {
            auto& bias = instance.input_memory(moe_gemm::MoEGemmInputIdx::BIAS);
            dnnl::memory bias_mem = dnnl::ocl_interop::make_memory(_pd.weights_desc(1), onednn_engine, dnnl::ocl_interop::memory_kind::usm,
                reinterpret_cast<uint8_t*>(bias.buffer_ptr()));
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
            attr->set_scales(DNNL_ARG_WEIGHTS,
                             (1 << 0) | (1 << 1) | (1 << 2),
                             {moe_cfg.weight_group_size, 1},
                             convert_data_type(impl_params.get_input_layout(moe_cfg.weight_scale_idx).data_type));
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

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::moe_gemm_onednn)
