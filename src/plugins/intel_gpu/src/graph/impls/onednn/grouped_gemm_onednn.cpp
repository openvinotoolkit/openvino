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

    static std::unique_ptr<primitive_impl> create(const grouped_matmul_node& /*arg*/,
                                                  const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        auto attr = impl_params.attrs_onednn;

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
