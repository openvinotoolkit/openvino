// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"
#include "scaled_dot_product_attention_inst.h"
#include "sdpa/sdpa_kernel_selector.h"
#include "sdpa/sdpa_kernel_base.h"

namespace cldnn {
namespace ocl {
struct scaled_dot_product_attention_impl : typed_primitive_impl_ocl<scaled_dot_product_attention> {
    using parent = typed_primitive_impl_ocl<scaled_dot_product_attention>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::sdpa_kernel_selector;
    using kernel_params_t = kernel_selector::sdpa_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::scaled_dot_product_attention_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<scaled_dot_product_attention_impl>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

    static kernel_selector::sdpa_configuration get_sdpa_configuration(const kernel_impl_params& impl_param) {
        kernel_selector::sdpa_configuration config;

        auto transpose_pshape = [](const ov::PartialShape& pshape, const std::vector<int64_t>& order) {
            if (order.empty())
                return pshape;

            auto transposed_pshape = ov::PartialShape::dynamic(pshape.rank());
            for (size_t i = 0; i < order.size(); i++) {
                transposed_pshape[i] = pshape[order[i]];
            }
            return transposed_pshape;
        };

        const auto& prim = impl_param.typed_desc<scaled_dot_product_attention>();
        const auto query_shape = transpose_pshape(impl_param.get_input_layout(0).get_partial_shape(), prim->input_q_transpose_order);
        const auto key_shape = transpose_pshape(impl_param.get_input_layout(1).get_partial_shape(), prim->input_k_transpose_order);
        const auto value_shape = transpose_pshape(impl_param.get_input_layout(2).get_partial_shape(), prim->input_v_transpose_order);

        OPENVINO_ASSERT(key_shape == value_shape, "[GPU] The shapes of key and value inputs are expected to be equal");
        for (size_t i = 0; i < query_shape.size(); ++i) {
            if (query_shape[i].is_static() && key_shape[i].is_static() && value_shape[i].is_static()) {
                if (query_shape[i].get_length() > key_shape[i].get_length()) {
                    config.broadcast_axis = prim->input_k_transpose_order[i];
                    config.group_size = query_shape[i].get_length() / key_shape[i].get_length();
                }
            }
        }

        if (query_shape[query_shape.size() - 1].is_static())
            config.head_size = query_shape[query_shape.size() - 1].get_length();

        config.is_causal = prim->is_causal;

        return config;
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_dynamic) {
        auto params = get_default_params<kernel_selector::sdpa_params>(impl_param, is_dynamic);

        const auto inputs_num = impl_param.input_layouts.size();
        params.inputs.resize(inputs_num);
        for (size_t i = 0; i < inputs_num; i++) {
            params.inputs[i] = convert_data_tensor(impl_param.get_input_layout(i));
        }

        params.conf = get_sdpa_configuration(impl_param);

        const auto& prim = impl_param.typed_desc<scaled_dot_product_attention>();
        params.input0_order = prim->input_q_transpose_order;
        params.input1_order = prim->input_k_transpose_order;
        params.input2_order = prim->input_v_transpose_order;
        params.output_order = prim->output_transpose_order;

        params.set_dynamic_shape_offsets();

        return params;
    }

    static std::unique_ptr<primitive_impl> create(const typed_program_node<scaled_dot_product_attention>& arg, const kernel_impl_params& impl_param) {
        auto sdpa_kernel_params = get_kernel_params(impl_param, impl_param.is_dynamic());
        auto& sdpa_kernel_selector = kernel_selector_t::Instance();
        auto kd = sdpa_kernel_selector.get_best_kernel(sdpa_kernel_params);

        return cldnn::make_unique<scaled_dot_product_attention_impl>(kd);
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
       auto kernel_params = get_kernel_params(impl_param, true);
       (_kernel_data.update_dispatch_data_func)(kernel_params, _kernel_data);
    }
};

namespace detail {

attach_scaled_dot_product_attention_impl::attach_scaled_dot_product_attention_impl() {
    using sdpa_prim = scaled_dot_product_attention;

    auto types = {
        data_types::f32,
        data_types::f16,
    };

    auto formats = {
        format::bfyx,
    };

    implementation_map<sdpa_prim>::add(impl_types::ocl,
                                       shape_types::static_shape,
                                       scaled_dot_product_attention_impl::create,
                                       types,
                                       formats);

    implementation_map<sdpa_prim>::add(impl_types::ocl,
                                       shape_types::dynamic_shape,
                                       scaled_dot_product_attention_impl::create,
                                       types,
                                       formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::scaled_dot_product_attention_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::scaled_dot_product_attention)
