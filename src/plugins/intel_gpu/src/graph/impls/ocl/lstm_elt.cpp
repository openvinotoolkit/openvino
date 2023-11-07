// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "lstm_elt_inst.h"
#include "lstm/lstm_elt_kernel_selector.h"
#include "lstm/lstm_elt_kernel_base.h"

namespace cldnn {
namespace ocl {

struct lstm_elt_impl : typed_primitive_impl_ocl<lstm_elt> {
    using parent = typed_primitive_impl_ocl<lstm_elt>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::lstm_elt_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::lstm_elt_params, kernel_selector::lstm_elt_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::lstm_elt_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<lstm_elt_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<lstm_elt>& instance) const override {
        kernel_arguments_data args = parent::get_arguments(instance);

        args.cell = instance.cell_term() ? instance.cell_memory() : nullptr;
        args.outputs = { instance.output_memory_ptr() };

        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<lstm_elt>();
        auto params = get_default_params<kernel_selector::lstm_elt_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::lstm_elt_optional_params>(impl_param.get_program());

        if (!primitive->cell.empty()) {
            const auto& cell_idx = 1;
            const auto& cell_layout = impl_param.input_layouts[cell_idx];
            params.SetCell(convert_data_tensor(cell_layout));
            // TODO: make a generic function to get the direction
            if (cell_layout.spatial(1) > 1) {
                params.cell_direction = primitive->direction;
            }
        }

        if (!primitive->activations.empty()) {
            auto a_sz = primitive->activations.size();
            auto param_sz = primitive->activation_params.size();
            OPENVINO_ASSERT(param_sz == 0|| a_sz == param_sz, "[GPU] Unexpected activation params count in lstm_elt impl: ", param_sz);
            for (size_t i = 0; i < a_sz; i++) {
                params.activations.emplace_back(get_kernel_selector_activation_param(primitive->activations[i]),
                                                         param_sz ? primitive->activation_params[i].a : 0.0f,
                                                         param_sz ? primitive->activation_params[i].b : 0.0f);
            }
        }

        if (primitive->clip > 0.0f) {
            params.activations.emplace_back(get_kernel_selector_activation_param(activation_func::clamp), -primitive->clip, primitive->clip);
        }

        params.SetOffsetOrder(static_cast<int32_t>(primitive->offset_order));
        params.clip = primitive->clip;
        params.input_forget = primitive->input_forget;
        params.direction = primitive->direction;

        return {params, optional_params};
    }

    static kernel_impl_params static_canonicalize_shapes(const kernel_impl_params& impl_params) {
        auto updated_impl_params = canonicalize_fused_shapes(impl_params);

        auto& input_layout = updated_impl_params.input_layouts[0];
        auto& weights_layout = updated_impl_params.input_layouts[1];
        auto& output_layout = updated_impl_params.output_layouts[0];

        auto input_pshape = input_layout.get_partial_shape();
        auto weights_pshape = weights_layout.get_partial_shape();
        auto output_pshape = output_layout.get_partial_shape();

        if (input_pshape.size() == 2) {
            auto lstm_input_size = static_cast<cldnn::tensor::value_type>(input_pshape[1].get_length());
            auto lstm_batch_size = static_cast<cldnn::tensor::value_type>(input_pshape[0].get_length());
            auto lstm_hidden_size = static_cast<cldnn::tensor::value_type>(lstm_input_size / 4);

            GPU_DEBUG_LOG << "lstm_input_size   : " << lstm_input_size << std::endl;
            GPU_DEBUG_LOG << "lstm_batch_size   : " << lstm_batch_size << std::endl;
            GPU_DEBUG_LOG << "lstm_hidden_size  : " << lstm_hidden_size << std::endl;

            GPU_DEBUG_LOG << "origin input_pshape   : " << input_layout.to_short_string() << std::endl;
            GPU_DEBUG_LOG << "origin weights_layout : " << weights_layout.to_short_string() << std::endl;

            input_pshape = {lstm_batch_size, 1, 1, lstm_input_size};
            input_layout.set_partial_shape(input_pshape);

            weights_pshape = {lstm_batch_size, 1, 1, lstm_hidden_size}; // {batch, direction, 1, hidden_size}
            weights_layout.format = format::adjust_to_rank(weights_layout.format, weights_pshape.size());
            weights_layout.set_partial_shape(weights_pshape);

            updated_impl_params.weights_layout = weights_layout;
        }

        GPU_DEBUG_LOG << "input_layout   : " << input_layout.to_short_string() << std::endl;
        GPU_DEBUG_LOG << "weights_layout : " << weights_layout.to_short_string() << std::endl;
        GPU_DEBUG_LOG << "output_layout  : " << output_layout.to_short_string() << std::endl;

        OPENVINO_ASSERT(input_pshape.size() == 4 && weights_pshape.size() == 4, "input and weights shape should be rank 4");
        return updated_impl_params;
    }

    kernel_impl_params canonicalize_shapes(const kernel_impl_params& impl_params) const override {
        return static_canonicalize_shapes(impl_params);
    }
};

namespace detail {

attach_lstm_elt_impl::attach_lstm_elt_impl() {
    implementation_map<lstm_elt>::add(impl_types::ocl, typed_primitive_impl_ocl<lstm_elt>::create<lstm_elt_impl>, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::f32, format::fyxb),
        std::make_tuple(data_types::f16, format::fyxb),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::lstm_elt_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::lstm_elt)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::lstm)
