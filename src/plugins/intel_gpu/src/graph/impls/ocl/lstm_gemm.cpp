// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "lstm_gemm_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "lstm/lstm_gemm_kernel_selector.h"
#include "lstm/lstm_gemm_kernel_base.h"
#include "intel_gpu/runtime/error_handler.hpp"

namespace cldnn {
namespace ocl {

struct lstm_gemm_impl : typed_primitive_impl_ocl<lstm_gemm> {
    using parent = typed_primitive_impl_ocl<lstm_gemm>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::lstm_gemm_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::lstm_gemm_params, kernel_selector::lstm_gemm_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<lstm_gemm_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<lstm_gemm>& instance, int32_t) const override {
        kernel_arguments_data args = parent::get_arguments(instance, 0);

        args.outputs = { instance.output_memory_ptr() };
        args.weights = instance.weights_memory();
        args.recurrent = instance.recurrent_memory();
        args.bias = instance.bias_term() ? instance.bias_memory() : nullptr;
        args.hidden = instance.hidden_term() ? instance.hidden_memory() : nullptr;

        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<lstm_gemm>();
        const auto input_idx = 0;
        const auto weight_idx = 1;
        const auto recurrent_idx = 2;
        const auto bias_idx = 3;
        const bool bias_term = !primitive->bias.empty();
        const bool hidden_term = !primitive->hidden.empty();
        const auto hidden_idx = bias_term ? 4 : 3;
        const auto direction = primitive->direction;

        const auto& weights_layout = impl_param.input_layouts[weight_idx];
        auto lstm_gemm_params = get_default_params<kernel_selector::lstm_gemm_params>(impl_param);
        lstm_gemm_params.weights = convert_data_tensor(weights_layout);

        if (bias_term) {
            const auto& bias_layout = impl_param.input_layouts[bias_idx];
            lstm_gemm_params.SetBias(convert_data_tensor(bias_layout));
        }
        if (hidden_term) {
            const auto& recurrent_layout = impl_param.input_layouts[recurrent_idx];
            lstm_gemm_params.recurrent = convert_data_tensor(recurrent_layout);

            const auto& hidden_layout = impl_param.input_layouts[hidden_idx];
            lstm_gemm_params.SetHidden(convert_data_tensor(hidden_layout));
            // TODO: make a generic function to get the direction
            if (hidden_layout.spatial(1) > 1) {
                lstm_gemm_params.hidden_direction = direction;
            }
        }
        lstm_gemm_params.direction = direction;

        // Update the direction of the input for the gemm kernel
        const auto& input_layout = impl_param.input_layouts[input_idx];
        size_t input_directions = input_layout.spatial(1);

        if (input_directions > 1) {  // For bidirection input, input direction can be 1 or 0
            lstm_gemm_params.input_direction = direction;
        } else {  // For unidirectional input
            lstm_gemm_params.input_direction = 0;
        }

        auto lstm_gemm_optional_params =
            get_default_optional_params<kernel_selector::lstm_gemm_optional_params>(impl_param.get_program());

        return {lstm_gemm_params, lstm_gemm_optional_params};
    }
};

namespace detail {

attach_lstm_gemm_impl::attach_lstm_gemm_impl() {
    implementation_map<lstm_gemm>::add(impl_types::ocl, typed_primitive_impl_ocl<lstm_gemm>::create<lstm_gemm_impl>, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::f32, format::fyxb),
        std::make_tuple(data_types::f16, format::fyxb),
    });

    impl_hash_key<lstm_gemm>::add(typed_primitive_impl_ocl<lstm_gemm>::get_impl_key<lstm_gemm_impl>);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::lstm_gemm_impl)
