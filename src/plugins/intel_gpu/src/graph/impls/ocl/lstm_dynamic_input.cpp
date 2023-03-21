// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "lstm_dynamic_input_inst.h"
#include "lstm_dynamic/lstm_dynamic_input_kernel_selector.h"
#include "lstm_dynamic/lstm_dynamic_input_kernel_base.h"

namespace cldnn {
namespace ocl {

struct lstm_dynamic_input_impl : typed_primitive_impl_ocl<lstm_dynamic_input> {
    using parent = typed_primitive_impl_ocl<lstm_dynamic_input>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::lstm_dynamic_input_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::lstm_dynamic_input_params, kernel_selector::lstm_dynamic_input_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<lstm_dynamic_input_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<lstm_dynamic_input>& instance) const override {
        kernel_arguments_data args;
        args.inputs = { instance.input_memory_ptr(), instance.dyn_length_memory()};
        args.outputs = { instance.output_memory_ptr() };
        args.weights = instance.weights_memory();
        args.bias = instance.bias_term() ? instance.bias_memory() : nullptr;
        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<lstm_dynamic_input>();
        auto params = get_default_params<kernel_selector::lstm_dynamic_input_params>(impl_param);

        const auto dyn_len_idx = 1;
        const auto weights_idx = 2;
        const auto bias_idx = 3;

        const auto& weights_layout = impl_param.get_input_layout(weights_idx);
        params.weights = convert_weights_tensor(weights_layout);

        auto has_bias = !primitive->bias.empty();
        if (has_bias) {
            const auto& bias_layout = impl_param.get_input_layout(bias_idx);
            params.bias.push_back(convert_data_tensor(bias_layout));
        }

        const auto& dyn_length_tensor = impl_param.input_layouts[dyn_len_idx];
        params.inputs.push_back(convert_data_tensor(dyn_length_tensor));

        params.direction = weights_layout.feature();

        auto optional_params = get_default_weights_bias_optional_params<kernel_selector::lstm_dynamic_input_optional_params>(impl_param.get_program());
        return {params, optional_params};
    }
};

namespace detail {

attach_lstm_dynamic_input_impl::attach_lstm_dynamic_input_impl() {
    implementation_map<lstm_dynamic_input>::add(impl_types::ocl, typed_primitive_impl_ocl<lstm_dynamic_input>::create<lstm_dynamic_input_impl>, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::lstm_dynamic_input_impl)
