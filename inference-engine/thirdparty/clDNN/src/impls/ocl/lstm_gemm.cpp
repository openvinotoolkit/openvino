// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "lstm_gemm_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "lstm/lstm_gemm_kernel_selector.h"
#include "lstm/lstm_gemm_kernel_base.h"
#include "network_impl.h"
#include "cldnn/runtime/error_handler.hpp"

namespace cldnn {
namespace ocl {

struct lstm_gemm_impl : typed_primitive_impl_ocl<lstm_gemm> {
    using parent = typed_primitive_impl_ocl<lstm_gemm>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<lstm_gemm_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(typed_primitive_inst<lstm_gemm>& instance, int32_t) const override {
        kernel_arguments_data args = parent::get_arguments(instance, 0);

        args.output = instance.output_memory_ptr();
        args.weights = instance.weights_memory();
        args.recurrent = instance.recurrent_memory();
        args.bias = instance.bias_term() ? instance.bias_memory() : nullptr;
        args.hidden = instance.hidden_term() ? instance.hidden_memory() : nullptr;

        return args;
    }

public:
    static primitive_impl* create(const lstm_gemm_node& arg) {
        const auto& weights_layout = arg.weights().get_output_layout();

        auto lstm_gemm_params = get_default_params<kernel_selector::lstm_gemm_params>(arg);
        lstm_gemm_params.weights = convert_data_tensor(weights_layout);

        if (arg.bias_term()) {
            const auto& bias_layout = arg.bias().get_output_layout();
            lstm_gemm_params.SetBias(convert_data_tensor(bias_layout));
        }
        if (arg.hidden_term()) {
            const auto& recurrent_layout = arg.recurrent().get_output_layout();
            lstm_gemm_params.recurrent = convert_data_tensor(recurrent_layout);

            const auto& hidden_layout = arg.hidden().get_output_layout();
            lstm_gemm_params.SetHidden(convert_data_tensor(hidden_layout));
            // TODO: make a generic function to get the direction
            if (hidden_layout.size.spatial[1] > 1) {
                lstm_gemm_params.hidden_direction = arg.direction();
            }
        }
        lstm_gemm_params.direction = arg.direction();

        // Update the direction of the input for the gemm kernel
        const auto& input_layout = arg.input().get_output_layout();
        size_t input_directions = input_layout.size.spatial[1];

        if (input_directions > 1) {  // For bidirection input, input direction can be 1 or 0
            lstm_gemm_params.input_direction = arg.direction();
        } else {  // For unidirectional input
            lstm_gemm_params.input_direction = 0;
        }

        auto lstm_gemm_optional_params =
            get_default_optional_params<kernel_selector::lstm_gemm_optional_params>(arg.get_program());

        auto& kernel_selector = kernel_selector::lstm_gemm_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(lstm_gemm_params, lstm_gemm_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto lstm_gemm = new lstm_gemm_impl(arg, best_kernels[0]);

        return lstm_gemm;
    }
};

namespace detail {

attach_lstm_gemm_impl::attach_lstm_gemm_impl() {
    implementation_map<lstm_gemm>::add(impl_types::ocl, lstm_gemm_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::f32, format::fyxb),
        std::make_tuple(data_types::f16, format::fyxb),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
