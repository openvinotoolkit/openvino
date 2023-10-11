// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "lstm_dynamic_timeloop_inst.h"
#include "lstm_dynamic/lstm_dynamic_timeloop_kernel_selector.h"
#include "lstm_dynamic/lstm_dynamic_timeloop_kernel_base.h"

namespace cldnn {
namespace ocl {

struct lstm_dynamic_timeloop_impl : typed_primitive_impl_ocl<lstm_dynamic_timeloop> {
    using parent = typed_primitive_impl_ocl<lstm_dynamic_timeloop>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::lstm_dynamic_timeloop_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::lstm_dynamic_timeloop_params, kernel_selector::lstm_dynamic_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::lstm_dynamic_timeloop_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<lstm_dynamic_timeloop_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<lstm_dynamic_timeloop>& instance) const override {
        kernel_arguments_data args;
        args.inputs = {instance.input_memory_ptr(), instance.dyn_length_memory()};
        if (instance.last_hidden_output_term())
            args.inputs.push_back(instance.last_hidden_output_memory());
        if (instance.last_cell_output_term())
            args.inputs.push_back(instance.last_cell_output_memory());
        args.outputs = { instance.output_memory_ptr() };
        args.recurrent = instance.recurrent_memory();
        args.hidden = instance.initial_hidden_term() ? instance.initial_hidden_memory() : nullptr;
        args.cell = instance.initial_cell_term() ? instance.initial_cell_memory() : nullptr;
        return args;
    }

public:
    static std::unique_ptr<primitive_impl> create(const lstm_dynamic_timeloop_node& arg, const kernel_impl_params& impl_param) {
        auto dlstm_timeloop_params = get_default_params<kernel_selector::lstm_dynamic_timeloop_params>(impl_param);

        // dyn length
        const auto& dyn_length_tensor = impl_param.input_layouts[arg.get_dependency_idx("dyn_length")];
        dlstm_timeloop_params.inputs.push_back(convert_data_tensor(dyn_length_tensor));

        // recurrent
        const auto& recurrent_layout = impl_param.input_layouts[arg.get_dependency_idx("recurrent")];
        dlstm_timeloop_params.recurrent = convert_data_tensor(recurrent_layout);

        dlstm_timeloop_params.direction = arg.direction();

        if (arg.initial_cell_term()) {
            const auto& cell_layout = impl_param.input_layouts[arg.get_dependency_idx("initial_cell")];
            dlstm_timeloop_params.set_cell(convert_data_tensor(cell_layout));
        }

        if (arg.last_hidden_output_term()) {
            const auto& last_hidden_output_layout = impl_param.input_layouts[arg.get_dependency_idx("last_hidden_output")];
            dlstm_timeloop_params.set_last_hidden_output(convert_data_tensor(last_hidden_output_layout));
        }

        if (arg.initial_hidden_term()) {
            const auto& hidden_layout = impl_param.input_layouts[arg.get_dependency_idx("initial_hidden")];
            dlstm_timeloop_params.set_hidden(convert_data_tensor(hidden_layout));
        }

        if (arg.last_cell_output_term()) {
            const auto& last_cell_state_layout = impl_param.input_layouts[arg.get_dependency_idx("last_cell_output")];
            dlstm_timeloop_params.set_last_cell_output(convert_data_tensor(last_cell_state_layout));
        }
        dlstm_timeloop_params.set_dynamic_shape_offsets();
        // finially get best kernel
        auto dlstm_timeloop_optional_params =
            get_default_optional_params<kernel_selector::lstm_dynamic_optional_params>(impl_param.get_program());

        auto& kernel_selector = kernel_selector::lstm_dynamic_timeloop_kernel_selector::Instance();
        auto best_kernel = kernel_selector.get_best_kernel(dlstm_timeloop_params, dlstm_timeloop_optional_params);

        return make_unique<lstm_dynamic_timeloop_impl>(best_kernel);
    }
};

namespace detail {

attach_lstm_dynamic_timeloop_impl::attach_lstm_dynamic_timeloop_impl() {
    implementation_map<lstm_dynamic_timeloop>::add(impl_types::ocl, lstm_dynamic_timeloop_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::lstm_dynamic_timeloop_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::lstm_dynamic_timeloop)
