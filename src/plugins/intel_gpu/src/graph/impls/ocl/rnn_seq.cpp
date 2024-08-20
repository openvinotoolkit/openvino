// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "lstm_seq_inst.h"
#include "lstm/lstm_seq_kernel_selector.h"
#include "lstm/lstm_kernel_base.h"
#include "openvino/op/lstm_sequence.hpp"

namespace cldnn {
namespace ocl {

struct rnn_seq_impl : typed_primitive_impl_ocl<lstm_seq> {
    using parent = typed_primitive_impl_ocl<lstm_seq>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::lstm_seq_kernel_selector;
    using kernel_params_t = kernel_selector::lstm_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::rnn_seq_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<rnn_seq_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<lstm_seq>& instance) const override {
        kernel_arguments_data args;
        size_t op_input_size = 4 + (instance.has_cell() ? 1 : 0);
        for (size_t i = 0; i < op_input_size; i++) {
            args.inputs.push_back(instance.input_memory_ptr(i));
        }

        for (size_t i = 0; i < instance.outputs_memory_count(); i++) {
            args.outputs.push_back(instance.output_memory_ptr(i));
        }
        for (size_t i = op_input_size; i < instance.inputs_memory_count(); i++) {
            args.outputs.push_back(instance.dep_memory_ptr(i));
        }
        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<lstm_seq>();
        auto params = get_default_params<kernel_selector::lstm_params>(impl_param);
        for (size_t i = 1; i < 5; ++i) {
            params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
        }
        if (!primitive->params.activations.empty()) {
            auto a_sz = primitive->params.activations.size();
            auto param_sz = primitive->params.activation_params.size();
            OPENVINO_ASSERT(param_sz == 0|| a_sz == param_sz, "[GPU] Unexpected activation params count in lstm_seq impl: ", param_sz);
            for (size_t i = 0; i < a_sz; i++) {
                params.activations.emplace_back(get_kernel_selector_activation_param(primitive->params.activations[i]),
                                                         param_sz ? primitive->params.activation_params[i].a : 0.0f,
                                                         param_sz ? primitive->params.activation_params[i].b : 0.0f);
            }
        }

        if (primitive->params.clip > 0.0f) {
            params.activations.emplace_back(get_kernel_selector_activation_param(activation_func::clamp), -primitive->params.clip, primitive->params.clip);
        }

        params.SetOffsetOrder(static_cast<int32_t>(primitive->params.offset_order));
        params.clip = primitive->params.clip;
        params.direction = primitive->params.direction;
        //Legacy multi-output
        params.outputs.push_back(convert_data_tensor(impl_param.input_layouts[1]));
        if (!primitive->params.initial_cell_state.pid.empty()) {
            params.outputs.push_back(convert_data_tensor(impl_param.input_layouts[1]));
        }
        return params;
    }

    static kernel_impl_params static_canonicalize_shapes(const kernel_impl_params& impl_params) {
        if (impl_params.get_input_layout().get_partial_shape().size() != 3) {
            return primitive_impl::static_canonicalize_shapes(impl_params);
        }
        auto updated_impl_params = canonicalize_fused_shapes(impl_params);
        return updated_impl_params;
    }

    kernel_impl_params canonicalize_shapes(const kernel_impl_params& impl_params) const override {
        return static_canonicalize_shapes(impl_params);
    }
};

namespace detail {

attach_lstm_seq_impl::attach_lstm_seq_impl() {
    implementation_map<lstm_seq>::add(impl_types::ocl, typed_primitive_impl_ocl<lstm_seq>::create<rnn_seq_impl>, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::f32, format::fyxb),
        std::make_tuple(data_types::f16, format::fyxb),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::rnn_seq_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::lstm_seq)
