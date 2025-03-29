// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "lstm_seq_inst.h"
#include "rnn_seq.hpp"
#include "lstm/lstm_cell_and_seq_kernel_selector.h"
#include "lstm/lstm_kernel_base.h"
#include "openvino/op/lstm_sequence.hpp"
#include "registry/implementation_manager.hpp"

namespace cldnn {
namespace ocl {

struct rnn_seq_impl : typed_primitive_impl_ocl<lstm_seq> {
    using parent = typed_primitive_impl_ocl<lstm_seq>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::lstm_cell_and_seq_kernel_selector;
    using kernel_params_t = kernel_selector::lstm_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::rnn_seq_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<rnn_seq_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<lstm_seq>& instance) const override {
        kernel_arguments_data args;
        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            args.inputs.push_back(instance.input_memory_ptr(i));
        }

        for (size_t i = 0; i < instance.outputs_memory_count(); i++) {
            args.outputs.push_back(instance.output_memory_ptr(i));
        }
        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<lstm_seq>();
        auto params = get_default_params<kernel_selector::lstm_params>(impl_param);
        params.sequential = true;
        for (size_t i = 1; i < impl_param.input_layouts.size(); ++i) {
            params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
        }

        if (!primitive->activations.empty()) {
            auto a_sz = primitive->activations.size();
            auto param_sz = primitive->activation_params.size();
            OPENVINO_ASSERT(param_sz == 0|| a_sz == param_sz, "[GPU] Unexpected activation params count in lstm_seq impl: ", param_sz);
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
        params.direction = primitive->direction;
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

std::unique_ptr<primitive_impl> RNNSeqImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<lstm_seq>());
    return typed_primitive_impl_ocl<lstm_seq>::create<rnn_seq_impl>(static_cast<const lstm_seq_node&>(node), params);
}

}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::rnn_seq_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::lstm_seq)
