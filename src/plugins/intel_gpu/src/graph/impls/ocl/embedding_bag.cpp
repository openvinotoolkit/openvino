// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "embedding_bag_inst.h"
#include "embedding_bag/embedding_bag_kernel_selector.h"
#include "embedding_bag/embedding_bag_kernel_ref.h"

namespace cldnn {
namespace ocl {
namespace {
std::vector<size_t> get_kernel_arg_indices(size_t num_inputs, embedding_bag::embedding_bag_type type) {
    std::vector<size_t> input_idx = {0, 1}; // common input indices
    switch (type) {
        case embedding_bag::packed_sum: {
            if (num_inputs == 3) {
                input_idx.push_back(2); // optional per_sample_weights
            }
            break;
        }
        case embedding_bag::offsets_sum: {
            input_idx.push_back(2); // offsets
            if (num_inputs == 5) {
                input_idx.push_back(4); // optional per_sample_weights
            }
            break;
        }
        case embedding_bag::segments_sum:
            input_idx.push_back(2); // segment_ids
            if (num_inputs == 6) {
                input_idx.push_back(5); // optional per_sample_weights
            }
            break;
        }
    return input_idx;
}

}  // namespace
struct embedding_bag_impl : typed_primitive_impl_ocl<embedding_bag> {
    using parent = typed_primitive_impl_ocl<embedding_bag>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::embedding_bag_kernel_selector;
    using kernel_params_t = kernel_selector::embedding_bag_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::embedding_bag_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<embedding_bag_impl, kernel_params_t>(*this);
    }

    kernel_arguments_data get_arguments(const typed_primitive_inst<embedding_bag>& instance) const override {
        kernel_arguments_data args = parent::get_arguments(instance);
        args.inputs.clear();
        auto primitive = instance.get_typed_desc<embedding_bag>();

        auto input_idx = get_kernel_arg_indices(primitive->input_size(), primitive->type);
        for (size_t i = 0; i < input_idx.size(); i++) {
            args.inputs.push_back(instance.input_memory_ptr(input_idx[i]));
        }

        return args;
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<embedding_bag>();
        auto params = get_default_params<kernel_selector::embedding_bag_params>(impl_param);
        switch (primitive->type) {
        case embedding_bag::packed_sum:
            params.type = kernel_selector::EmbeddingBagType::PACKED_SUM;
            break;
        case embedding_bag::offsets_sum:
            params.type = kernel_selector::EmbeddingBagType::OFFSETS_SUM;
            break;
        case embedding_bag::segments_sum:
            params.type = kernel_selector::EmbeddingBagType::SEGMENTS_SUM;
            break;
        default: OPENVINO_ASSERT(false, "[GPU] Unknown embedding_bag type in primitive ", primitive->id);
        }

        params.inputs.clear();
        auto input_idx = get_kernel_arg_indices(primitive->input_size(), primitive->type);
        for (size_t i = 0; i < input_idx.size(); i++) {
            params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[input_idx[i]]));
        }

        params.default_index = primitive->default_index;
        return params;
    }
};

namespace detail {

attach_embedding_bag_impl::attach_embedding_bag_impl() {
    implementation_map<embedding_bag>::add(impl_types::ocl, typed_primitive_impl_ocl<embedding_bag>::create<embedding_bag_impl>, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::embedding_bag_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::embedding_bag)
