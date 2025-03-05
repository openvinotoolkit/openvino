// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "embedding_bag_inst.h"
#include "embedding_bag/embedding_bag_kernel_selector.h"
#include "embedding_bag/embedding_bag_kernel_ref.h"

namespace cldnn {
namespace ocl {
struct embedding_bag_impl : typed_primitive_impl_ocl<embedding_bag> {
    using parent = typed_primitive_impl_ocl<embedding_bag>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::embedding_bag_kernel_selector;
    using kernel_params_t = kernel_selector::embedding_bag_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::embedding_bag_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<embedding_bag_impl, kernel_params_t>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<embedding_bag>();
        auto params = get_default_params<kernel_selector::embedding_bag_params>(impl_param);

        auto inputs_count = impl_param.input_layouts.size();

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

        for (size_t i = 1; i < inputs_count; i++) {
            params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[i]));
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
        std::make_tuple(data_types::i32, format::bfyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::embedding_bag_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::embedding_bag)
