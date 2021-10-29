// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "embedding_bag_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "embedding_bag/embedding_bag_kernel_selector.h"
#include "embedding_bag/embedding_bag_kernel_ref.h"
#include "cldnn/runtime/error_handler.hpp"
#include "data_inst.h"

using namespace cldnn;

namespace cldnn {
namespace ocl {
struct embedding_bag_impl : typed_primitive_impl_ocl<embedding_bag> {
    using parent = typed_primitive_impl_ocl<embedding_bag>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<embedding_bag_impl>(*this);
    }

public:
    static primitive_impl* create(const embedding_bag_node& arg) {
        auto embedding_bag_params = get_default_params<kernel_selector::embedding_bag_params>(arg);
        auto embedding_bag_optional_params =
            get_default_optional_params<kernel_selector::embedding_bag_optional_params>(arg.get_program());

        switch (arg.get_primitive()->type) {
        case embedding_bag::packed_sum:
            embedding_bag_params.type = kernel_selector::EmbeddingBagType::PACKED_SUM;
            break;
        case embedding_bag::offsets_sum:
            embedding_bag_params.type = kernel_selector::EmbeddingBagType::OFFSETS_SUM;
            break;
        case embedding_bag::segments_sum:
            embedding_bag_params.type = kernel_selector::EmbeddingBagType::SEGMENTS_SUM;
            break;
        default:
            CLDNN_ERROR_MESSAGE(arg.id(), "Unknown EmbeddingBag type");
            break;
        }

        for (size_t i = 1; i < arg.inputs_count(); i++) {
            embedding_bag_params.inputs.push_back(convert_data_tensor(arg.input(i).get_output_layout()));
        }

        embedding_bag_params.default_index = arg.get_primitive()->default_index;

        auto& kernel_selector = kernel_selector::embedding_bag_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(embedding_bag_params, embedding_bag_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto embedding_bag = new embedding_bag_impl(arg, best_kernels[0]);

        return embedding_bag;
    }
};

namespace detail {

attach_embedding_bag_impl::attach_embedding_bag_impl() {
    implementation_map<embedding_bag>::add(impl_types::ocl, embedding_bag_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
