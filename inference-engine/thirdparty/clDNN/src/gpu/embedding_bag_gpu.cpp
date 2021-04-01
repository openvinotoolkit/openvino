/*
// Copyright (c) 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "embedding_bag_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "embedding_bag/embedding_bag_kernel_selector.h"
#include "embedding_bag/embedding_bag_kernel_ref.h"
#include "error_handler.h"
#include "data_inst.h"

using namespace cldnn;

namespace cldnn {
namespace gpu {
struct embedding_bag_gpu : typed_primitive_gpu_impl<embedding_bag> {
    using parent = typed_primitive_gpu_impl<embedding_bag>;
    using parent::parent;

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

        auto embedding_bag = new embedding_bag_gpu(arg, best_kernels[0]);

        return embedding_bag;
    }
};

namespace detail {

attach_embedding_bag_gpu::attach_embedding_bag_gpu() {
    auto val_fw = embedding_bag_gpu::create;
    implementation_map<embedding_bag>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
    implementation_map<embedding_bag>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
