// Copyright (c) 2019 Intel Corporation
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

#include "contract_inst.h"

#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "error_handler.h"
#include "contract/contract_kernel_selector.h"
#include "contract/contract_kernel_base.h"

namespace cldnn {
namespace gpu {

namespace {
inline kernel_selector::ContractMode convert_to_contract_mode(contract_mode mode) {
    switch (mode) {
        case contract_mode::sum:
            return kernel_selector::ContractMode::SUM;
        case contract_mode::prod:
            return kernel_selector::ContractMode::PRODUCT;
        case contract_mode::all:
            return kernel_selector::ContractMode::ALL;
        case contract_mode::any:
            return kernel_selector::ContractMode::ANY;
        case contract_mode::max:
            return kernel_selector::ContractMode::MAX;

        default:
            return kernel_selector::ContractMode::SUM;
    }
}
}  // namespace

struct contract_gpu : typed_primitive_gpu_impl<contract> {
    using parent = typed_primitive_gpu_impl<contract>;
    using parent::parent;

    static primitive_impl* create(const contract_node& arg) {
        auto c_params = get_default_params<kernel_selector::contract_params>(arg, 1);
        auto c_optional_params =
            get_default_optional_params<kernel_selector::contract_optional_params>(arg.get_program());

        c_params.reduction_axes = arg.get_primitive()->reduction_axes;
        c_params.mode = convert_to_contract_mode(arg.get_primitive()->mode);

        auto& kernel_selector = kernel_selector::contract_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(c_params, c_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new contract_gpu(arg, best_kernels[0]);
    }
};

namespace detail {

attach_contract_gpu::attach_contract_gpu() {
    auto val_fw = contract_gpu::create;

    implementation_map<contract>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
    implementation_map<contract>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
    implementation_map<contract>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx), val_fw);
    implementation_map<contract>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfyx), val_fw);
    implementation_map<contract>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfyx), val_fw);
    implementation_map<contract>::add(std::make_tuple(engine_types::ocl, data_types::i64, format::bfyx), val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
