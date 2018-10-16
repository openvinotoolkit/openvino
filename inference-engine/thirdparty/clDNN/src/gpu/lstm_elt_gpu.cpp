/*
// Copyright (c) 2016 Intel Corporation
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

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "lstm_elt_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "lstm/lstm_elt_kernel_selector.h"
#include "lstm/lstm_elt_kernel_base.h"
#include "network_impl.h"
#include "error_handler.h"
#include "lstm/lstm_elt_kernel_selector.h"
#include "lstm/lstm_elt_kernel_base.h"


namespace cldnn { namespace gpu {

struct lstm_elt_gpu : typed_primitive_gpu_impl<lstm_elt>
{
    using parent = typed_primitive_gpu_impl<lstm_elt>;
    using parent::parent;

protected:

    virtual kernel::kernel_arguments_data get_arguments(typed_primitive_inst<lstm_elt>& instance, int32_t) const override
    {
        kernel::kernel_arguments_data args = parent::get_arguments(instance, 0);

        args.cell       = instance.cell_term() ? &instance.cell_memory() : nullptr;
        args.output     = &instance.output_memory();

        return args;
    }

    virtual bool validate(typed_primitive_inst<lstm_elt>& instance) const override
    {
        bool res = parent::validate(instance);

        return res;
    }
public:

    static primitive_impl* create(const lstm_elt_node& arg)
    {
        auto lstm_elt_params = get_default_params<kernel_selector::lstm_elt_params>(arg);
        auto lstm_elt_optional_params = get_default_optional_params<kernel_selector::lstm_elt_optional_params>(arg.get_program());

        if (arg.cell_term())
        {
            const auto& cell_layout = arg.cell().get_output_layout();
            lstm_elt_params.SetCell(convert_data_tensor(cell_layout));
        }

        lstm_elt_params.SetOffsetOrder(arg.offset_order());
        lstm_elt_params.clip = arg.clip();
        lstm_elt_params.input_forget = arg.input_forget();

        auto& kernel_selector = kernel_selector::lstm_elt_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(lstm_elt_params, lstm_elt_optional_params);

        CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");

        auto lstm_elt = new lstm_elt_gpu(arg, best_kernels[0]);

        return lstm_elt;
    };
};


namespace {
    struct attach {
        attach() {
            auto val_fw = lstm_elt_gpu::create;

            implementation_map<lstm_elt>::add({
                { std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw },
                { std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw },
            });
        }
        ~attach() {}
    };
    attach attach_impl;
}
} }
