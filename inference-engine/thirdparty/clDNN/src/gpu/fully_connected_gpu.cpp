/*
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
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "fully_connected_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "fully_connected/fully_connected_kernel_selector.h"
#include "fully_connected/fully_connected_params.h"

#include "network_impl.h"
#include "error_handler.h"
#include "kernel_runner.h"

#include "api/CPP/reorder.hpp"
#include "api/CPP/input_layout.hpp"

namespace cldnn { namespace gpu {


struct fully_connected_gpu : typed_primitive_gpu_impl<fully_connected>
{
    using parent = typed_primitive_gpu_impl<fully_connected>;

    std::vector<network_impl::ptr> _reorders;   // TODO: move this reorder to graph compiler
    memory_impl::cptr new_input_mem;      // TODO: remove this hack

    fully_connected_gpu(const fully_connected_node& arg, const kernel_selector::kernel_data& kd, std::vector<network_impl::ptr> reorders)
        : parent(arg, kd)
        , _reorders(reorders)
    {}

protected:

    virtual kernel::kernel_arguments_data get_arguments(typed_primitive_inst<fully_connected>& instance, int32_t) const override
    {
        kernel::kernel_arguments_data args;

        args.inputs     = { new_input_mem };
        args.output     = &instance.output_memory();
        args.weights    = &instance.weights_memory();
        args.bias       = instance.bias_term() ? &instance.bias_memory() : nullptr;
        args.weights_quantization_factors = instance.weights_quantization_factors_term() ? &instance.weights_quantization_factors_memory() : nullptr;
        args.output_calibration_factors = instance.output_calibration_factors_term() ? &instance.output_calibration_factors_memory() : nullptr;

        return args;
    }

public:

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, fully_connected_inst& instance) override
    {
        std::vector<event_impl::ptr> tmp_events(events);

        if (_reorders.empty())
        {
            new_input_mem = &instance.input_memory();
        }
        else
        {
            auto network = _reorders[0];
            network->set_input_data("input", instance.input_memory());
            network->execute(tmp_events);
            auto output_id = network->get_output_ids()[0];
            new_input_mem = &network->get_primitive(output_id)->output_memory();
            tmp_events.clear();
            tmp_events.push_back(network->get_primitive_event(output_id));
        }

        return parent::execute_impl(tmp_events, instance);
    }

    static primitive_impl* create(const fully_connected_node& arg)
    {
        auto fc_params = get_weights_bias_default_params<kernel_selector::fully_connected_params>(arg);
        auto fc_optional_params = get_default_weights_bias_optional_params<kernel_selector::fully_connected_optional_params>(arg.get_program());
        fc_optional_params.allowInputReordering = true;

        if(arg.get_primitive()->with_activation)
            convert_activation_func_params(arg.get_primitive(), fc_params.activation);

        fc_params.output = fc_params.output.FlattenFeatureAndSpatials();

        const auto primitive = arg.get_primitive();

        if (primitive->weights_quantization_factors.size() > 0)
        {
            fc_params.int8_quantization = true;
            fc_params.weights_quantization_factors.push_back(convert_data_tensor(arg.weights_quantization_factors().get_output_layout()).FlattenFeatureAndSpatials());
            fc_params.input_quantization_factor = arg.get_input_qf();

            if (primitive->output_calibration_factors.size() > 0)
            {
                fc_params.output_calibration = true;
                fc_params.output_calibration_factors.push_back(convert_data_tensor(arg.output_calibration_factors().get_output_layout()).FlattenFeatureAndSpatials());
            }
            else
                fc_params.output_quantization_factor = arg.get_output_qf();
        }

        fc_optional_params.tuningParams.runner = std::make_shared<gpu::kernel_runner>(arg.get_program().get_engine(), true);

        auto& kernel_selector = kernel_selector::fully_connected_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(fc_params, fc_optional_params);

        CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");

        const auto& new_fc_params = *static_cast<kernel_selector::fully_connected_params*>(best_kernels[0].params.get());
        std::vector<network_impl::ptr> reorders; 
        if (fc_params.inputs[0].GetLayout() != new_fc_params.inputs[0].GetLayout())
        {
            const auto& input_layout = arg.input().get_output_layout();
            topology_impl tpl;
            tpl.add(std::make_shared<cldnn::input_layout>("input", input_layout));
            tpl.add(std::make_shared<cldnn::reorder>("reorder", "input", from_data_layout(new_fc_params.inputs[0].GetLayout()), input_layout.data_type));
            reorders.push_back(arg.get_program().get_engine().build_network(tpl, cldnn::build_options(), true));
        }

        auto fc = new fully_connected_gpu(arg, best_kernels[0], reorders);

        return fc;
    };
};


namespace {
    struct attach {
        attach() {
            auto val_fw = fully_connected_gpu::create;

            implementation_map<fully_connected>::add({
                { std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb), val_fw },
                { std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb), val_fw },
                { std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw },
                { std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw },
                { std::make_tuple(engine_types::ocl, data_types::f32, format::byxf), val_fw },
                { std::make_tuple(engine_types::ocl, data_types::f16, format::byxf), val_fw },
                { std::make_tuple(engine_types::ocl, data_types::i8,  format::bfyx), val_fw },
                // MMAD
                { std::make_tuple(engine_types::ocl, data_types::i8,  format::byxf_af32), val_fw },
                { std::make_tuple(engine_types::ocl, data_types::i8,  format::fs_bs_yx_bsv4_fsv32), val_fw },
                // IMAD
                { std::make_tuple(engine_types::ocl, data_types::i8,  format::b_fs_yx_fsv4), val_fw },
                { std::make_tuple(engine_types::ocl, data_types::u8,  format::b_fs_yx_fsv4), val_fw },
            });
        }
        ~attach() {}
    };
    attach attach_impl;
}
} }
