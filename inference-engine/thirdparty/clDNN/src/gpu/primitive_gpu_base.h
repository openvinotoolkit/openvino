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
#pragma once

#include <thread>
#include "primitive_inst.h"
#include "program_impl.h"
#include "kernel.h"
#include "events_waiter.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "network_impl.h"
#include "register_gpu.hpp"
#include <vector>
#include <list>
#include <utility>

namespace cldnn {
namespace gpu {

// checks if any user in a list is a cpu primitive
bool is_any_user_cpu(const std::list<const program_node*>& users);

/*
Base class for all GPU implementation of specified primitive type.
For example, all gpu convolution implementations should derive from typed_primitive_gpu_impl<convolution>.
*/
template <class PType>
struct typed_primitive_gpu_impl : public typed_primitive_impl<PType> {
    const typed_program_node<PType>& _outer;
    engine_info_internal _engine_info;
    kernel_selector::kernel_data _kernel_data;
    std::vector<gpu::kernel> _kernels;
    std::vector<memory_impl::cptr> _intermediates_memory;

    typed_primitive_gpu_impl(const typed_program_node<PType>& arg, const kernel_selector::kernel_data& kd)
        : typed_primitive_impl<PType>(kd.weightsReorderParams, kd.kernelName),
          _outer(arg),
          _engine_info(arg.get_program().get_engine().get_context()->get_engine_info()),
          _kernel_data(kd) {
        _kernels.reserve(kd.kernels.size());
        for (size_t i = 0; i < kd.kernels.size(); ++i) {
            gpu::kernel kernel(_outer.get_program().get_engine().get_context(), kd.kernels[i].kernelString);
            _kernels.emplace_back(std::move(kernel));
        }

        for (auto size : kd.internalBufferSizes) {
            auto dtype = from_data_type(kd.intenralBufferDataType);
            const auto bpp = data_type_traits::size_of(dtype);
            layout expected_layout = {dtype,
                                      format::bfyx,  // simple linear format (flatten to x channel)
                                      {1, 1, 1, (tensor::value_type)(size / bpp)}};

            auto& eimpl = arg.get_program().get_engine();
            _intermediates_memory.push_back(eimpl.allocate_memory(expected_layout, 0));
        }
    }
    bool is_cpu() const override { return false; }

protected:
    virtual bool optimized_out(typed_primitive_inst<PType>&) const { return false; }

    virtual kernel::kernel_arguments_data get_arguments(typed_primitive_inst<PType>& instance,
                                                        int32_t /*split*/) const {
        kernel::kernel_arguments_data args;

        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            args.inputs.push_back((memory_impl::cptr)&instance.input_memory(i));
        }

        if (instance.has_fused_primitives()) {
            size_t count = instance.get_fused_mem_count();
            for (size_t i = 0; i < count; i++) {
                args.fused_op_inputs.push_back((memory_impl::cptr) &instance.fused_memory(i));
            }
        }

        args.output = (memory_impl::cptr) &instance.output_memory();

        return args;
    }

    virtual int32_t get_split() const { return 1; }

    virtual uint32_t get_groups() const { return 1; }

    event_impl::ptr aggregate_events(const std::vector<event_impl::ptr>& events,
                                     uint16_t stream_id,
                                     bool group = false) const {
        if (events.size() == 1)
            return events[0];

        if (group)
            return _outer.get_program().get_engine().get_context()->group_events(stream_id, events);

        return events_waiter(_outer.get_program().get_engine().get_context()).run(stream_id, events);
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events,
                                         typed_primitive_inst<PType>& instance) override {
        uint16_t stream_id = instance.get_network().get_stream_id();
        if (optimized_out(instance)) {
            return aggregate_events(events, stream_id);
        }

        std::vector<event_impl::ptr> tmp_events(events);

        // TODO - split should be handle in kernel selector by providing multiple kernels.
        auto split = get_split();
        auto groups = get_groups();
        if (split == 1)
            split = groups;

        // we iterate over split first in order to be able parallelism with OOOQ mechanism.
        for (size_t k = 0; k < _kernels.size(); ++k) {
            std::vector<event_impl::ptr> new_events;
            for (decltype(split) i = 0; i < split; i++) {
                auto args = get_arguments(instance, i);
                args.scalars = &_kernel_data.kernels[k].scalars;
                args.split = i;

                for (const auto& m : _intermediates_memory) {
                    args.intermediates.push_back(m);
                }

                // is any user of the prim's users is an detecion output, set prim as a output event (event won't be
                // nullptr)
                auto users = instance.node.get_users();
                bool next_prim_is_cpu = is_any_user_cpu(users);
                if (next_prim_is_cpu) {
                    _kernels[k].set_output_event(stream_id, true);
                } else {
                    _kernels[k].set_output_event(stream_id, instance.node.is_output());
                }

                auto event = _kernels[k].run(stream_id, _kernel_data.kernels[k], tmp_events, args);
                new_events.push_back(event);
            }

            tmp_events = new_events;
        }

        bool group_events = split > 1 ? true : false;
        return aggregate_events(tmp_events, stream_id, group_events);
    }
};

}  // namespace gpu
}  // namespace cldnn
