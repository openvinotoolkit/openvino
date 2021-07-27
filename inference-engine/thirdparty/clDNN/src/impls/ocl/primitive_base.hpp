// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <thread>
#include "primitive_inst.h"
#include "program_impl.h"
#include "cldnn/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "network_impl.h"
#include "register.hpp"
#include <vector>
#include <list>
#include <utility>

namespace cldnn {
namespace ocl {

// checks if any user in a list is a cpu primitive
bool is_any_user_cpu(const std::list<const program_node*>& users);

/*
Base class for all GPU implementation of specified primitive type.
For example, all gpu convolution implementations should derive from typed_primitive_impl_ocl<convolution>.
*/
template <class PType>
struct typed_primitive_impl_ocl : public typed_primitive_impl<PType> {
    const typed_program_node<PType>& _outer;
    kernel_selector::kernel_data _kernel_data;
    std::vector<kernel_id> _kernel_ids;
    std::vector<kernel::ptr> _kernels;
    std::vector<memory::cptr> _intermediates_memory;

    typed_primitive_impl_ocl(const typed_primitive_impl_ocl<PType>& other)
    : typed_primitive_impl<PType>(other._weights_reorder_params, other._kernel_name)
    , _outer(other._outer)
    , _kernel_data(other._kernel_data)
    , _kernel_ids(other._kernel_ids)
    , _kernels({})
    , _intermediates_memory({}) {
        _kernels.reserve(other._kernels.size());
        for (size_t k = 0; k < other._kernels.size(); ++k) {
            _kernels.emplace_back(other._kernels[k]->clone());
        }
        for (auto& mem : other._intermediates_memory) {
            auto& engine = _outer.get_program().get_engine();
            auto new_mem = engine.allocate_memory(mem->get_layout(), mem->get_allocation_type());
            _intermediates_memory.push_back(new_mem);
        }
    }

    typed_primitive_impl_ocl(const typed_program_node<PType>& arg, const kernel_selector::kernel_data& kd)
        : typed_primitive_impl<PType>(kd.weightsReorderParams, kd.kernelName),
          _outer(arg),
          _kernel_data(kd) {
        // weights reorder params got copied to parent, clear in _kernel_data to release shared ptr
        _kernel_data.weightsReorderParams.engine = kernel_selector::generic_kernel_params::Engine::NONE;
        _kernel_data.weightsReorderParams.cpuKernel = nullptr;
        _kernel_data.weightsReorderParams.clKernel = nullptr;

        _kernel_ids.reserve(kd.kernels.size());
        // Add selected kernels to kernels_cache for the following compilation and save output ids
        for (size_t i = 0; i < kd.kernels.size(); ++i) {
            _kernel_ids.emplace_back(_outer.get_program().add_kernel(kd.kernels[i].code.kernelString));
        }

        for (auto size : kd.internalBufferSizes) {
            auto dtype = from_data_type(kd.internalBufferDataType);
            const auto bpp = data_type_traits::size_of(dtype);
            layout expected_layout = {dtype,
                                      format::bfyx,  // simple linear format (flatten to x channel)
                                      {1, 1, 1, (tensor::value_type)(size / bpp)}};

            auto& eimpl = arg.get_program().get_engine();
            _intermediates_memory.push_back(eimpl.allocate_memory(expected_layout));
        }
    }
    bool is_cpu() const override { return false; }

protected:
    virtual bool optimized_out(typed_primitive_inst<PType>&) const { return false; }

    virtual kernel_arguments_data get_arguments(typed_primitive_inst<PType>& instance, int32_t /*split*/) const {
        kernel_arguments_data args;

        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            args.inputs.push_back(instance.input_memory_ptr(i));
        }

        if (instance.has_fused_primitives()) {
            size_t count = instance.get_fused_mem_count();
            for (size_t i = 0; i < count; i++) {
                args.fused_op_inputs.push_back(instance.fused_memory(i));
            }
        }

        args.output = instance.output_memory_ptr();

        return args;
    }

    virtual int32_t get_split() const { return 1; }
    virtual uint32_t get_groups() const { return 1; }
    virtual bool get_depthwise_sep_opt() const { return false; }

    event::ptr aggregate_events(const std::vector<event::ptr>& events, stream& stream, bool group = false, bool is_output = false) const {
        if (events.size() == 1 && !is_output)
            return events[0];

        if (group && !is_output)
            return stream.group_events(events);

        return stream.enqueue_marker(events, is_output);
    }

    void init_kernels() override {
        if (is_cpu()) {
            return;
        }
        _kernels.clear();

        _kernels.reserve(_kernel_ids.size());
        for (size_t k = 0; k < _kernel_ids.size(); ++k) {
            _kernels.emplace_back(std::move(_outer.get_program().get_kernel(_kernel_ids[k])));
        }
    }

    void set_arguments_impl(typed_primitive_inst<PType>& instance) override {
        if (optimized_out(instance) || is_cpu()) {
            return;
        }

        auto split = get_split();

        stream& stream = instance.get_network().get_stream();

        // we iterate over split first in order to be able parallelism with OOOQ mechanism.
        for (size_t k = 0; k < _kernels.size(); ++k) {
            for (decltype(split) i = 0; i < split; i++) {
                auto args = get_arguments(instance, i);
                args.scalars = &_kernel_data.kernels[k].params.scalars;
                args.split = i;

                for (const auto& m : _intermediates_memory) {
                    args.intermediates.push_back(m);
                }


                stream.set_arguments(*_kernels[k], _kernel_data.kernels[k].params, args);
            }
        }
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events,
                            typed_primitive_inst<PType>& instance) override {
        stream& stream = instance.get_network().get_stream();
        if (optimized_out(instance)) {
            return aggregate_events(events, stream, false, instance.is_output());
        }

        std::vector<event::ptr> tmp_events(events);
        std::vector<event::ptr> all_events;

        // TODO - split should be handle in kernel selector by providing multiple kernels.
        auto split = get_split();

        // we iterate over split first in order to be able parallelism with OOOQ mechanism.
        for (size_t k = 0; k < _kernels.size(); ++k) {
            std::vector<event::ptr> new_events;
            for (decltype(split) i = 0; i < split; i++) {
                // is any user of the prim's users is an detecion output, set prim as a output event (event won't be nullptr)
                auto users = instance.node.get_users();
                bool is_output_event = is_any_user_cpu(users) || instance.node.is_output();

                auto args = get_arguments(instance, i);
                args.scalars = &_kernel_data.kernels[k].params.scalars;
                args.split = i;

                for (const auto& m : _intermediates_memory) {
                    args.intermediates.push_back(m);
                }

                auto ev = stream.enqueue_kernel(*_kernels[k], _kernel_data.kernels[k].params, args, tmp_events, is_output_event);
                new_events.push_back(ev);
                all_events.push_back(ev);
            }

            tmp_events = new_events;
        }

        if ((all_events.size() == 0) && (tmp_events.size() > 0))
            return aggregate_events(tmp_events, stream);

        bool group_events = (all_events.size() > 1);
        return aggregate_events(all_events, stream, group_events);
    }
};

}  // namespace ocl
}  // namespace cldnn
