// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/graph/serialization/cl_kernel_data_serializer.hpp"
#include "intel_gpu/graph/serialization/helpers.hpp"
#include "intel_gpu/graph/serialization/set_serializer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/graph/serialization/vector_serializer.hpp"
#include "intel_gpu/graph/program.hpp"

#include "primitive_inst.h"
#include "kernel_selector_helper.h"
#include "register.hpp"
#include "implementation_map.hpp"

#include <vector>
#include <list>
#include <utility>

namespace cldnn {
namespace ocl {

/*
Base class for all GPU implementation of specified primitive type.
For example, all gpu convolution implementations should derive from typed_primitive_impl_ocl<convolution>.
*/
template <class PType>
struct typed_primitive_impl_ocl : public typed_primitive_impl<PType> {
    kernel_selector::kernel_data _kernel_data;
    std::vector<kernel_id> _kernel_ids;
    std::vector<kernel::ptr> _kernels;

    typed_primitive_impl_ocl() :  _kernel_data({}), _kernel_ids({}), _kernels({}) {
        _kernel_data.weightsReorderParams.engine = kernel_selector::generic_kernel_params::Engine::NONE;
        _kernel_data.weightsReorderParams.cpuKernel = nullptr;
        _kernel_data.weightsReorderParams.clKernel = nullptr;
    }

    typed_primitive_impl_ocl(const typed_primitive_impl_ocl<PType>& other)
    : typed_primitive_impl<PType>(other._weights_reorder_params, other._kernel_name, other._is_dynamic)
    , _kernel_data(other._kernel_data)
    , _kernel_ids(other._kernel_ids)
    , _kernels({}) {
        _kernels.reserve(other._kernels.size());
        for (size_t k = 0; k < other._kernels.size(); ++k) {
            _kernels.emplace_back(other._kernels[k]->clone());
        }
        this->can_reuse_memory = _kernel_data.can_reuse_memory;
    }

    typed_primitive_impl_ocl(const kernel_selector::kernel_data& kd)
        : typed_primitive_impl<PType>(kd.weightsReorderParams, kd.kernelName),
          _kernel_data(kd) {
        // weights reorder params got copied to parent, clear in _kernel_data to release shared ptr
        _kernel_data.weightsReorderParams.engine = kernel_selector::generic_kernel_params::Engine::NONE;
        _kernel_data.weightsReorderParams.cpuKernel = nullptr;
        _kernel_data.weightsReorderParams.clKernel = nullptr;

        this->can_reuse_memory = _kernel_data.can_reuse_memory;
    }

    bool is_cpu() const override { return false; }

    // Cache blob format:
    //     [ kernel_selector::kernel_data ]
    //     [ kernel_id ]
    //     [ kernel_arguments ]
    void save(BinaryOutputBuffer& ob) const override {
        ob << make_data(&_kernel_data.internalBufferDataType, sizeof(kernel_selector::Datatype));
        ob << _kernel_data.internalBufferSizes;
        ob << _kernel_data.kernels;
        ob << _kernel_ids;
    }

    void load(BinaryInputBuffer& ib) override {
        ib >> make_data(&_kernel_data.internalBufferDataType, sizeof(kernel_selector::Datatype));
        ib >> _kernel_data.internalBufferSizes;
        ib >> _kernel_data.kernels;
        ib >> _kernel_ids;
    }

    template<typename ImplType>
    static std::unique_ptr<primitive_impl> create(const typed_program_node<PType>& arg, const kernel_impl_params& impl_param) {
        if (arg.can_be_optimized()) {
            return make_unique<ImplType>(kernel_selector::kernel_data{});
        }
        auto kernel_params = ImplType::get_kernel_params(impl_param);
        kernel_params.first.is_shape_agnostic = impl_param.is_dynamic();
        auto& kernel_selector = ImplType::kernel_selector_t::Instance();
        auto best_kernel = kernel_selector.get_best_kernel(kernel_params.first, kernel_params.second);

        return make_unique<ImplType>(best_kernel);
    }

private:
    using primitive_impl::get_arguments;

protected:
    virtual kernel_arguments_data get_arguments(const typed_primitive_inst<PType>& instance) const {
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

        for (size_t i = 0; i < instance.outputs_memory_count(); i++) {
            args.outputs.push_back(instance.output_memory_ptr(i));
        }

        args.shape_info = instance.shape_info_memory_ptr();

        return args;
    }

    event::ptr aggregate_events(const std::vector<event::ptr>& events, stream& stream, bool group = false, bool is_output = false) const {
        if (events.size() == 1 && !is_output)
            return events[0];

        if (group && !is_output)
            return stream.group_events(events);

        return stream.enqueue_marker(events, is_output);
    }

    void init_kernels(const kernels_cache& kernels_cache) override {
        if (is_cpu()) {
            return;
        }
        _kernels.clear();

        _kernels.reserve(_kernel_ids.size());
        for (size_t k = 0; k < _kernel_ids.size(); ++k) {
            _kernels.emplace_back(kernels_cache.get_kernel(_kernel_ids[k]));
        }
    }

    std::vector<std::string> get_kernel_ids() const override {
        return _kernel_ids;
    }

    std::vector<kernel::ptr> get_kernels() const override {
        return _kernels;
    }

    std::vector<layout> get_internal_buffer_layouts_impl() const override {
        if (_kernel_data.internalBufferSizes.empty())
            return {};

        std::vector<layout> layouts;
        auto dtype = from_data_type(_kernel_data.internalBufferDataType);
        const auto bpp = data_type_traits::size_of(dtype);
        for (auto size : _kernel_data.internalBufferSizes) {
            layout inbuf_layout = {dtype, format::bfyx, // simple linear format (flattern to x channel)
                                    {1, 1, 1, (tensor::value_type)(size / bpp)}};
            layouts.push_back(inbuf_layout);
        }
        return layouts;
    }

    void set_arguments_impl(typed_primitive_inst<PType>& instance) override {
        if (instance.can_be_optimized() || is_cpu()) {
            return;
        }

        stream& stream = instance.get_network().get_stream();
        size_t k_idx = 0;
        for (size_t kd_idx = 0; kd_idx < _kernel_data.kernels.size(); ++kd_idx) {
            if (_kernel_data.kernels[kd_idx].skip_execution) {
                continue;
            }

            auto args = get_arguments(instance);
            args.scalars = &_kernel_data.kernels[kd_idx].params.scalars;

            for (const auto& m : instance.get_intermediates_memories()) {
                args.intermediates.push_back(m);
            }

            stream.set_arguments(*_kernels[k_idx++], _kernel_data.kernels[kd_idx].params, args);
        }
    }

    kernel_arguments_data get_arguments_impl(const typed_primitive_inst<PType>& instance) const override {
        for (size_t k = 0; k < _kernels.size(); ++k) {
            auto args = get_arguments(instance);
            args.scalars = &_kernel_data.kernels[k].params.scalars;

            for (const auto& m : instance.get_intermediates_memories()) {
                args.intermediates.push_back(m);
            }

            return args;
        }

        kernel_arguments_data args;
        return args;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events,
                            typed_primitive_inst<PType>& instance) override {
        stream& stream = instance.get_network().get_stream();
        if (instance.can_be_optimized()) {
            return aggregate_events(events, stream, false, instance.is_output());
        }
        std::vector<event::ptr> tmp_events(events);
        std::vector<event::ptr> all_events;
        size_t k_idx = 0;
        for (size_t kd_idx = 0; kd_idx < _kernel_data.kernels.size(); ++kd_idx) {
            if (_kernel_data.kernels[kd_idx].skip_execution)
                continue;
            std::vector<event::ptr> new_events;
            // is any user of the prim's users is an detecion output, set prim as a output event (event won't be nullptr)
            bool is_output_event;
            if (instance.node != nullptr) {
                auto users = instance.node->get_users();
                is_output_event = is_any_user_cpu(users) || instance.node->is_output();
            } else {
                is_output_event = instance.is_output_event();
            }

            auto args = get_arguments(instance);
            args.scalars = &_kernel_data.kernels[kd_idx].params.scalars;

            for (const auto& m : instance.get_intermediates_memories()) {
                args.intermediates.push_back(m);
            }

            auto ev = stream.enqueue_kernel(*_kernels[k_idx++], _kernel_data.kernels[kd_idx].params, args, tmp_events, is_output_event);
            new_events.push_back(ev);
            all_events.push_back(ev);

            tmp_events = new_events;
        }

        if ((all_events.size() == 0) && (tmp_events.size() > 0))
            return aggregate_events(tmp_events, stream);

        bool group_events = (all_events.size() > 1);
        return aggregate_events(all_events, stream, group_events);
    }

    void set_kernel_ids(std::vector<kernel_id> kernel_ids) override {
        _kernel_ids = kernel_ids;
    }

    std::vector<std::shared_ptr<cldnn::kernel_string>> get_kernels_source() override {
        std::vector<std::shared_ptr<cldnn::kernel_string>> kernel_strings;
        for (size_t i = 0; i < _kernel_data.kernels.size(); ++i) {
            if (!_kernel_data.kernels[i].skip_execution)
                kernel_strings.push_back(_kernel_data.kernels[i].code.kernelString);
        }
        return kernel_strings;
    }

    void reset_kernels_source() override {
        for (size_t i = 0; i < _kernel_data.kernels.size(); ++i) {
            _kernel_data.kernels[i].code.kernelString.reset();
        }
    }

    void update_kernels_list_to_skip() {
        for (size_t i = 0; i < _kernel_data.kernels.size(); ++i) {
            auto gws = _kernel_data.kernels[0].params.workGroups.global;
            _kernel_data.kernels[0].skip_execution =
                (std::accumulate(gws.begin(), gws.end(), 1, std::multiplies<size_t>()) == 0);
        }
    }

    void set_kernels(std::map<const std::string, kernel::ptr>& kernels) override {
        if (is_cpu())
            return;

        _kernel_ids.clear();
        _kernels.clear();
        _kernels.reserve(kernels.size());
        for (auto& k : kernels) {
            _kernel_ids.push_back(k.first);
            _kernels.emplace_back(std::move(k.second));
        }
    }
};

}  // namespace ocl
}  // namespace cldnn
