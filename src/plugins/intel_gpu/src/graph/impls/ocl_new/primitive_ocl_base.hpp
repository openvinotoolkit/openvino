// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/graph/serialization/helpers.hpp"
#include "intel_gpu/graph/serialization/set_serializer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/graph/serialization/vector_serializer.hpp"
#include "intel_gpu/graph/program.hpp"

#include "primitive_inst.h"
#include "utils/kernel_base.hpp"


#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <vector>
#include <utility>

namespace ov::intel_gpu::ocl {

// Base class for all GPU implementation of specified primitive type.
// For example, all gpu convolution implementations should derive from primitive_impl_ocl<convolution>.
struct PrimitiveImplOCL : public primitive_impl {
    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::PrimitiveImplOCL)

    std::unordered_map<size_t, ov::intel_gpu::ocl::KernelData> _kernels_data;
    std::vector<size_t> _stages_registration_order;
    std::unordered_map<size_t, kernel::ptr> _kernels;

    template<typename StageType, size_t stage_id, typename... Args>
    void add_stage(const kernel_impl_params& params, Args&&... args) {
        static_assert(std::is_base_of<ov::intel_gpu::ocl::KernelGeneratorBase, StageType>::value, "StageType must derive from KernelGeneratorBase");
        auto stage = std::make_unique<StageType>(std::forward<Args>(args)...);
        _kernels_data.emplace(stage_id, stage->get_kernel_data(params));
        _stages_registration_order.push_back(stage_id);
    }

    PrimitiveImplOCL() = default;

    PrimitiveImplOCL(const PrimitiveImplOCL& other)
    : primitive_impl(other._weights_reorder_params, other._kernel_name, other._is_dynamic)
    , _kernels_data(other._kernels_data)
    , _stages_registration_order(other._stages_registration_order)
    , _kernels({}) {
        _kernels.reserve(other._kernels.size());
        for (const auto& [id, kernel] : other._kernels) {
            _kernels.emplace(id, kernel->clone(other.can_share_kernels));
        }
        this->m_manager = other.m_manager;
    }

    PrimitiveImplOCL(const std::string& impl_name) : primitive_impl(nullptr, impl_name) {}

    bool is_cpu() const override { return false; }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_impl::save(ob);
        ob << _stages_registration_order;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_impl::load(ib);
        ib >> _stages_registration_order;
    }

    void update(primitive_inst& inst, const kernel_impl_params& impl_params) override {
        update_dispatch_data(impl_params);
        inst.update_shape_info_tensor(impl_params);
    }

protected:
    virtual kernel_arguments_data get_arguments(const primitive_inst& instance, size_t stage) const {
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

        auto intermediates = instance.get_intermediates_memories();
        args.intermediates = {intermediates.begin(), intermediates.end()};

        return args;
    }

    void init_kernels(const kernels_cache& kernels_cache, const kernel_impl_params& params) override {
        _kernels.clear();
        if (!_kernels_data.empty()) {
            auto compiled_kernels = kernels_cache.get_kernels(params);
            for (size_t i = 0; i < _stages_registration_order.size(); i++) {
                _kernels.emplace(_stages_registration_order[i], compiled_kernels[i]);
            }
        }
    }

    void init_by_cached_kernels(const kernels_cache& kernels_cache, std::vector<std::string>& cached_kernel_ids) override {
        _kernels.clear();
        _kernels.reserve(cached_kernel_ids.size());

        for (size_t i = 0; i < cached_kernel_ids.size(); ++i) {
            _kernels.emplace(_stages_registration_order[i], kernels_cache.get_kernel_from_cached_kernels(cached_kernel_ids[i]));
        }
        this->can_share_kernels = kernels_cache.get_kernels_reuse();
    }

    std::vector<std::string> get_cached_kernel_ids(const kernels_cache& kernels_cache) override {
        std::vector<kernel::ptr> kernels;
        for (size_t i = 0; i < _kernels.size(); i++) {
            kernels.push_back(_kernels[_stages_registration_order[i]]);
        }
        return {kernels_cache.get_cached_kernel_ids(kernels)};
    }

    std::vector<kernel::ptr> get_kernels() const override {
        std::vector<kernel::ptr> kernels;
        std::transform(_kernels.begin(), _kernels.end(), std::back_inserter(kernels), [](const decltype(_kernels)::value_type& e) { return e.second; });
        return kernels;
    }

    void set_arguments(primitive_inst& instance) override { }
    void set_arguments(primitive_inst& instance, kernel_arguments_data& args) override { }

    bool has_stage(size_t stage) const {
        return _kernels.count(stage) > 0;
    }

    event::ptr execute_stage(const std::vector<event::ptr>& events, primitive_inst& instance, size_t stage) {
        stream& stream = instance.get_network().get_stream();
        // If any user of the desc's users is CPU implementation or network's output, set desc as a output event (event won't be nullptr)
        bool needs_completion_event = instance.needs_completion_event();

        auto& params = _kernels_data[stage].params;
        auto args = get_arguments(instance, stage);
        args.scalars = &params.scalars;

        if (instance.get_flag(ExecutionFlags::MEMORY_CHANGED) || instance.get_flag(ExecutionFlags::IMPL_CHANGED)) {
            stream.set_arguments(*_kernels[stage], params, args);
            // TODO: Can we call update dispatch data here for required kernels only?
            // Seems that it may conflict with fake alignment as get_impl_params stores not fake aligned data
            _kernels_data[stage].update_dispatch_data_func(*instance.get_impl_params(), _kernels_data[stage]);
        }

        const auto& gws = params.workGroups.global;
        const auto& lws = params.workGroups.local;

        GPU_DEBUG_TRACE_DETAIL << "Enqueue stage " << stage << " kernel: gws=[" << gws[0] << ", " << gws[1] << ", " << gws[2] << "] "
                               << "lws=[" << lws[0] << ", " << lws[1] << ", " << lws[2] << "]"
                               << (needs_completion_event ? " has_completion_event=true" : "") << std::endl;

        return stream.enqueue_kernel(*_kernels[stage], params, args, events, needs_completion_event);
    }

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        stream& stream = instance.get_network().get_stream();
        if (instance.can_be_optimized()) {
            return stream.aggregate_events(events, false, instance.is_output());
        }

        std::vector<event::ptr> tmp_events(events);
        // Default impl just runs each stage in registration order
        for (const auto& stage : _stages_registration_order) {
            tmp_events = { execute_stage(tmp_events, instance, stage) };
        }

        return tmp_events[0];
    }

    std::vector<std::shared_ptr<cldnn::kernel_string>> get_kernels_source() override {
        std::vector<std::shared_ptr<cldnn::kernel_string>> kernel_strings;
        for (size_t i = 0; i < _kernels_data.size(); ++i) {
            kernel_strings.push_back(_kernels_data[_stages_registration_order[i]].code.kernel_string);
        }
        return kernel_strings;
    }

    void reset_kernels_source() override {
        for (auto& [stage, kd] : _kernels_data) {
            kd.code.kernel_string.reset();
        }
    }

    void set_kernels(cldnn::kernels_cache::compiled_kernels kernels) override {
        OPENVINO_ASSERT(kernels.size() == 1, "Only the kernels of the single primitive should be allowed.");
        auto& kernel_vec = kernels.begin()->second;
        _kernels.clear();
        for (auto& [kernel, sub_kernel_idx] : kernel_vec) {
            _kernels[sub_kernel_idx] = kernel;
        }
    }

    std::pair<std::string, std::string> get_kernels_dump_info() const override { return {}; }
    virtual void update_dispatch_data(const kernel_impl_params& impl_params) { }
};

}  // namespace ov::intel_gpu::ocl
