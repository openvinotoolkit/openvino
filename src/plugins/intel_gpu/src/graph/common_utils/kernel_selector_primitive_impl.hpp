// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common_utils/gpu_execution_plan.hpp"
#include "common_utils/gpu_kernel_lifecycle.hpp"
#include "intel_gpu/graph/serialization/cl_kernel_data_serializer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/graph/serialization/vector_serializer.hpp"
#include "primitive_inst.h"

namespace cldnn {

/// Common implementation lifecycle for GPU primitives produced by kernel_selector.
///
/// The class owns backend-neutral kernel metadata, compiled kernel handles, and the standard
/// argument-binding lifecycle. The selected stream remains responsible for backend execution.
template <class Primitive>
class typed_primitive_impl_kernel_selector : public typed_primitive_impl<Primitive> {
public:
    using parent = typed_primitive_impl<Primitive>;
    using instance_type = typed_primitive_inst<Primitive>;

    typed_primitive_impl_kernel_selector() = default;

    explicit typed_primitive_impl_kernel_selector(std::string kernel_name, bool is_dynamic = false) : parent(std::move(kernel_name), is_dynamic) {}

    typed_primitive_impl_kernel_selector(kernel_selector::KernelData kernel_data, bool is_dynamic = false)
        : typed_primitive_impl_kernel_selector(nullptr, std::move(kernel_data), is_dynamic) {}

    typed_primitive_impl_kernel_selector(std::shared_ptr<WeightsReorderParams> weights_reorder_params,
                                         kernel_selector::KernelData kernel_data,
                                         bool is_dynamic = false)
        : parent(std::move(weights_reorder_params), kernel_data.kernelName, is_dynamic),
          _kernel_data(std::move(kernel_data)) {
        this->can_reuse_memory = _kernel_data.can_reuse_memory;
        rebuild_execution_plan();
    }

    typed_primitive_impl_kernel_selector(const typed_primitive_impl_kernel_selector& other)
        : parent(other._weights_reorder_params, other._kernel_name, other._is_dynamic),
          _kernel_data(other._kernel_data),
          _execution_plan(other._execution_plan) {
        _kernels.clone_from(other._kernels, other.can_share_kernels);
        this->can_reuse_memory = _kernel_data.can_reuse_memory;
        this->can_share_kernels = other.can_share_kernels;
        this->m_manager = other.m_manager;
        rebuild_execution_plan();
    }

    bool is_cpu() const final {
        return false;
    }

    void save(BinaryOutputBuffer& buffer) const override {
        parent::save(buffer);
        buffer << make_data(&_kernel_data.internalBufferDataType, sizeof(kernel_selector::Datatype));
        buffer << _kernel_data.internalBuffers;
        buffer << _kernel_data.kernels;
        buffer << _kernel_data.kernelName;
    }

    void load(BinaryInputBuffer& buffer) override {
        parent::load(buffer);
        buffer >> make_data(&_kernel_data.internalBufferDataType, sizeof(kernel_selector::Datatype));
        buffer >> _kernel_data.internalBuffers;
        buffer >> _kernel_data.kernels;
        buffer >> _kernel_data.kernelName;
        rebuild_execution_plan();
    }

    void update(primitive_inst& instance, const kernel_impl_params& impl_params) override {
        auto canonical_params = this->canonicalize_shapes(impl_params);
        update_dispatch_data(canonical_params);
        rebuild_execution_plan();
        instance.update_shape_info_tensor(canonical_params);
    }

protected:
    virtual kernel_arguments_data get_arguments(const instance_type& instance) const {
        kernel_arguments_data arguments;

        for (size_t index = 0; index < instance.inputs_memory_count(); ++index) {
            arguments.inputs.push_back(instance.input_memory_ptr(index));
        }
        for (size_t index = 0; index < instance.get_fused_mem_count(); ++index) {
            arguments.fused_op_inputs.push_back(instance.fused_memory(index));
        }
        for (size_t index = 0; index < instance.outputs_memory_count(); ++index) {
            arguments.outputs.push_back(instance.output_memory_ptr(index));
        }
        arguments.shape_info = instance.shape_info_memory_ptr();

        return arguments;
    }

    void init_kernels(const kernels_cache& cache, const kernel_impl_params& params) override {
        _kernels.clear();
        if (!_kernel_data.kernels.empty()) {
            this->can_share_kernels = _kernels.initialize(cache, params);
        }
    }

    void init_by_cached_kernels(const kernels_cache& cache, std::vector<std::string>& cached_kernel_ids) override {
        this->can_share_kernels = _kernels.restore(cache, cached_kernel_ids);
    }

    std::vector<std::string> get_cached_kernel_ids(const kernels_cache& cache) override {
        return _kernels.get_cached_kernel_ids(cache);
    }

    std::vector<std::shared_ptr<kernel_string>> get_kernels_source() override {
        std::vector<std::shared_ptr<kernel_string>> sources;
        sources.reserve(_kernel_data.kernels.size());
        for (const auto& kernel_data : _kernel_data.kernels) {
            sources.push_back(kernel_data.code.kernelString);
        }
        return sources;
    }

    void reset_kernels_source() override {
        for (auto& kernel_data : _kernel_data.kernels) {
            kernel_data.code.kernelString.reset();
        }
    }

    std::vector<kernel::ptr> get_kernels() const override {
        return _kernels.copy_kernels();
    }

    void set_kernels(kernels_cache::compiled_kernels kernels) override {
        _kernels.adopt_compiled(std::move(kernels));
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, instance_type& instance) override {
        auto& stream = instance.get_network().get_stream();
        if (instance.can_be_optimized()) {
            return stream.aggregate_events(events, events.size() > 1, instance.is_output());
        }

        OPENVINO_ASSERT(_kernels.size() == _kernel_data.kernels.size(), "[GPU] Mismatch between compiled kernels and kernel-selector dispatches");
        return _execution_plan.execute(stream, _kernels, events, instance.needs_completion_event(), [&](size_t dispatch_index) {
            auto& descriptor = _kernel_data.kernels[dispatch_index].params;
            auto arguments = get_arguments(instance);
            arguments.scalars = &descriptor.scalars;
            arguments.local_memory_args = &descriptor.local_memory_args;
            for (const auto& memory : instance.get_intermediates_memories()) {
                arguments.intermediates.push_back(memory);
            }
            return gpu_dispatch_binding{&descriptor, std::move(arguments)};
        });
    }

    virtual void update_dispatch_data(const kernel_impl_params&) {
        OPENVINO_ASSERT(this->_is_dynamic, "[GPU] update_dispatch_data() is called for static implementation ", this->_kernel_name);
        OPENVINO_ASSERT(false, "[GPU] Dynamic dispatch update is not implemented for ", this->_kernel_name);
    }

    void rebuild_execution_plan() {
        _execution_plan.resize(_kernel_data.kernels.size());
        _execution_plan.set_completion_policy({true, true});
        for (size_t index = 0; index < _kernel_data.kernels.size(); ++index) {
            auto& dispatch = _execution_plan[index];
            dispatch.dependency = _kernel_data.needs_sub_kernels_sync ? gpu_dispatch_dependency_policy::previous : gpu_dispatch_dependency_policy::external;
            dispatch.skip_execution = _kernel_data.kernels[index].skip_execution;
        }
    }

    kernel_selector::KernelData _kernel_data;
    gpu_kernel_lifecycle _kernels;
    gpu_execution_plan _execution_plan;
};

}  // namespace cldnn
