// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "common_utils/kernel_generator_base.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/graph/serialization/helpers.hpp"
#include "intel_gpu/graph/serialization/set_serializer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/graph/serialization/vector_serializer.hpp"
#include "openvino/core/except.hpp"
#include "primitive_inst.h"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {

class Stage {
public:
    using Ptr = std::unique_ptr<Stage>;
    Stage(std::shared_ptr<KernelGeneratorBase>&& ptr) : codegen(std::move(ptr)) {}
    std::shared_ptr<KernelGeneratorBase> codegen;
    KernelData kd{};
    Kernel::ptr kernel = nullptr;
};

// Base class for all GPU implementation of specified primitive type.
// For example, all gpu convolution implementations should derive from primitive_impl_ocl<convolution>.
struct PrimitiveImplOCL : public primitive_impl {
    std::vector<Stage*> _stages;
    std::vector<size_t> _order;
    std::unique_ptr<RuntimeParams> m_rt_params = nullptr;

    template <typename CodeGenType, typename... Args>
    Stage::Ptr make_stage(Args&&... args) {
        auto stage = std::make_unique<Stage>(std::make_shared<CodeGenType>(std::forward<Args>(args)...));
        _stages.push_back(stage.get());
        return stage;
    }

    void add_stage(Stage::Ptr& stage, const kernel_impl_params& params) {
        for (size_t i = 0; i < _stages.size(); i++) {
            if (stage.get() == _stages[i]) {
                _order.push_back(i);
                stage->kd = stage->codegen->get_kernel_data(params);
                break;
            }
        }
    }

    template <typename ImplType>
    static std::unique_ptr<ImplType> make_deep_copy(const ImplType* impl) {
        auto copy = std::make_unique<ImplType>();  // Use default c-tor to initialize stages
        copy->_order = impl->_order;
        copy->m_rt_params = nullptr;  // don't copy RT params
        copy->m_manager = impl->m_manager;
        copy->can_reuse_memory = impl->can_reuse_memory;
        copy->can_share_kernels = impl->can_share_kernels;
        copy->_weights_reorder_params = impl->_weights_reorder_params;
        copy->_kernel_name = impl->_kernel_name;
        copy->_is_dynamic = impl->_is_dynamic;

        for (size_t i = 0; i < copy->_stages.size(); i++) {
            copy->_stages[i]->kd = impl->_stages[i]->kd;
            if (impl->_stages[i]->kernel)
                copy->_stages[i]->kernel = impl->_stages[i]->kernel->clone();
        }

        return copy;
    }

    PrimitiveImplOCL() = default;
    PrimitiveImplOCL(const std::string& impl_name) : primitive_impl(nullptr, impl_name) {}

    bool is_cpu() const override {
        return false;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_impl::save(ob);
        ob << _order;
        for (const auto& i : _order) {
            ob << _stages[i]->kd;
        }
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_impl::load(ib);
        ib >> _order;
        for (const auto& i : _order) {
            ib >> _stages[i]->kd;
            _stages[i]->kd.update_dispatch_data_func = _stages[i]->codegen->get_dispatch_data_func();
        }
    }

    void update(primitive_inst& inst, const kernel_impl_params& impl_params) override {
        inst.update_shape_info_tensor(impl_params);
    }

    std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        return {};
    }

    void init_kernels(const kernels_cache& kernels_cache, const kernel_impl_params& params) override {
        auto compiled_kernels = kernels_cache.get_kernels(params);
        for (size_t i = 0; i < _order.size(); i++) {
            _stages[_order[i]]->kernel = compiled_kernels[i];
        }
    }

    void init_by_cached_kernels(const kernels_cache& kernels_cache, std::vector<std::string>& cached_kernel_ids) override {
        OPENVINO_ASSERT(cached_kernel_ids.size() == _order.size());
        for (size_t i = 0; i < cached_kernel_ids.size(); ++i) {
            _stages[_order[i]]->kernel = kernels_cache.get_kernel_from_cached_kernels(cached_kernel_ids[i]);
        }
        this->can_share_kernels = kernels_cache.get_kernels_reuse();
    }

    std::vector<std::string> get_cached_kernel_ids(const kernels_cache& kernels_cache) override {
        std::vector<kernel::ptr> kernels;
        for (size_t i = 0; i < _order.size(); i++) {
            kernels.push_back(_stages[_order[i]]->kernel);
            OPENVINO_ASSERT(kernels.back() != nullptr);
        }
        return {kernels_cache.get_cached_kernel_ids(kernels)};
    }

    std::vector<kernel::ptr> get_kernels() const override {
        std::vector<kernel::ptr> kernels;
        for (size_t i = 0; i < _order.size(); i++) {
            kernels.push_back(_stages[_order[i]]->kernel);
        }
        return kernels;
    }

    virtual kernel_arguments_data get_arguments(const primitive_inst& instance) const {
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

    void set_arguments(primitive_inst& instance) override {}
    void set_arguments(primitive_inst& instance, kernel_arguments_data& args) override {}

    bool has_stage(const Stage::Ptr& stage) const {
        for (size_t i = 0; i < _stages.size(); i++) {
            if (_stages[i] == stage.get()) {
                return std::find(_order.begin(), _order.end(), i) != _order.end();
            }
        }

        return false;
    }

    void update_stages_flags(const primitive_inst& instance) {
        if (instance.get_flag(ExecutionFlags::MEMORY_CHANGED) || instance.get_flag(ExecutionFlags::IMPL_CHANGED) ||
            instance.get_flag(ExecutionFlags::SHAPE_CHANGED)) {
            for (auto& stage : _stages) {
                stage->kd.need_args_update = true;
            }
        }
    }

    virtual void update_rt_params(const primitive_inst& instance) {
        update_stages_flags(instance);
    }

    event::ptr execute_stage(const std::vector<event::ptr>& events, primitive_inst& instance, Stage::Ptr& stage) {
        return execute_stage(events, instance, *stage);
    }

    event::ptr execute_stage(const std::vector<event::ptr>& events, primitive_inst& instance, Stage& stage) {
        stream& stream = instance.get_network().get_stream();
        // If any user of the desc's users is CPU implementation or network's output, set desc as a output event (event
        // won't be nullptr)
        bool needs_completion_event = instance.needs_completion_event();

        auto& kd = stage.kd;
        auto& params = kd.params;

        if (kd.need_args_update) {
            kd.update_dispatch_data_func(*instance.get_impl_params(), kd, m_rt_params.get());
            auto args = get_arguments(instance);
            args.scalars = &params.scalars;
            stream.set_arguments(*stage.kernel, params, args);
            kd.need_args_update = false;
        }

        const auto& gws = params.workGroups.global;
        const auto& lws = params.workGroups.local;

        GPU_DEBUG_TRACE_DETAIL << "Enqueue stage " << stage.kernel->get_id() << " : gws=[" << gws[0] << ", " << gws[1] << ", " << gws[2] << "] "
                               << "lws=[" << lws[0] << ", " << lws[1] << ", " << lws[2] << "]" << (needs_completion_event ? " has_completion_event=true" : "")
                               << std::endl;

        return stream.enqueue_kernel(*stage.kernel, params, {}, events, needs_completion_event);
    }

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        stream& stream = instance.get_network().get_stream();
        if (instance.can_be_optimized()) {
            return stream.aggregate_events(events, false, instance.is_output());
        }

        update_rt_params(instance);

        std::vector<event::ptr> tmp_events(events);
        // Default impl just runs each stage in registration order
        for (const auto& stage_id : _order) {
            tmp_events = {execute_stage(tmp_events, instance, *_stages[stage_id])};
        }

        return tmp_events[0];
    }

    std::vector<std::shared_ptr<cldnn::kernel_string>> get_kernels_source() override {
        std::vector<std::shared_ptr<cldnn::kernel_string>> kernel_strings;
        for (size_t i = 0; i < _order.size(); ++i) {
            kernel_strings.push_back(_stages[_order[i]]->kd.code.kernel_string);
            OPENVINO_ASSERT(kernel_strings.back() != nullptr);
        }
        return kernel_strings;
    }

    void reset_kernels_source() override {
        for (auto& stage : _stages) {
            stage->kd.code.kernel_string.reset();
        }
    }

    void set_kernels(cldnn::kernels_cache::compiled_kernels kernels) override {
        OPENVINO_ASSERT(kernels.size() == 1, "Only the kernels of the single primitive should be allowed.");
        auto& kernel_vec = kernels.begin()->second;
        for (auto& [kernel, sub_kernel_idx] : kernel_vec) {
            _stages[sub_kernel_idx]->kernel = kernel;
        }
    }

    std::pair<std::string, std::string> get_kernels_dump_info() const override {
        return {};
    }

private:
    // Copy should be done by make_deep_copy method
    PrimitiveImplOCL(const PrimitiveImplOCL& other) = delete;
    PrimitiveImplOCL& operator=(const PrimitiveImplOCL&) = delete;
};

}  // namespace ov::intel_gpu::ocl
