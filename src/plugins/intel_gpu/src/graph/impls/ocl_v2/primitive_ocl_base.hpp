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
    explicit Stage(std::shared_ptr<KernelGeneratorBase>&& ptr) : codegen(std::move(ptr)) {}
    std::shared_ptr<KernelGeneratorBase> codegen;
    KernelData kd{};
    Kernel::ptr kernel = nullptr;
};

// Base class for all GPU implementation of specified primitive type.
// For example, all gpu convolution implementations should derive from primitive_impl_ocl<convolution>.
// Derived classes are supposed to define Stage-s as members of the class
// and ensure that default constructor initializes all of them to automatically set-up update_dispatch_data_func
// for each stage on primitive load from the cache.
// Stage registration doesn't mean that it's actually compiled and exectued - it's mainly a holder for stage-related objects.
// In order to activate a stage, add_stage method must be called explicitly for each stage required to execute a particular set
// of parameters. Note: if execute() method is not overriden, then stages are executed in the order of activation (not order of construction)
struct PrimitiveImplOCL : public cldnn::primitive_impl {
    std::vector<Stage*> _stages;
    std::vector<size_t> _order;
    std::unique_ptr<ImplRuntimeParams> m_rt_params = nullptr;

    template <typename CodeGenType, typename... Args>
    Stage::Ptr make_stage(Args&&... args) {
        auto stage = std::make_unique<Stage>(std::make_shared<CodeGenType>(std::forward<Args>(args)...));
        _stages.push_back(stage.get());
        return stage;
    }

    void add_stage(Stage::Ptr& stage, const RuntimeParams& params) {
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
            if (impl->_stages[i]->kernel) {
                copy->_stages[i]->kernel = impl->_stages[i]->kernel->clone();
            }
        }

        return copy;
    }

    PrimitiveImplOCL() = default;
    PrimitiveImplOCL(PrimitiveImplOCL&&) = delete;
    PrimitiveImplOCL& operator=(PrimitiveImplOCL&&) = delete;

    // Copy should be done by make_deep_copy method
    PrimitiveImplOCL(const PrimitiveImplOCL& other) = delete;
    PrimitiveImplOCL& operator=(const PrimitiveImplOCL&) = delete;

    explicit PrimitiveImplOCL(const std::string& impl_name) : primitive_impl(nullptr, impl_name) {}
    ~PrimitiveImplOCL() = default;

    [[nodiscard]] bool is_cpu() const override {
        return false;
    }

    void save(cldnn::BinaryOutputBuffer& ob) const override {
        primitive_impl::save(ob);
        ob << _order;
        for (const auto& i : _order) {
            ob << _stages[i]->kd;
        }
    }

    void load(cldnn::BinaryInputBuffer& ib) override {
        primitive_impl::load(ib);
        ib >> _order;
        for (const auto& i : _order) {
            ib >> _stages[i]->kd;
            _stages[i]->kd.update_dispatch_data_func = _stages[i]->codegen->get_dispatch_data_func();
        }
    }

    void update(cldnn::primitive_inst& inst, const RuntimeParams& impl_params) override {
        inst.update_shape_info_tensor(impl_params);
    }

    std::vector<cldnn::BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        return {};
    }

    void init_kernels(const cldnn::kernels_cache& kernels_cache, const RuntimeParams& params) override {
        auto compiled_kernels = kernels_cache.get_kernels(params);
        for (size_t i = 0; i < _order.size(); i++) {
            _stages[_order[i]]->kernel = compiled_kernels[i];
        }
    }

    void init_by_cached_kernels(const cldnn::kernels_cache& kernels_cache, std::vector<std::string>& cached_kernel_ids) override {
        OPENVINO_ASSERT(cached_kernel_ids.size() == _order.size());
        for (size_t i = 0; i < cached_kernel_ids.size(); ++i) {
            _stages[_order[i]]->kernel = kernels_cache.get_kernel_from_cached_kernels(cached_kernel_ids[i]);
        }
        this->can_share_kernels = kernels_cache.get_kernels_reuse();
    }

    std::vector<std::string> get_cached_kernel_ids(const cldnn::kernels_cache& kernels_cache) override {
        std::vector<Kernel::ptr> kernels;
        for (const auto& i : _order) {
            kernels.push_back(_stages[i]->kernel);
            OPENVINO_ASSERT(kernels.back() != nullptr);
        }
        return {kernels_cache.get_cached_kernel_ids(kernels)};
    }

    std::vector<Kernel::ptr> get_kernels() const override {
        std::vector<Kernel::ptr> kernels;
        for (size_t i = 0; i < _order.size(); i++) {
            kernels.push_back(_stages[_order[i]]->kernel);
        }
        return kernels;
    }

    [[nodiscard]] virtual cldnn::kernel_arguments_data get_arguments(const cldnn::primitive_inst& instance) const {
        cldnn::kernel_arguments_data args;

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

    void set_arguments(cldnn::primitive_inst& instance) override {}
    void set_arguments(cldnn::primitive_inst& instance, cldnn::kernel_arguments_data& args) override {}

    [[nodiscard]] bool has_stage(const Stage::Ptr& stage) const {
        for (size_t i = 0; i < _stages.size(); i++) {
            if (_stages[i] == stage.get()) {
                return std::find(_order.begin(), _order.end(), i) != _order.end();
            }
        }

        return false;
    }

    void update_stages_flags(const cldnn::primitive_inst& instance) {
        const auto current_flags = instance.get_impl_params()->flags.to_ulong();
        constexpr size_t mask_args = (1 << cldnn::ExecutionFlags::MEMORY_CHANGED) | (1 << cldnn::ExecutionFlags::IMPL_CHANGED);
        constexpr size_t mask_dispatch = (1 << cldnn::ExecutionFlags::SHAPE_CHANGED);
        for (auto& stage : _stages) {
            stage->kd.need_args_update = (current_flags & mask_args) != 0;
            stage->kd.need_dispatch_data_update = (current_flags & mask_dispatch) != 0;
        }
    }

    virtual void update_rt_params(const cldnn::primitive_inst& instance) {
        update_stages_flags(instance);
    }

    cldnn::event::ptr execute_stage(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance, Stage::Ptr& stage) const {
        return execute_stage(events, instance, *stage);
    }

    cldnn::event::ptr execute_stage(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance, Stage& stage) const {
        cldnn::stream& stream = instance.get_network().get_stream();
        // If any user of the desc's users is CPU implementation or network's output, set desc as a output event (event
        // won't be nullptr)
        bool needs_completion_event = instance.needs_completion_event();

        auto& kd = stage.kd;
        auto& params = kd.params;

        if (kd.need_dispatch_data_update) {
            kd.update_dispatch_data_func(*instance.get_impl_params(), kd, m_rt_params.get());
            kd.need_dispatch_data_update = false;
        }

        if (kd.need_args_update) {
            auto args = get_arguments(instance);
            args.scalars = &params.scalars;
            stream.set_arguments(*stage.kernel, params, args);
            kd.need_args_update = false;
        }

        const auto& gws = params.workGroups.global;
        const auto& lws = params.workGroups.local;

        GPU_DEBUG_TRACE_DETAIL << "Enqueue stage " << stage.kernel->get_id() << " : gws=[" << gws[0] << ", " << gws[1] << ", " << gws[2] << "] "
                               << "lws=[" << lws[0] << ", " << lws[1] << ", " << lws[2] << "]" << (needs_completion_event ? " has_completion_event=true" : "")
                               << '\n';

        return stream.enqueue_kernel(*stage.kernel, params, {}, events, needs_completion_event);
    }

    cldnn::event::ptr execute(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance) override {
        cldnn::stream& stream = instance.get_network().get_stream();
        if (instance.can_be_optimized()) {
            return stream.aggregate_events(events, false, instance.is_output());
        }

        update_rt_params(instance);

        if (_order.size() == 1) {
            return execute_stage(events, instance, *_stages[_order[0]]);
        }

        std::vector<cldnn::event::ptr> tmp_events(events);
        // Default impl just runs each stage in registration order
        for (const auto& stage_id : _order) {
            tmp_events = {execute_stage(tmp_events, instance, *_stages[stage_id])};
        }

        return tmp_events[0];
    }

    std::vector<std::shared_ptr<cldnn::kernel_string>> get_kernels_source() override {
        std::vector<std::shared_ptr<cldnn::kernel_string>> kernel_strings;
        for (const auto& i : _order) {
            kernel_strings.push_back(_stages[i]->kd.code);
            OPENVINO_ASSERT(kernel_strings.back() != nullptr);
        }
        return kernel_strings;
    }

    void reset_kernels_source() override {
        for (auto& stage : _stages) {
            stage->kd.code.reset();
            stage->codegen.reset();
        }
    }

    void set_kernels(cldnn::kernels_cache::compiled_kernels kernels) override {
        OPENVINO_ASSERT(kernels.size() == 1, "Only the kernels of the single primitive should be allowed.");
        auto& kernel_vec = kernels.begin()->second;
        for (auto& [kernel, sub_kernel_idx] : kernel_vec) {
            _stages[sub_kernel_idx]->kernel = kernel;
        }
    }

    [[nodiscard]] std::pair<std::string, std::string> get_kernels_dump_info() const override {
        return {};
    }
};

}  // namespace ov::intel_gpu::ocl
