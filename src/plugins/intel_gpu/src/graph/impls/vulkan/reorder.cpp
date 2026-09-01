// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "common_utils/gpu_execution_plan.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "openvino/core/except.hpp"
#include "registry/implementation_map.hpp"
#include "reorder_convert_spirv.hpp"
#include "shader_scalar_type.hpp"
#include "vulkan/vulkan_stream.hpp"
#include "vulkan_shader_abi.hpp"

namespace cldnn {
namespace vulkan {
namespace {

enum class metadata_field : size_t {
    element_count,
    input_type,
    output_type,
    count,
};

constexpr size_t metadata_index(metadata_field field) {
    return static_cast<size_t>(field);
}

constexpr size_t metadata_words = metadata_index(metadata_field::count);
constexpr uint32_t portable_max_local_work_group_size = 128;

bool is_copy_compatible_format(format fmt) {
    return fmt == format::any || format::is_default_format(fmt);
}

uint32_t checked_u32(size_t value, const char* description) {
    OPENVINO_ASSERT(value <= std::numeric_limits<uint32_t>::max(), "[GPU][Vulkan] Reorder ", description, " exceeds the 32-bit shader metadata range");
    return static_cast<uint32_t>(value);
}

uint32_t select_local_work_group_size(uint32_t element_count, uint64_t device_max_work_group_size) {
    const auto limit = static_cast<uint32_t>(std::min<uint64_t>(portable_max_local_work_group_size, device_max_work_group_size));
    OPENVINO_ASSERT(limit > 0, "[GPU][Vulkan] Device reports a zero maximum work-group size");

    uint32_t local_size = 1;
    while (local_size < element_count && local_size <= limit / 2) {
        local_size *= 2;
    }
    return local_size;
}

std::array<uint32_t, metadata_words> make_metadata(const reorder_inst& instance) {
    const auto& input_layout = instance.get_input_layout(0);
    const auto& output_layout = instance.get_output_layout(0);
    OPENVINO_ASSERT(!input_layout.is_dynamic() && !output_layout.is_dynamic(), "[GPU][Vulkan] Reorder execution requires resolved runtime layouts");
    OPENVINO_ASSERT(input_layout.count() == output_layout.count(), "[GPU][Vulkan] Reorder conversion requires equal input and output element counts");

    std::array<uint32_t, metadata_words> metadata{};
    metadata[metadata_index(metadata_field::element_count)] = checked_u32(output_layout.count(), "element count");
    metadata[metadata_index(metadata_field::input_type)] = shader_abi::value(to_shader_scalar_type(input_layout.data_type));
    metadata[metadata_index(metadata_field::output_type)] = shader_abi::value(to_shader_scalar_type(output_layout.data_type));
    return metadata;
}

std::shared_ptr<kernel_string> make_convert_kernel_source() {
    auto source = std::make_shared<kernel_string>();
    source->str.assign(reinterpret_cast<const char*>(reorder_convert_spirv), sizeof(reorder_convert_spirv));
    source->entry_point = "main";
    source->batch_compilation = false;
    source->language = kernel_language::SPIRV;
    return source;
}

struct reorder_impl : typed_primitive_impl<reorder> {
    using parent = typed_primitive_impl<reorder>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::vulkan::reorder_impl)

    reorder_impl() : parent("vulkan_reorder") {}

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<reorder_impl>(*this);
    }

    bool is_cpu() const override {
        return false;
    }

    bool requires_lockable_input() const override {
        return false;
    }

    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}

    event::ptr execute_impl(const std::vector<event::ptr>& events, reorder_inst& instance) override {
        auto& stream = instance.get_network().get_stream();
        stream.wait_for_events(events);
        return instance.output_memory_ptr()->copy_from(stream, *instance.input_memory_ptr(), true);
    }
};

struct reorder_convert_impl : typed_primitive_impl<reorder> {
    using parent = typed_primitive_impl<reorder>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::vulkan::reorder_convert_impl)

    reorder_convert_impl() : parent("vulkan_reorder_convert"), _kernel_source(make_convert_kernel_source()) {}

    std::unique_ptr<primitive_impl> clone() const override {
        auto result = std::make_unique<reorder_convert_impl>(*this);
        result->_metadata_initialized = false;
        return result;
    }

    bool is_cpu() const override {
        return false;
    }

    bool requires_lockable_input() const override {
        return false;
    }

    std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params&) const override {
        return {BufferDescriptor(metadata_words, ov::element::u32, true, false)};
    }

    void init_kernels(const kernels_cache& cache, const kernel_impl_params& params) override {
        this->can_share_kernels = _kernels.initialize(cache, params);
        OPENVINO_ASSERT(_kernels.size() == 1, "[GPU][Vulkan] Reorder conversion expects exactly one SPIR-V kernel");
    }

    void init_by_cached_kernels(const kernels_cache& cache, std::vector<std::string>& cached_kernel_ids) override {
        this->can_share_kernels = _kernels.restore(cache, cached_kernel_ids);
    }

    std::vector<std::string> get_cached_kernel_ids(const kernels_cache& cache) override {
        return _kernels.get_cached_kernel_ids(cache);
    }

    std::vector<std::shared_ptr<kernel_string>> get_kernels_source() override {
        return _kernel_source == nullptr ? std::vector<std::shared_ptr<kernel_string>>{} : std::vector<std::shared_ptr<kernel_string>>{_kernel_source};
    }

    void reset_kernels_source() override {
        _kernel_source.reset();
    }

    std::vector<kernel::ptr> get_kernels() const override {
        return _kernels.copy_kernels();
    }

    void set_kernels(kernels_cache::compiled_kernels kernels) override {
        _kernels.adopt_compiled(std::move(kernels));
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, reorder_inst& instance) override {
        OPENVINO_ASSERT(_kernels.size() == 1 && _kernels.front() != nullptr, "[GPU][Vulkan] Reorder conversion kernel was not initialized");
        OPENVINO_ASSERT(instance.get_intermediates_memories().size() == 1, "[GPU][Vulkan] Reorder conversion metadata buffer was not allocated");

        auto& stream = instance.get_network().get_stream();
        const auto metadata = make_metadata(instance);
        auto metadata_memory = instance.get_intermediates_memories().front();
        if (!_metadata_initialized || metadata != _cached_metadata) {
            metadata_memory->copy_from(stream, metadata.data(), true);
            _cached_metadata = metadata;
            _metadata_initialized = true;
        }

        const auto element_count = metadata[metadata_index(metadata_field::element_count)];
        kernel_arguments_desc descriptor;
        descriptor.layerID = instance.id();
        descriptor.workGroups.global = {element_count, 1, 1};
        descriptor.workGroups.local = {select_local_work_group_size(element_count, instance.get_network().get_engine().get_device_info().max_work_group_size),
                                       1,
                                       1};
        const vulkan_specialization_constants specialization_constants = {
            {cldnn::vulkan::shader_abi::index(cldnn::vulkan::shader_abi::specialization_id::local_size_x),
             static_cast<uint32_t>(descriptor.workGroups.local.front())},
        };
        descriptor.arguments = {
            {argument_desc::Types::INPUT, 0},
            {argument_desc::Types::OUTPUT, 0},
            {argument_desc::Types::INTERNAL_BUFFER, 0},
        };

        kernel_arguments_data arguments;
        arguments.inputs = {instance.input_memory_ptr(0)};
        arguments.outputs = {instance.output_memory_ptr(0)};
        arguments.intermediates = {metadata_memory};
        auto& vulkan_dispatch_stream = dynamic_cast<vulkan_stream&>(stream);
        return _execution_plan.execute_with(
            stream,
            _kernels,
            events,
            instance.needs_completion_event(),
            [&](size_t) {
                return gpu_dispatch_binding{&descriptor, std::move(arguments)};
            },
            [&](size_t,
                kernel& selected_kernel,
                const kernel_arguments_desc& kernel_descriptor,
                const kernel_arguments_data& kernel_arguments,
                const std::vector<event::ptr>& dependencies,
                bool request_completion) {
                return vulkan_dispatch_stream
                    .enqueue_kernel(selected_kernel, kernel_descriptor, kernel_arguments, specialization_constants, dependencies, request_completion);
            });
    }

private:
    std::shared_ptr<kernel_string> _kernel_source;
    gpu_kernel_lifecycle _kernels;
    gpu_execution_plan _execution_plan{1};
    std::array<uint32_t, metadata_words> _cached_metadata{};
    bool _metadata_initialized = false;
};

}  // namespace

bool ReorderImplementationManager::validate_impl(const program_node& node) const {
    OPENVINO_ASSERT(node.is_type<reorder>(), "[GPU][Vulkan] Invalid node type passed to Reorder manager");
    if (node.get_program().get_engine().runtime_type() != runtime_types::vulkan) {
        return false;
    }
    const auto& input_layout = node.get_input_layout(0);
    const auto& output_layout = node.get_output_layout(0);
    if (!is_supported_shader_scalar_type(input_layout.data_type) || !is_supported_shader_scalar_type(output_layout.data_type) ||
        !is_copy_compatible_format(input_layout.format) || !is_copy_compatible_format(output_layout.format) || static_cast<bool>(input_layout.data_padding) ||
        static_cast<bool>(output_layout.data_padding)) {
        return false;
    }

    if (input_layout.is_dynamic() || output_layout.is_dynamic()) {
        return input_layout.is_dynamic() && output_layout.is_dynamic() && input_layout.get_partial_shape().same_scheme(output_layout.get_partial_shape());
    }

    if (input_layout.get_linear_offset() != 0 || output_layout.get_linear_offset() != 0 || input_layout.count() != output_layout.count()) {
        return false;
    }

    return input_layout.data_type != output_layout.data_type || input_layout.bytes_count() == output_layout.bytes_count();
}

std::unique_ptr<primitive_impl> ReorderImplementationManager::create_impl(const program_node& node, const kernel_impl_params&) const {
    OPENVINO_ASSERT(node.is_type<reorder>(), "[GPU][Vulkan] Invalid node type passed to Reorder manager");
    if (node.get_input_layout(0).data_type != node.get_output_layout(0).data_type) {
        return std::make_unique<reorder_convert_impl>();
    }
    return std::make_unique<reorder_impl>();
}

}  // namespace vulkan
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::vulkan::reorder_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::vulkan::reorder_convert_impl)
