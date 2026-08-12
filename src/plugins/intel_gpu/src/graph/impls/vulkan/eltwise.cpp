// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "eltwise_spirv.hpp"
#include "impls/ocl/kernels_cache.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "openvino/core/except.hpp"
#include "registry/implementation_map.hpp"

namespace cldnn {
namespace vulkan {
namespace {

constexpr uint32_t max_rank = 8;
constexpr uint32_t header_words = 4;
constexpr uint32_t tensor_words = max_rank * 2 + 1;
constexpr uint32_t tensor_count = 3;
constexpr uint32_t metadata_words = header_words + tensor_count * tensor_words;
constexpr uint32_t local_work_group_size = 64;

bool is_supported_mode(eltwise_mode mode) {
    return one_of(
        mode,
        {eltwise_mode::sum, eltwise_mode::sub, eltwise_mode::max, eltwise_mode::prod, eltwise_mode::div, eltwise_mode::min, eltwise_mode::squared_diff});
}

bool is_supported_format(format::type fmt) {
    return one_of(fmt, {format::any, format::bfyx, format::yxfb, format::bfzyx, format::bfwzyx, format::bfuwzyx, format::bfvuwzyx});
}

uint32_t checked_u32(size_t value, const char* description) {
    OPENVINO_ASSERT(value <= std::numeric_limits<uint32_t>::max(), "[GPU][Vulkan] Eltwise ", description, " exceeds the 32-bit shader metadata range");
    return static_cast<uint32_t>(value);
}

void write_tensor_metadata(std::array<uint32_t, metadata_words>& metadata, uint32_t tensor_index, const layout& tensor_layout, uint32_t output_rank) {
    const auto shape = tensor_layout.get_shape();
    const auto pitches = tensor_layout.get_pitches();
    OPENVINO_ASSERT(shape.size() <= pitches.size() && shape.size() <= output_rank, "[GPU][Vulkan] Eltwise received an invalid tensor rank");

    const auto base = header_words + tensor_index * tensor_words;
    const auto leading_dimensions = output_rank - checked_u32(shape.size(), "rank");
    for (uint32_t axis = 0; axis < max_rank; ++axis) {
        metadata[base + axis] = 1;
        metadata[base + max_rank + axis] = 0;
    }
    for (uint32_t axis = 0; axis < shape.size(); ++axis) {
        const auto output_axis = leading_dimensions + axis;
        metadata[base + output_axis] = checked_u32(shape[axis], "dimension");
        metadata[base + max_rank + output_axis] = checked_u32(pitches[axis], "pitch");
    }
    metadata[base + max_rank * 2] = checked_u32(tensor_layout.get_linear_offset(), "base offset");
}

std::array<uint32_t, metadata_words> make_metadata(const eltwise_inst& instance) {
    const auto& input0_layout = instance.get_input_layout(0);
    const auto& input1_layout = instance.get_input_layout(1);
    const auto& output_layout = instance.get_output_layout(0);
    const auto output_rank = checked_u32(output_layout.get_shape().size(), "output rank");
    OPENVINO_ASSERT(output_rank <= max_rank, "[GPU][Vulkan] Eltwise supports tensors with rank up to ", max_rank);

    std::array<uint32_t, metadata_words> metadata{};
    metadata[0] = checked_u32(output_layout.count(), "element count");
    metadata[1] = static_cast<uint32_t>(instance.get_typed_desc<eltwise>()->mode);
    metadata[2] = output_rank;
    write_tensor_metadata(metadata, 0, input0_layout, output_rank);
    write_tensor_metadata(metadata, 1, input1_layout, output_rank);
    write_tensor_metadata(metadata, 2, output_layout, output_rank);
    return metadata;
}

std::shared_ptr<kernel_string> make_kernel_source() {
    auto source = std::make_shared<kernel_string>();
    source->str.assign(reinterpret_cast<const char*>(eltwise_spirv), sizeof(eltwise_spirv));
    source->entry_point = "main";
    source->batch_compilation = false;
    source->language = kernel_language::SPIRV;
    return source;
}

struct eltwise_impl : typed_primitive_impl<eltwise> {
    using parent = typed_primitive_impl<eltwise>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::vulkan::eltwise_impl)

    eltwise_impl() : parent("vulkan_eltwise"), _kernel_source(make_kernel_source()) {}

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<eltwise_impl>(*this);
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
        _kernels = cache.get_kernels(params);
        OPENVINO_ASSERT(_kernels.size() == 1, "[GPU][Vulkan] Eltwise expects exactly one SPIR-V kernel");
    }

    void init_by_cached_kernels(const kernels_cache& cache, std::vector<std::string>& cached_kernel_ids) override {
        _kernels.clear();
        for (const auto& id : cached_kernel_ids) {
            _kernels.push_back(cache.get_kernel_from_cached_kernels(id));
        }
    }

    std::vector<std::string> get_cached_kernel_ids(const kernels_cache& cache) override {
        return cache.get_cached_kernel_ids(_kernels);
    }

    std::vector<std::shared_ptr<kernel_string>> get_kernels_source() override {
        return _kernel_source == nullptr ? std::vector<std::shared_ptr<kernel_string>>{} : std::vector<std::shared_ptr<kernel_string>>{_kernel_source};
    }

    void reset_kernels_source() override {
        _kernel_source.reset();
    }

    std::vector<kernel::ptr> get_kernels() const override {
        return _kernels;
    }

    void set_kernels(kernels_cache::compiled_kernels kernels) override {
        OPENVINO_ASSERT(kernels.size() == 1, "[GPU][Vulkan] Eltwise expects one compiled kernel set");
        const auto& entries = kernels.begin()->second;
        _kernels.resize(entries.size());
        for (const auto& entry : entries) {
            _kernels.at(entry.second) = entry.first;
        }
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, eltwise_inst& instance) override {
        OPENVINO_ASSERT(_kernels.size() == 1 && _kernels.front() != nullptr, "[GPU][Vulkan] Eltwise kernel was not initialized");
        OPENVINO_ASSERT(instance.inputs_memory_count() == 2, "[GPU][Vulkan] Eltwise supports exactly two inputs");
        OPENVINO_ASSERT(instance.get_intermediates_memories().size() == 1, "[GPU][Vulkan] Eltwise metadata buffer was not allocated");

        auto& stream = instance.get_network().get_stream();
        const auto metadata = make_metadata(instance);
        auto metadata_memory = instance.get_intermediates_memories().front();
        metadata_memory->copy_from(stream, metadata.data(), true);

        kernel_arguments_desc descriptor;
        descriptor.layerID = instance.id();
        descriptor.workGroups.global = {metadata[0], 1, 1};
        descriptor.workGroups.local = {local_work_group_size, 1, 1};
        descriptor.arguments = {
            {argument_desc::Types::INPUT, 0},
            {argument_desc::Types::INPUT, 1},
            {argument_desc::Types::OUTPUT, 0},
            {argument_desc::Types::INTERNAL_BUFFER, 0},
        };

        kernel_arguments_data arguments;
        arguments.inputs = {instance.input_memory_ptr(0), instance.input_memory_ptr(1)};
        arguments.outputs = {instance.output_memory_ptr(0)};
        arguments.intermediates = {metadata_memory};
        return stream.enqueue_kernel(*_kernels.front(), descriptor, arguments, events, instance.needs_completion_event());
    }

private:
    std::shared_ptr<kernel_string> _kernel_source;
    std::vector<kernel::ptr> _kernels;
};

}  // namespace

bool EltwiseImplementationManager::validate_impl(const program_node& node) const {
    OPENVINO_ASSERT(node.is_type<eltwise>(), "[GPU][Vulkan] Invalid node type passed to Eltwise manager");
    if (node.get_program().get_engine().runtime_type() != runtime_types::vulkan || node.get_dependencies().size() != 2 || node.has_fused_primitives()) {
        return false;
    }

    const auto& desc = node.as<eltwise>().get_primitive();
    if (!is_supported_mode(desc->mode) || !desc->coefficients.empty() || !desc->stride.empty() ||
        (desc->broadcast_spec.m_type != ov::op::AutoBroadcastType::NUMPY && desc->broadcast_spec.m_type != ov::op::AutoBroadcastType::NONE)) {
        return false;
    }

    for (size_t index = 0; index < 2; ++index) {
        const auto& input_layout = node.get_input_layout(index);
        if (input_layout.data_type != data_types::f32 || !is_supported_format(input_layout.format.value) || input_layout.is_dynamic()) {
            return false;
        }
    }
    const auto& output_layout = node.get_output_layout(0);
    return output_layout.data_type == data_types::f32 && is_supported_format(output_layout.format.value) && !output_layout.is_dynamic() &&
           output_layout.get_rank() <= max_rank;
}

std::unique_ptr<primitive_impl> EltwiseImplementationManager::create_impl(const program_node& node, const kernel_impl_params&) const {
    OPENVINO_ASSERT(node.is_type<eltwise>(), "[GPU][Vulkan] Invalid node type passed to Eltwise manager");
    return std::make_unique<eltwise_impl>();
}

}  // namespace vulkan
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::vulkan::eltwise_impl)
