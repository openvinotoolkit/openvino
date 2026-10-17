// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/state_conversion_executor.hpp"

#include "impls/ocl/kernels_cache.hpp"
#include "intel_gpu/runtime/kernel_args.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "openvino/core/except.hpp"

#ifdef OV_GPU_WITH_OCL_RT
#include "runtime/ocl/ocl_device.hpp"
#include "runtime/ocl/ocl_kernel.hpp"
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace cldnn {
namespace {

std::shared_ptr<kernel_string> make_source(state_conversion_key key) {
    auto source = std::make_shared<kernel_string>();
    source->entry_point = "state_convert_" + std::to_string(static_cast<int>(key.first)) + "_" +
                          std::to_string(static_cast<int>(key.second));

    const char* input_type = nullptr;
    const char* output_type = nullptr;
    std::string value;
    switch (key.first) {
    case data_types::bf16:
        input_type = "ushort";
        output_type = "half";
        value = "convert_half_rte(as_float(((uint)input[src_index]) << 16))";
        break;
    case data_types::f32:
        input_type = "float";
        if (key.second == data_types::f16) {
            output_type = "half";
            value = "convert_half_rte(input[src_index])";
        } else {
            output_type = "double";
            value = "(double)input[src_index]";
        }
        break;
    case data_types::f64:
        input_type = "double";
        output_type = "float";
        value = "convert_float_rte(input[src_index])";
        break;
    case data_types::i32:
        input_type = "int";
        output_type = key.second == data_types::i64 ? "long" :
                      key.second == data_types::u64 ? "ulong" : "uint";
        value = "(" + std::string(output_type) + ")input[src_index]";
        break;
    default:
        OPENVINO_THROW("[GPU] Unsupported state conversion type");
    }

    if (key.first == data_types::f64 || key.second == data_types::f64)
        source->str += "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    if (key.second == data_types::f16)
        source->str += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
    source->str += "__kernel void " + source->entry_point + "(__global const " + input_type + "* input, "
                  "__global " + output_type + "* output, ulong count, ulong src_count, ulong dst_count,\n"
                  "    ulong src_offset, ulong padded, ulong transpose,\n"
                  "    ulong d0, ulong d1, ulong d2, ulong d3, ulong d4, ulong d5,\n"
                  "    ulong s0, ulong s1, ulong s2, ulong s3, ulong s4, ulong s5) {\n"
                  "    size_t index = get_global_id(0);\n"
                  "    if (index >= count) return;\n"
                  "    size_t src_index = src_offset + index;\n"
                  "    if (padded) {\n"
                  "        const ulong dims[6] = {d0, d1, d2, d3, d4, d5};\n"
                  "        const ulong strides[6] = {s0, s1, s2, s3, s4, s5};\n"
                  "        size_t remaining = index;\n"
                  "        src_index = src_offset;\n"
                  "        for (int axis = 5; axis >= 0; --axis) {\n"
                  "            src_index += (remaining % dims[axis]) * strides[axis];\n"
                  "            remaining /= dims[axis];\n"
                  "        }\n"
                  "    }\n"
                  "    size_t dst_index = index;\n"
                  "    if (transpose) {\n"
                  "        size_t plane = d4 * d5;\n"
                  "        dst_index = (index / plane) * plane + (index % d5) * d4 + (index / d5) % d4;\n"
                  "    }\n"
                  "    // Skip reads and writes outside the supplied buffers.\n"
                  "    if (src_index >= src_count || dst_index >= dst_count || dst_index >= count) return;\n"
                  "    output[dst_index] = " + value + ";\n"
                  "}\n";
    return source;
}

}  // namespace

void state_conversion_executor::set_kernels(const std::vector<state_conversion_key>& keys,
                                            const std::vector<kernel::ptr>& kernels, const engine& engine) {
    OPENVINO_ASSERT(keys.size() == kernels.size(), "[GPU] State conversion kernel count mismatch");
    OPENVINO_ASSERT(engine.runtime_type() == runtime_types::ocl, "[GPU] State conversion requires OpenCL");
#ifdef OV_GPU_WITH_OCL_RT
    const auto device = engine.get_device();
    const auto& ocl_device = downcast<ocl::ocl_device>(*device).get_device();
    const auto device_limit = static_cast<size_t>(engine.get_device_info().max_work_group_size);
    _kernels.clear();
    for (size_t i = 0; i < keys.size(); ++i) {
        OPENVINO_ASSERT(supports(keys[i]) && kernels[i], "[GPU] Invalid state conversion kernel");
        const auto& handle = downcast<ocl::ocl_kernel>(*kernels[i]).get_handle();
        // Query once for both compiled and imported kernels, never during set_state().
        const auto kernel_limit = handle.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(ocl_device);
        const auto limit = std::min(device_limit, kernel_limit);
        OPENVINO_ASSERT(limit > 0, "[GPU] Invalid state conversion work-group limit");
        OPENVINO_ASSERT(_kernels.emplace(keys[i], kernel_info{kernels[i], limit}).second,
                        "[GPU] Duplicate state conversion kernel");
    }
#else
    OPENVINO_THROW("[GPU] State conversion requires OpenCL");
#endif
}

bool state_conversion_executor::prepare_input(const layout& source_layout, const ov::Strides& strides,
                                               bool transpose, state_conversion_input& input) {
    const auto shape = source_layout.get_shape();
    if (shape.empty() || shape.size() > 6 || strides.size() != shape.size() ||
        !format::is_default_format(source_layout.format) || (transpose && shape.size() < 2))
        return false;

    const auto physical_shape = source_layout.get_padded_dims();
    const auto element_size = data_type_traits::size_of(source_layout.data_type);
    state_conversion_input prepared;
    prepared.source_span_bytes = element_size;
    prepared.padded = static_cast<bool>(source_layout.data_padding);
    prepared.transpose = transpose;
    size_t pitch_bytes = element_size;
    for (size_t i = shape.size(); i-- > 0;) {
        if (shape[i] == 0 || strides[i] == 0 || (shape[i] > 1 && strides[i] != pitch_bytes) ||
            source_layout.data_padding._lower_size[i] < 0 || source_layout.data_padding._upper_size[i] < 0 ||
            physical_shape[i] <= 0)
            return false;
        const auto lower = static_cast<size_t>(source_layout.data_padding._lower_size[i]);
        prepared.source_span_bytes += lower * pitch_bytes;
        prepared.source_offset += lower * (pitch_bytes / element_size);
        prepared.source_span_bytes += (shape[i] - 1) * pitch_bytes;
        // Right-align dimensions so axes 4 and 5 are always the last two tensor axes.
        const size_t axis = 6 - shape.size() + i;
        prepared.dimensions[axis] = shape[i];
        prepared.strides[axis] = pitch_bytes / element_size;
        prepared.count *= shape[i];
        pitch_bytes *= static_cast<size_t>(physical_shape[i]);
    }
    input = prepared;
    return true;
}

event::ptr state_conversion_executor::execute(state_conversion_key key, memory::cptr src, memory::cptr dst,
                                              stream& stream, const state_conversion_input& input,
                                              const std::vector<event::ptr>& dependencies) {
    auto it = _kernels.find(key);
    OPENVINO_ASSERT(it != _kernels.end(), "[GPU] State conversion kernel was not prepared");
    const auto count = input.count;
    if (count == 0)
        return nullptr;
    const auto dst_count = dst->size() / data_type_traits::size_of(key.second);
    OPENVINO_ASSERT(count <= dst_count, "[GPU] State conversion destination memory is too small");

    kernel_arguments_desc desc;
    const auto local_size = std::min(count, it->second.max_work_group_size);
    OPENVINO_ASSERT(local_size > 0, "[GPU] Invalid state conversion work-group limit");
    desc.workGroups.global = {((count - 1) / local_size + 1) * local_size, 1, 1};
    desc.workGroups.local = {local_size, 1, 1};
    desc.arguments = {{argument_desc::Types::INPUT, 0},
                      {argument_desc::Types::OUTPUT, 0}};
    const auto add_scalar = [&](uint64_t value) {
        desc.arguments.push_back({argument_desc::Types::SCALAR, static_cast<uint32_t>(desc.scalars.size())});
        scalar_desc scalar{};
        scalar.t = scalar_desc::Types::UINT64;
        scalar.v.u64 = value;
        desc.scalars.push_back(scalar);
    };
    add_scalar(count);
    add_scalar(src->size() / data_type_traits::size_of(key.first));
    add_scalar(dst_count);
    add_scalar(input.source_offset);
    add_scalar(static_cast<uint64_t>(input.padded));
    add_scalar(static_cast<uint64_t>(input.transpose));
    for (auto dimension : input.dimensions)
        add_scalar(dimension);
    for (auto stride : input.strides)
        add_scalar(stride);
    desc.layerID = "state_conversion";

    kernel_arguments_data args;
    args.inputs.push_back(std::move(src));
    args.outputs.push_back(std::move(dst));
    args.scalars = &desc.scalars;

    std::lock_guard<std::mutex> lock(_mutex);
    stream.set_arguments(*it->second.compiled_kernel, desc, args);
    return stream.enqueue_kernel(*it->second.compiled_kernel, desc, args, dependencies, true);
}

std::shared_ptr<state_conversion_executor> state_conversion_registry::get() const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _executor;
}

void state_conversion_registry::prepare(const engine& engine, kernels_cache& cache,
                                        const std::vector<state_conversion_key>& requested_keys) {
    std::lock_guard<std::mutex> lock(_mutex);

    std::set<state_conversion_key> unique_keys;
    if (engine.runtime_type() == runtime_types::ocl) {
        const auto& device_info = engine.get_device_info();
        for (const auto& key : requested_keys) {
            const bool needs_fp64 = key.first == data_types::f64 || key.second == data_types::f64;
            const bool needs_fp16 = key.first == data_types::f16 || key.second == data_types::f16;
            if (key.first != key.second && state_conversion_executor::supports(key) &&
                (!needs_fp64 || device_info.supports_fp64) && (!needs_fp16 || device_info.supports_fp16))
                unique_keys.insert(key);
        }
    }
    std::vector<state_conversion_key> keys(unique_keys.begin(), unique_keys.end());

    if (_prepared) {
        const auto existing_keys = _executor ? _executor->get_keys() : std::vector<state_conversion_key>{};
        OPENVINO_ASSERT(keys == existing_keys, "[GPU] State conversion kernels do not match the network");
        return;
    }

    if (!keys.empty()) {
        std::vector<std::shared_ptr<kernel_string>> sources;
        sources.reserve(keys.size());
        for (const auto& key : keys)
            sources.push_back(make_source(key));

        kernel_impl_params params;
        auto compiled = cache.compile(params, sources);
        OPENVINO_ASSERT(compiled.size() == 1, "[GPU] State conversion compilation failed");
        std::vector<kernel::ptr> kernels(keys.size());
        for (const auto& entry : compiled.begin()->second) {
            OPENVINO_ASSERT(entry.second < kernels.size(), "[GPU] Invalid state conversion kernel index");
            kernels[entry.second] = entry.first;
        }
        auto executor = std::make_shared<state_conversion_executor>();
        executor->set_kernels(keys, kernels, engine);
        _executor = std::move(executor);
    }
    _prepared = true;
}

void state_conversion_registry::restore(const engine& engine, const std::vector<state_conversion_key>& keys,
                                        const std::vector<kernel::ptr>& kernels) {
    std::lock_guard<std::mutex> lock(_mutex);
    if (!keys.empty()) {
        auto executor = std::make_shared<state_conversion_executor>();
        executor->set_kernels(keys, kernels, engine);
        _executor = std::move(executor);
    }
    _prepared = true;
}

std::shared_ptr<state_conversion_executor> program::get_state_conversion_executor() const {
    return _state_conversions->get();
}

void program::prepare_state_conversions(const std::vector<state_conversion_key>& requested_keys) {
    _state_conversions->prepare(_engine, *_kernels_cache, requested_keys);
}

}  // namespace cldnn
