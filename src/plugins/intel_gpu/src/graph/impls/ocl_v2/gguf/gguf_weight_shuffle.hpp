#pragma once

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "openvino/core/type/element_type.hpp"
#include "../utils/kernels_db.hpp"

namespace ov::intel_gpu::ocl {

inline cldnn::memory::ptr upload_shuffled_gguf_weights(cldnn::engine& engine,
                                                      const cldnn::layout& layout,
                                                      const ov::element::Type& type,
                                                      const ov::Shape& shape,
                                                      const void* data,
                                                      cldnn::kernel::ptr& compiled_kernel) {
    const std::array<ov::element::Type, 6> formats = {ov::element::gguf_q4_k,
                                                    ov::element::gguf_q5_k,
                                                    ov::element::gguf_q6_k,
                                                    ov::element::gguf_q4_0,
                                                    ov::element::gguf_q4_1,
                                                    ov::element::gguf_q8_0};
    const auto position = std::find(formats.begin(), formats.end(), type);
    OPENVINO_ASSERT(position != formats.end() && shape.size() == 2 && shape[0] != 0 && shape[1] != 0 &&
                        shape[0] % 16 == 0 && shape[1] % 256 == 0,
                    "Unsupported GPU GGUF shuffle geometry");
    OPENVINO_ASSERT(layout.bytes_count() == ov::shape_size(shape) / type.block_elem_count() * type.block_byte_size(),
                    "GPU GGUF shuffle requires an unpadded weight buffer");
    if (!compiled_kernel) {
        const auto source = SourcesDB::get_kernel_template("gguf_weight_shuffle");
        const auto options = "-cl-std=CL1.2 -DGGUF_SHUFFLE_FORMAT=" + std::to_string(position - formats.begin());
        std::vector<cldnn::kernel::ptr> kernels;
        engine.create_kernel_builder()->build_kernels(source.data(), source.size(), cldnn::KernelFormat::SOURCE, options, kernels);
        OPENVINO_ASSERT(kernels.size() == 1, "Expected one GGUF shuffle kernel");
        compiled_kernel = kernels.front();
    }
    auto& stream = engine.get_service_stream();
    auto input = engine.allocate_memory(layout, false);
    auto output = engine.allocate_memory(layout, false);
    input->copy_from(stream, data, true);
    const uint64_t blocks_per_row = shape[1] / 256;
    const uint64_t total_blocks = shape[0] * blocks_per_row;
    cldnn::kernel_arguments_desc descriptor;
    descriptor.workGroups.global = {static_cast<size_t>((total_blocks + 63) / 64 * 64), 1, 1};
    descriptor.workGroups.local = {64, 1, 1};
    descriptor.arguments = {{cldnn::argument_desc::Types::INPUT, 0},
                            {cldnn::argument_desc::Types::OUTPUT, 0},
                            {cldnn::argument_desc::Types::SCALAR, 0},
                            {cldnn::argument_desc::Types::SCALAR, 1}};
    cldnn::scalar_desc columns{};
    columns.t = cldnn::scalar_desc::Types::UINT64;
    columns.v.u64 = blocks_per_row;
    cldnn::scalar_desc blocks{};
    blocks.t = cldnn::scalar_desc::Types::UINT64;
    blocks.v.u64 = total_blocks;
    descriptor.scalars = {columns, blocks};
    cldnn::kernel_arguments_data arguments;
    arguments.inputs = {input};
    arguments.outputs = {output};
    arguments.scalars = &descriptor.scalars;
    stream.set_arguments(*compiled_kernel, descriptor, arguments);
    stream.enqueue_kernel(*compiled_kernel, descriptor, arguments, {}, true)->wait();
    return output;
}

}