// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "gather_gguf.hpp"

#include <cstdint>

#include "../primitive_ocl_base.hpp"
#include "../utils/jitter.hpp"
#include "../utils/kernel_generator.hpp"
#include "intel_gpu/primitives/gather_gguf.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_gpu::ocl {
namespace {

struct GgufBlockGeometry {
    const char* jit_flag;
    uint32_t block_elem;
    uint32_t block_bytes;
};

GgufBlockGeometry gguf_block_geometry(ov::element::Type_t t) {
    switch (t) {
    case ov::element::Type_t::gguf_q4_0:
        return {"GGUF_IS_Q4_0", 32, 18};
    case ov::element::Type_t::gguf_q4_1:
        return {"GGUF_IS_Q4_1", 32, 20};
    case ov::element::Type_t::gguf_q8_0:
        return {"GGUF_IS_Q8_0", 32, 34};
    case ov::element::Type_t::gguf_q4_k:
        return {"GGUF_IS_Q4_K", 256, 144};
    case ov::element::Type_t::gguf_q5_k:
        return {"GGUF_IS_Q5_K", 256, 176};
    case ov::element::Type_t::gguf_q6_k:
        return {"GGUF_IS_Q6_K", 256, 210};
    default:
        OPENVINO_THROW("[GPU] gather_gguf: unsupported GGUF element type ", ov::element::Type(t).get_type_name());
    }
}

// Output row count = product of all output dims except the trailing hidden dim.
size_t derive_num_rows(const RuntimeParams& params) {
    const auto& out = params.get_output_layout(0);
    const auto& pshape = out.get_partial_shape();
    size_t rows = 1;
    for (size_t i = 0; i + 1 < pshape.size(); ++i) {
        rows *= static_cast<size_t>(pshape[i].get_length());
    }
    return rows;
}

class GatherGGUFRefGenerator : public KernelGenerator {
public:
    GatherGGUFRefGenerator() : KernelGenerator("gather_gguf") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto& desc = params.typed_desc<gather_gguf>();
        const auto geom = gguf_block_geometry(desc->weight_type);
        const int64_t hidden = desc->hidden_size;
        OPENVINO_ASSERT(hidden % geom.block_elem == 0,
                        "[GPU] gather_gguf: hidden size ",
                        hidden,
                        " is not a multiple of block element count ",
                        geom.block_elem);
        const int64_t blocks_per_row = hidden / geom.block_elem;

        jit.make(geom.jit_flag, 1);
        jit.make("GGUF_BLOCK_ELEM", geom.block_elem);
        jit.make("GGUF_BLOCK_BYTES", geom.block_bytes);
        jit.make("HIDDEN_SIZE", hidden);
        jit.make("VOCAB_SIZE", desc->vocab_size);
        jit.make("BLOCKS_PER_ROW", blocks_per_row);
        jit.make("ROW_BYTES", blocks_per_row * geom.block_bytes);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});  // raw gguf weight bytes
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});  // token indices
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* /*rt_params*/) {
            auto& wgs = kd.params.workGroups;
            if (params.is_dynamic()) {
                return;
            }
            const auto& desc = params.typed_desc<gather_gguf>();
            const auto geom = gguf_block_geometry(desc->weight_type);
            const size_t blocks_per_row = static_cast<size_t>(desc->hidden_size) / geom.block_elem;
            const size_t num_rows = derive_num_rows(params);
            // One work-item decodes one GGUF block (block_elem outputs).
            wgs.global = {num_rows * blocks_per_row, 1, 1};
            wgs.local = {1, 1, 1};
        }};
    }
};

class GatherGGUFRefImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::GatherGGUFRefImpl)

    Stage::Ptr gather_gguf = make_stage<GatherGGUFRefGenerator>();

    GatherGGUFRefImpl() : PrimitiveImplOCL(GatherGGUFRef::get_type_info_static()) {}
    GatherGGUFRefImpl(const program_node& node, const RuntimeParams& params) : GatherGGUFRefImpl() {
        add_stage(gather_gguf, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<GatherGGUFRefImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> GatherGGUFRef::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<gather_gguf>());
    return std::make_unique<GatherGGUFRefImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::gather_gguf)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::GatherGGUFRefImpl)
