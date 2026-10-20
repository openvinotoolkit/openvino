// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "recurrent_cache_ref.hpp"

#include "intel_gpu/primitives/recurrent_cache.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"
namespace ov::intel_gpu::ocl {
namespace {
class RecurrentCacheRefGenerator : public KernelGenerator {
public:
    RecurrentCacheRefGenerator() : KernelGenerator("recurrent_cache_ref") {}

protected:
    JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.make("HISTORY", params.typed_desc<recurrent_cache>()->history);
        jit.make("STATE_KEY_SIZE", params.typed_desc<recurrent_cache>()->key_size);
        jit.make("STATE_VALUE_SIZE", params.typed_desc<recurrent_cache>()->value_size);
        return jit;
    }
    Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic())
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        for (uint32_t i = 0; i < params.input_layouts.size(); ++i)
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        for (uint32_t i = 0; i < params.output_layouts.size(); ++i)
            args.push_back({ArgumentDescriptor::Types::OUTPUT, i});
        return args;
    }
    DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            const size_t count = params.output_layouts.size() == 1 ? params.output_layouts[0].count()
                                                                   : std::max(params.output_layouts[0].count(), params.output_layouts[1].count());
            kd.params.workGroups.global = {((count + 255) / 256) * 256, 1, 1};
            kd.params.workGroups.local = {256, 1, 1};
        }};
    }
};
class RecurrentCacheRefImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::RecurrentCacheRefImpl)
    Stage::Ptr cache = make_stage<RecurrentCacheRefGenerator>();
    RecurrentCacheRefImpl() : PrimitiveImplOCL(RecurrentCacheRef::get_type_info_static()) {}
    RecurrentCacheRefImpl(const program_node&, const RuntimeParams& params) : RecurrentCacheRefImpl() {
        add_stage(cache, params);
    }
    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<RecurrentCacheRefImpl>(this);
    }
};
}  // namespace
std::unique_ptr<primitive_impl> RecurrentCacheRef::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<recurrent_cache>());
    return std::make_unique<RecurrentCacheRefImpl>(node, params);
}
}  // namespace ov::intel_gpu::ocl
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::recurrent_cache)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::RecurrentCacheRefImpl)
