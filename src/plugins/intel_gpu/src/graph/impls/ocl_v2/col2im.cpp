// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "col2im.hpp"

#include "col2im_inst.h"
#include "common_utils/dispatch_utils.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

static std::vector<size_t> get_orig_size(const kernel_impl_params& params) {
    constexpr size_t spatial_dims = 2;
    const auto& desc = params.typed_desc<col2im>();

    const auto& output_size = desc->output_shape;
    const auto& kernel_size = desc->kernel_shape;
    const auto& stride = desc->stride;
    const auto& dilation = desc->dilation;
    const auto& pads_begin = desc->padding_begin;
    const auto& pads_end = desc->padding_end;

    std::vector<size_t> orig_size(spatial_dims);
    for (size_t d = 0; d < spatial_dims; ++d) {
        orig_size[d] = ((output_size[d] + pads_begin[d] + pads_end[d] - (dilation[d] * (kernel_size[d] - 1)) - 1) / stride[d]) + 1;
    }

    return orig_size;
}

bool check_col2im_contain_batch(const kernel_impl_params& params) {
    auto input_layout = params.get_input_layout();
    auto orig_size = get_orig_size(params);

    // Check input size L which is the total number of blocks : product from d=1
    // to 2 of origin size
    if (input_layout.spatial(1) == 1 && input_layout.spatial(1) != static_cast<tensor::value_type>(orig_size[0] * orig_size[1])) {
        return false;
    }

    return true;
}

class Col2ImGenerator : public KernelGenerator {
public:
    Col2ImGenerator() : KernelGenerator("col2im") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto& desc = params.typed_desc<col2im>();

        auto input_layout = params.get_input_layout();

        auto orig_size = get_orig_size(params);

        // Consider input tensor : (N, C * Product(kernel_size), L)
        bool is_batched = check_col2im_contain_batch(params);

        const auto num_blocks = is_batched ? input_layout.spatial(1) : input_layout.feature();

        const size_t num_elements_for_block = is_batched ? input_layout.feature() : input_layout.batch();
        const size_t kernel_product = (size_t)(desc->kernel_shape[0] * desc->kernel_shape[1]);
        const size_t num_channels = std::max(num_elements_for_block / kernel_product, (size_t)1);

        GPU_DEBUG_TRACE << "  Col2im Batched " << (is_batched ? "true " : "false ") << " num_elements_for_block : " << num_elements_for_block
                        << ", num_channels : " << num_channels << ", num_blocks : " << num_blocks << " to " << desc->kernel_shape[0] << ", "
                        << desc->kernel_shape[1] << std::endl;

        jit.add({
            make_jit_constant("ORIG_HEIGHT", orig_size[0]),
            make_jit_constant("ORIG_WIDTH", orig_size[1]),
            make_jit_constant("NUM_ELEMENTS_FOR_BLOCK", num_elements_for_block),
            make_jit_constant("KERNEL_PRODUCT", kernel_product),
            make_jit_constant("NUM_CHANNELS", num_channels),
            make_jit_constant("NUM_BLOCKS", num_blocks),
            make_jit_constant("OUT_SIZE_0", desc->output_shape[0]),
            make_jit_constant("OUT_SIZE_1", desc->output_shape[1]),
            make_jit_constant("KERNEL_SIZE_0", desc->kernel_shape[0]),
            make_jit_constant("KERNEL_SIZE_1", desc->kernel_shape[1]),
            make_jit_constant("STRIDE_0", desc->stride[0]),
            make_jit_constant("STRIDE_1", desc->stride[1]),
            make_jit_constant("DILATION_0", desc->dilation[0]),
            make_jit_constant("DILATION_1", desc->dilation[1]),
            make_jit_constant("PAD_BEGIN_0", desc->padding_begin[0]),
            make_jit_constant("PAD_BEGIN_1", desc->padding_begin[1]),
            make_jit_constant("PAD_END_0", desc->padding_end[0]),
            make_jit_constant("PAD_END_1", desc->padding_end[1]),
        });
        return jit;
    }

    Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            const auto& desc = params.typed_desc<col2im>();

            auto& wgs = kd.params.workGroups;
            auto input_layout = params.get_input_layout();
            auto output_layout = params.get_output_layout();

            bool is_batched = check_col2im_contain_batch(params);

            const auto batches = is_batched ? (size_t)output_layout.batch() : (size_t)1;

            const size_t num_elements_for_block = is_batched ? input_layout.feature() : input_layout.batch();
            const size_t kernel_product = (size_t)(desc->kernel_shape[0] * desc->kernel_shape[1]);
            const size_t num_channels = std::max(num_elements_for_block / kernel_product, (size_t)1);

            wgs.global = {num_channels, 1, batches};
            wgs.local = {1, 1, 1};
        }};
    }
};

class Col2ImImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::Col2ImImpl)

    Stage::Ptr col2im = make_stage<Col2ImGenerator>();

    Col2ImImpl() : PrimitiveImplOCL(Col2Im::get_type_info_static()) {}
    Col2ImImpl(const program_node& node, const RuntimeParams& params) : Col2ImImpl() {
        add_stage(col2im, params);
    }
    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<Col2ImImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> Col2Im::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<col2im>());
    return std::make_unique<Col2ImImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::col2im)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::Col2ImImpl)
