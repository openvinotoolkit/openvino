// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "implementation_map.hpp"
#include "intel_gpu/primitives/arg_max_min.hpp"
#include "intel_gpu/primitives/dynamic_quantize.hpp"

#define OV_GPU_WITH_ONEDNN ENABLE_ONEDNN_FOR_GPU
#define OV_GPU_WITH_OCL 1
#define OV_GPU_WITH_COMMON 1
#define OV_GPU_WITH_CPU 1

#define COUNT_N(_1, _2, _3, _4, _5, N, ...) N
#define COUNT(...) COUNT_N(__VA_ARGS__, 5, 4, 3, 2, 1)
#define CAT(a, b) a ## b

#define IDENTITY(N) N

#define IMPL_TYPE_CPU_D impl_types::cpu, cldnn::shape_types::dynamic_shape
#define IMPL_TYPE_CPU_S impl_types::cpu, cldnn::shape_types::static_shape
#define IMPL_TYPE_OCL_D impl_types::ocl, cldnn::shape_types::dynamic_shape
#define IMPL_TYPE_OCL_S impl_types::ocl, cldnn::shape_types::static_shape
#define IMPL_TYPE_COMMON_D impl_types::common, cldnn::shape_types::dynamic_shape
#define IMPL_TYPE_COMMON_S impl_types::common, cldnn::shape_types::static_shape

#define INSTANTIATE_1(prim, suffix) cldnn::implementation_map<cldnn::prim>::get(cldnn::CAT(IMPL_TYPE_, suffix))
#define INSTANTIATE_2(prim, suffix, ...) INSTANTIATE_1(prim, suffix), INSTANTIATE_1(prim, __VA_ARGS__)
#define INSTANTIATE_3(prim, suffix, ...) INSTANTIATE_1(prim, suffix), INSTANTIATE_2(prim, __VA_ARGS__)
#define INSTANTIATE_4(prim, suffix, ...) INSTANTIATE_1(prim, suffix), INSTANTIATE_3(prim, __VA_ARGS__)

#define FOR_EACH_(N, prim, ...) CAT(INSTANTIATE_, N)(prim, __VA_ARGS__)
#define INSTANTIATE(prim, ...) IDENTITY(FOR_EACH_(COUNT(__VA_ARGS__), prim, __VA_ARGS__))

#define CREATE_INSTANCE(Type, ...) std::make_shared<cldnn::Type>(__VA_ARGS__)
#define GET_INSTANCE(Type, ...) cldnn::implementation_map<cldnn::Type>::get(__VA_ARGS__)

#if defined(OV_GPU_WITH_ONEDNN)
#    define OV_GPU_CREATE_INSTANCE_ONEDNN(...) CREATE_INSTANCE(__VA_ARGS__)
#else
#    define OV_GPU_CREATE_INSTANCE_ONEDNN(...)
#endif

#if defined(OV_GPU_WITH_OCL)
#    define OV_GPU_CREATE_INSTANCE_OCL(...) CREATE_INSTANCE(__VA_ARGS__)
#else
#    define OV_GPU_CREATE_INSTANCE_OCL(...)
#endif

#if defined(OV_GPU_WITH_OCL)
#    define OV_GPU_GET_INSTANCE_OCL_1(prim, shape_types) GET_INSTANCE(prim, cldnn::impl_types::ocl, shape_types)
#    define OV_GPU_GET_INSTANCE_OCL_2(prim, shape_types, verify_callback) \
    std::make_shared<cldnn::ImplementationManagerLegacy<cldnn::prim>>( \
    std::dynamic_pointer_cast<cldnn::ImplementationManagerLegacy<cldnn::prim>>(GET_INSTANCE(prim, cldnn::impl_types::ocl, shape_types)).get(), verify_callback)
#    define SELECT(N, prim, ...) CAT(OV_GPU_GET_INSTANCE_OCL_, N)(prim, __VA_ARGS__)
#    define OV_GPU_GET_INSTANCE_OCL(prim, ...) IDENTITY(SELECT(COUNT(__VA_ARGS__), prim, __VA_ARGS__))
#else
#    define OV_GPU_GET_INSTANCE_OCL(...)
#endif

#if defined(OV_GPU_WITH_COMMON)
#    define OV_GPU_GET_INSTANCE_COMMON(prim, ...) GET_INSTANCE(prim, cldnn::impl_types::common, __VA_ARGS__)
#else
#    define OV_GPU_GET_INSTANCE_COMMON(...)
#endif
#if defined(OV_GPU_WITH_CPU)
#    define OV_GPU_GET_INSTANCE_CPU(prim, ...) GET_INSTANCE(prim, cldnn::impl_types::cpu, __VA_ARGS__)
#else
#    define OV_GPU_GET_INSTANCE_CPU(...)
#endif

#define REGISTER_DEFAULT_IMPLS(prim, ...)  \
    namespace cldnn { struct prim; } \
    template<> struct ov::intel_gpu::Registry<cldnn::prim> { \
        static const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& get_implementations() { \
            static const std::vector<std::shared_ptr<cldnn::ImplementationManager>> impls = { \
                INSTANTIATE(prim, __VA_ARGS__)  \
            }; \
            return impls; \
        } \
    }

#define REGISTER_IMPLS(prim)  \
    namespace cldnn { struct prim; } \
    template<> struct ov::intel_gpu::Registry<cldnn::prim> { \
        static const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& get_implementations(); \
    }

namespace ov {
namespace intel_gpu {

// Global list of implementations for given primitive type
// List must be sorted by priority of implementations
// Same impls may repeat multiple times with different configurations
template<typename PrimitiveType>
struct Registry {
    static const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& get_implementations() {
        static_assert(cldnn::meta::always_false<PrimitiveType>::value, "Only specialization instantiations are allowed");
        OPENVINO_NOT_IMPLEMENTED;
    }
};

}  // namespace intel_gpu
}  // namespace ov

REGISTER_IMPLS(arg_max_min);
REGISTER_IMPLS(concatenation);
REGISTER_IMPLS(convolution);
REGISTER_IMPLS(deconvolution);
REGISTER_IMPLS(fully_connected);
REGISTER_IMPLS(gemm);
REGISTER_IMPLS(pooling);
REGISTER_IMPLS(reduce);
REGISTER_IMPLS(reorder);
REGISTER_IMPLS(reshape);

REGISTER_DEFAULT_IMPLS(assign, CPU_S, CPU_D);
REGISTER_DEFAULT_IMPLS(read_value, CPU_S, CPU_D);
REGISTER_DEFAULT_IMPLS(condition, COMMON_S, COMMON_D);
REGISTER_DEFAULT_IMPLS(loop, COMMON_S, COMMON_D);
REGISTER_DEFAULT_IMPLS(input_layout, COMMON_S, COMMON_D);
REGISTER_DEFAULT_IMPLS(non_max_suppression_gather, CPU_S);
REGISTER_DEFAULT_IMPLS(proposal, CPU_S, CPU_D);
REGISTER_DEFAULT_IMPLS(activation, OCL_S, OCL_D, CPU_S, CPU_D);
REGISTER_DEFAULT_IMPLS(adaptive_pooling, OCL_S);
REGISTER_DEFAULT_IMPLS(batch_to_space, OCL_S);
REGISTER_DEFAULT_IMPLS(border, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(broadcast, OCL_S, OCL_D, CPU_S, CPU_D);
REGISTER_DEFAULT_IMPLS(bucketize, OCL_S);
REGISTER_DEFAULT_IMPLS(crop, OCL_S, OCL_D, CPU_S, CPU_D);
REGISTER_DEFAULT_IMPLS(custom_gpu_primitive, OCL_S);
REGISTER_DEFAULT_IMPLS(data, COMMON_S, COMMON_D);
REGISTER_DEFAULT_IMPLS(depth_to_space, OCL_S);
REGISTER_DEFAULT_IMPLS(detection_output, OCL_S, CPU_S, CPU_D);
REGISTER_DEFAULT_IMPLS(dft, OCL_S);
REGISTER_DEFAULT_IMPLS(dynamic_quantize, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(experimental_detectron_detection_output, OCL_S);
REGISTER_DEFAULT_IMPLS(experimental_detectron_generate_proposals_single_image, OCL_S);
REGISTER_DEFAULT_IMPLS(experimental_detectron_prior_grid_generator, OCL_S);
REGISTER_DEFAULT_IMPLS(experimental_detectron_roi_feature_extractor, OCL_S);
REGISTER_DEFAULT_IMPLS(experimental_detectron_topk_rois, OCL_S);
REGISTER_DEFAULT_IMPLS(eltwise, OCL_S, OCL_D, CPU_S, CPU_D);
REGISTER_DEFAULT_IMPLS(gather, OCL_S, OCL_D, CPU_S, CPU_D);
REGISTER_DEFAULT_IMPLS(gather_nd, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(gather_elements, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(generate_proposals, OCL_S);
REGISTER_DEFAULT_IMPLS(grid_sample, OCL_S);
REGISTER_DEFAULT_IMPLS(group_normalization, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(kv_cache, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(lrn, OCL_S);
REGISTER_DEFAULT_IMPLS(lstm_elt, OCL_S);
REGISTER_DEFAULT_IMPLS(multiclass_nms, OCL_S);
REGISTER_DEFAULT_IMPLS(multinomial, OCL_S);
REGISTER_DEFAULT_IMPLS(mutable_data, OCL_S);
REGISTER_DEFAULT_IMPLS(mvn, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(non_max_suppression, OCL_S, CPU_S);
REGISTER_DEFAULT_IMPLS(matrix_nms, OCL_S);
REGISTER_DEFAULT_IMPLS(normalize, OCL_S);
REGISTER_DEFAULT_IMPLS(one_hot, OCL_S);
REGISTER_DEFAULT_IMPLS(permute, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(prior_box, OCL_S, COMMON_S);
REGISTER_DEFAULT_IMPLS(quantize, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(random_uniform, OCL_S);
REGISTER_DEFAULT_IMPLS(range, OCL_S, OCL_D, CPU_S, CPU_D);
REGISTER_DEFAULT_IMPLS(region_yolo, OCL_S);
REGISTER_DEFAULT_IMPLS(reorg_yolo, OCL_S);
REGISTER_DEFAULT_IMPLS(reverse, OCL_S);
REGISTER_DEFAULT_IMPLS(reverse_sequence, OCL_S);
REGISTER_DEFAULT_IMPLS(rms, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(roi_align, OCL_S);
REGISTER_DEFAULT_IMPLS(roi_pooling, OCL_S);
REGISTER_DEFAULT_IMPLS(roll, OCL_S);
REGISTER_DEFAULT_IMPLS(scatter_update, OCL_S, OCL_D, CPU_S, CPU_D);
REGISTER_DEFAULT_IMPLS(scatter_elements_update, OCL_S);
REGISTER_DEFAULT_IMPLS(scatter_nd_update, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(select, OCL_S, OCL_D, CPU_S, CPU_D);
REGISTER_DEFAULT_IMPLS(shape_of, OCL_S, OCL_D, CPU_S, CPU_D);
REGISTER_DEFAULT_IMPLS(shuffle_channels, OCL_S);
REGISTER_DEFAULT_IMPLS(slice, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(softmax, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(space_to_batch, OCL_S);
REGISTER_DEFAULT_IMPLS(space_to_depth, OCL_S);
REGISTER_DEFAULT_IMPLS(strided_slice, OCL_S, OCL_D, CPU_S, CPU_D);
REGISTER_DEFAULT_IMPLS(swiglu, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(tile, OCL_S, OCL_D, CPU_S, CPU_D);
REGISTER_DEFAULT_IMPLS(gather_tree, OCL_S);
REGISTER_DEFAULT_IMPLS(resample, OCL_S);
REGISTER_DEFAULT_IMPLS(grn, OCL_S);
REGISTER_DEFAULT_IMPLS(ctc_greedy_decoder, OCL_S);
REGISTER_DEFAULT_IMPLS(ctc_loss, OCL_S);
REGISTER_DEFAULT_IMPLS(cum_sum, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(embedding_bag, OCL_S);
REGISTER_DEFAULT_IMPLS(extract_image_patches, OCL_S);
REGISTER_DEFAULT_IMPLS(convert_color, OCL_S);
REGISTER_DEFAULT_IMPLS(count_nonzero, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(gather_nonzero, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(eye, OCL_S);
REGISTER_DEFAULT_IMPLS(unique_count, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(unique_gather, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(scaled_dot_product_attention, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(rope, OCL_S, OCL_D);
