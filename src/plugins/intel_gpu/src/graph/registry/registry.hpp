// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "implementation_map.hpp"

#ifdef ENABLE_ONEDNN_FOR_GPU
    #define OV_GPU_WITH_ONEDNN 1
#else
    #define OV_GPU_WITH_ONEDNN 0
#endif

#if !defined(OV_GPU_WITH_SYCL)
    #define OV_GPU_WITH_SYCL 0
#endif

#define OV_GPU_WITH_OCL 1
#define OV_GPU_WITH_COMMON 1
#define OV_GPU_WITH_CPU 1
#define OV_GPU_WITH_CM 1

#define COUNT_N(_1, _2, _3, _4, _5, N, ...) N
#define COUNT(...) EXPAND(COUNT_N(__VA_ARGS__, 5, 4, 3, 2, 1))
#define CAT(a, b) a ## b

#define EXPAND(N) N

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

#define FOR_EACH_(N, prim, ...) EXPAND(CAT(INSTANTIATE_, N)(prim, __VA_ARGS__))
#define INSTANTIATE(prim, ...) EXPAND(FOR_EACH_(COUNT(__VA_ARGS__), prim, __VA_ARGS__))

#define CREATE_INSTANCE(Type, ...) std::make_shared<Type>(__VA_ARGS__),
#define GET_INSTANCE(Type, ...) cldnn::implementation_map<cldnn::Type>::get(__VA_ARGS__)

#define OV_GPU_GET_INSTANCE_1(prim, impl_type, shape_types) GET_INSTANCE(prim, impl_type, shape_types),
#define OV_GPU_GET_INSTANCE_2(prim, impl_type, shape_types, verify_callback) \
    std::make_shared<cldnn::ImplementationManagerLegacy<cldnn::prim>>( \
    std::dynamic_pointer_cast<cldnn::ImplementationManagerLegacy<cldnn::prim>>(GET_INSTANCE(prim, impl_type, shape_types)).get(), verify_callback),

#define SELECT(N, ...) EXPAND(CAT(OV_GPU_GET_INSTANCE_, N)(__VA_ARGS__))

#if OV_GPU_WITH_ONEDNN
#    define OV_GPU_CREATE_INSTANCE_ONEDNN(...) EXPAND(CREATE_INSTANCE(__VA_ARGS__))
#else
#    define OV_GPU_CREATE_INSTANCE_ONEDNN(...)
#endif

#if OV_GPU_WITH_SYCL
#    define OV_GPU_CREATE_INSTANCE_SYCL(...) EXPAND(CREATE_INSTANCE(__VA_ARGS__))
#else
#    define OV_GPU_CREATE_INSTANCE_SYCL(...)
#endif

#if OV_GPU_WITH_CM
#    define OV_GPU_CREATE_INSTANCE_CM(...) EXPAND(CREATE_INSTANCE(__VA_ARGS__))
#else
#    define OV_GPU_CREATE_INSTANCE_CM(...)
#endif

#if OV_GPU_WITH_OCL
#    define OV_GPU_CREATE_INSTANCE_OCL(...) EXPAND(CREATE_INSTANCE(__VA_ARGS__))
#    define OV_GPU_GET_INSTANCE_OCL(prim, ...) EXPAND(SELECT(COUNT(__VA_ARGS__), prim, impl_types::ocl, __VA_ARGS__))
#else
#    define OV_GPU_CREATE_INSTANCE_OCL(...)
#    define OV_GPU_GET_INSTANCE_OCL(...)
#endif

#if OV_GPU_WITH_COMMON
#    define OV_GPU_GET_INSTANCE_COMMON(prim, ...) EXPAND(GET_INSTANCE(prim, cldnn::impl_types::common, __VA_ARGS__))
#else
#    define OV_GPU_GET_INSTANCE_COMMON(...)
#endif

#if OV_GPU_WITH_CPU
#    define OV_GPU_GET_INSTANCE_CPU(prim, ...) EXPAND(SELECT(COUNT(__VA_ARGS__), prim, impl_types::cpu, __VA_ARGS__))
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

namespace ov::intel_gpu {

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

}  // namespace ov::intel_gpu

REGISTER_IMPLS(activation);
REGISTER_IMPLS(arg_max_min);
REGISTER_IMPLS(broadcast);
REGISTER_IMPLS(concatenation);
REGISTER_IMPLS(convolution);
REGISTER_IMPLS(crop);
REGISTER_IMPLS(ctc_loss);
REGISTER_IMPLS(deconvolution);
REGISTER_IMPLS(detection_output);
REGISTER_IMPLS(eltwise);
REGISTER_IMPLS(fake_convert);
REGISTER_IMPLS(fully_connected);
REGISTER_IMPLS(gather);
REGISTER_IMPLS(gather_nd);
REGISTER_IMPLS(gemm);
REGISTER_IMPLS(group_normalization);
REGISTER_IMPLS(lstm_cell);
REGISTER_IMPLS(lstm_seq);
REGISTER_IMPLS(non_max_suppression);
REGISTER_IMPLS(pooling);
REGISTER_IMPLS(reduce);
REGISTER_IMPLS(reorder);
REGISTER_IMPLS(reshape);
REGISTER_IMPLS(range);
REGISTER_IMPLS(rope);
REGISTER_IMPLS(select);
REGISTER_IMPLS(scatter_update);
REGISTER_IMPLS(scatter_elements_update);
REGISTER_IMPLS(softmax);
REGISTER_IMPLS(shape_of);
REGISTER_IMPLS(strided_slice);
REGISTER_IMPLS(tile);

REGISTER_DEFAULT_IMPLS(assign, CPU_S, CPU_D);
REGISTER_DEFAULT_IMPLS(read_value, CPU_S, CPU_D);
REGISTER_DEFAULT_IMPLS(condition, COMMON_S, COMMON_D);
REGISTER_DEFAULT_IMPLS(loop, COMMON_S, COMMON_D);
REGISTER_DEFAULT_IMPLS(input_layout, COMMON_S, COMMON_D);
REGISTER_DEFAULT_IMPLS(non_max_suppression_gather, CPU_S);
REGISTER_DEFAULT_IMPLS(proposal, CPU_S, CPU_D);
REGISTER_DEFAULT_IMPLS(adaptive_pooling, OCL_S);
REGISTER_DEFAULT_IMPLS(batch_to_space, OCL_S);
REGISTER_DEFAULT_IMPLS(border, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(bucketize, OCL_S);
REGISTER_DEFAULT_IMPLS(col2im, OCL_S);
REGISTER_DEFAULT_IMPLS(custom_gpu_primitive, OCL_S);
REGISTER_DEFAULT_IMPLS(data, COMMON_S, COMMON_D);
REGISTER_DEFAULT_IMPLS(depth_to_space, OCL_S);
REGISTER_DEFAULT_IMPLS(dft, OCL_S);
REGISTER_DEFAULT_IMPLS(dynamic_quantize, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(experimental_detectron_detection_output, OCL_S);
REGISTER_DEFAULT_IMPLS(experimental_detectron_generate_proposals_single_image, OCL_S);
REGISTER_DEFAULT_IMPLS(experimental_detectron_prior_grid_generator, OCL_S);
REGISTER_DEFAULT_IMPLS(experimental_detectron_roi_feature_extractor, OCL_S);
REGISTER_DEFAULT_IMPLS(experimental_detectron_topk_rois, OCL_S);
REGISTER_DEFAULT_IMPLS(gather_elements, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(generate_proposals, OCL_S);
REGISTER_DEFAULT_IMPLS(grid_sample, OCL_S);
REGISTER_DEFAULT_IMPLS(kv_cache, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(lrn, OCL_S);
REGISTER_DEFAULT_IMPLS(multiclass_nms, OCL_S);
REGISTER_DEFAULT_IMPLS(multinomial, OCL_S);
REGISTER_DEFAULT_IMPLS(mutable_data, OCL_S);
REGISTER_DEFAULT_IMPLS(mvn, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(matrix_nms, OCL_S);
REGISTER_DEFAULT_IMPLS(normalize, OCL_S);
REGISTER_DEFAULT_IMPLS(one_hot, OCL_S);
REGISTER_DEFAULT_IMPLS(paged_attention, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(permute, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(prior_box, OCL_S);
REGISTER_DEFAULT_IMPLS(quantize, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(random_uniform, OCL_S);
REGISTER_DEFAULT_IMPLS(region_yolo, OCL_S);
REGISTER_DEFAULT_IMPLS(reorg_yolo, OCL_S);
REGISTER_DEFAULT_IMPLS(reverse, OCL_S);
REGISTER_DEFAULT_IMPLS(reverse_sequence, OCL_S);
REGISTER_DEFAULT_IMPLS(rms, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(roi_align, OCL_S);
REGISTER_DEFAULT_IMPLS(roi_pooling, OCL_S);
REGISTER_DEFAULT_IMPLS(roll, OCL_S);
REGISTER_DEFAULT_IMPLS(scatter_nd_update, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(shuffle_channels, OCL_S);
REGISTER_DEFAULT_IMPLS(slice, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(space_to_batch, OCL_S);
REGISTER_DEFAULT_IMPLS(space_to_depth, OCL_S);
REGISTER_DEFAULT_IMPLS(swiglu, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(gather_tree, OCL_S);
REGISTER_DEFAULT_IMPLS(resample, OCL_S);
REGISTER_DEFAULT_IMPLS(grn, OCL_S);
REGISTER_DEFAULT_IMPLS(ctc_greedy_decoder, OCL_S);
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
REGISTER_DEFAULT_IMPLS(search_sorted, OCL_S, OCL_D);
REGISTER_DEFAULT_IMPLS(STFT, OCL_S, OCL_D);
