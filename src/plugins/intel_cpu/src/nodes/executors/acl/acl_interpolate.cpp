// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_interpolate.hpp"

#include "acl_utils.hpp"
#include "utils/debug_capabilities.h"

bool ov::intel_cpu::ACLInterpolateExecutor::init(const InterpolateAttrs& interpolateAttrs,
                                                 const std::vector<MemoryDescPtr>& srcDescs,
                                                 const std::vector<MemoryDescPtr>& dstDescs,
                                                 const dnnl::primitive_attr& attr) {
    aclInterpolateAttrs = interpolateAttrs;
    InterpolateExecutor::init(aclInterpolateAttrs, srcDescs, dstDescs, attr);
    acl_coord = arm_compute::SamplingPolicy::TOP_LEFT;
    auto& out_shape = dstDescs[0]->getShape().getDims();

    static const size_t index_h = 2;
    static const size_t index_w = 3;
    if ((aclInterpolateAttrs.coordTransMode == InterpolateCoordTransMode::pytorch_half_pixel &&
         out_shape[index_h] > 1 && out_shape[index_w] > 1) ||
        aclInterpolateAttrs.coordTransMode == InterpolateCoordTransMode::half_pixel) {
        acl_coord = arm_compute::SamplingPolicy::CENTER;
    }

    switch (aclInterpolateAttrs.mode) {
    case InterpolateMode::linear:
    case InterpolateMode::linear_onnx:
        acl_policy = arm_compute::InterpolationPolicy::BILINEAR;
        break;
    case InterpolateMode::nearest:
        acl_policy = arm_compute::InterpolationPolicy::NEAREST_NEIGHBOR;
        break;
    default:
        DEBUG_LOG("Unsupported interpolate mode: ", static_cast<int>(aclInterpolateAttrs.mode));
        return false;
    }

    auto srcDims = shapeCast(srcDescs[0]->getShape().getDims());
    auto dstDims = shapeCast(dstDescs[0]->getShape().getDims());

    if (srcDescs[0]->hasLayoutType(LayoutType::nspc) && dstDescs[0]->hasLayoutType(LayoutType::nspc)) {
        changeLayoutToNH_C({&srcDims, &dstDims});
    }

    auto srcTensorInfo = arm_compute::TensorInfo(srcDims,
                                                 1,
                                                 precisionToAclDataType(srcDescs[0]->getPrecision()),
                                                 getAclDataLayoutByMemoryDesc(srcDescs[0]));
    auto dstTensorInfo = arm_compute::TensorInfo(dstDims,
                                                 1,
                                                 precisionToAclDataType(dstDescs[0]->getPrecision()),
                                                 getAclDataLayoutByMemoryDesc(dstDescs[0]));

    arm_compute::Status status = arm_compute::NEScale::validate(
        &srcTensorInfo,
        &dstTensorInfo,
        arm_compute::ScaleKernelInfo(acl_policy,
                                     arm_compute::BorderMode::REPLICATE,
                                     arm_compute::PixelValue(),
                                     acl_coord,
                                     false,
                                     aclInterpolateAttrs.coordTransMode == InterpolateCoordTransMode::align_corners,
                                     getAclDataLayoutByMemoryDesc(srcDescs[0])));
    if (!status) {
        DEBUG_LOG("NEScale validation failed: ", status.error_description());
        return false;
    }

    srcTensor.allocator()->init(srcTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    acl_scale = std::make_unique<arm_compute::NEScale>();
    configureThreadSafe([&] {
        acl_scale->configure(
            &srcTensor,
            &dstTensor,
            arm_compute::ScaleKernelInfo(acl_policy,
                                         arm_compute::BorderMode::REPLICATE,
                                         arm_compute::PixelValue(),
                                         acl_coord,
                                         false,
                                         aclInterpolateAttrs.coordTransMode == InterpolateCoordTransMode::align_corners,
                                         getAclDataLayoutByMemoryDesc(srcDescs[0])));
    });
    return true;
}

void ov::intel_cpu::ACLInterpolateExecutor::exec(const std::vector<MemoryCPtr>& src,
                                                 const std::vector<MemoryPtr>& dst,
                                                 const void* post_ops_data_) {
    auto in_ptr_ = padPreprocess(src, dst);
    srcTensor.allocator()->import_memory(const_cast<void*>(reinterpret_cast<const void*>(in_ptr_)));
    dstTensor.allocator()->import_memory(dst[0]->getData());

    acl_scale->run();

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
}

bool ov::intel_cpu::ACLInterpolateExecutorBuilder::isSupportedConfiguration(
    const ov::intel_cpu::InterpolateAttrs& interpolateAttrs,
    const std::vector<MemoryDescPtr>& srcDescs,
    const std::vector<MemoryDescPtr>& dstDescs) {
    OPENVINO_ASSERT(srcDescs[0]->getShape().getDims().size() == 4);

    auto& inp_shape = srcDescs[0]->getShape().getDims();
    auto& out_shape = dstDescs[0]->getShape().getDims();

    static const size_t index_h = 2;
    static const size_t index_w = 3;
    float scale_h = static_cast<float>(out_shape[index_h]) / inp_shape[index_h];
    float scale_w = static_cast<float>(out_shape[index_w]) / inp_shape[index_w];
    bool is_upsample = scale_h > 1 && scale_w > 1;

    auto& coord_mode = interpolateAttrs.coordTransMode;
    auto& nearest_mode = interpolateAttrs.nearestMode;

    if (coord_mode == InterpolateCoordTransMode::align_corners &&
        nearest_mode == InterpolateNearestMode::round_prefer_ceil) {
        DEBUG_LOG("InterpolateCoordTransMode::align_corners with InterpolateNearestMode::round_prefer_ceil supported");
        return true;
    }

    if (coord_mode == InterpolateCoordTransMode::half_pixel &&
        (nearest_mode == InterpolateNearestMode::simple || nearest_mode == InterpolateNearestMode::round_prefer_ceil)) {
        DEBUG_LOG("InterpolateCoordTransMode half_pixel is not supported for InterpolateNearestMode simple and "
                  "round_prefer_ceil");
        return false;
    }

    if (coord_mode == InterpolateCoordTransMode::asymmetric &&
        (nearest_mode == InterpolateNearestMode::simple || nearest_mode == InterpolateNearestMode::floor)) {
        DEBUG_LOG("asymmetric && (simple || floor) mode with upsample: ", is_upsample);
        return is_upsample;
    }

    if (is_upsample) {
        bool int_factor = scale_h == static_cast<int>(scale_h) && scale_w == static_cast<int>(scale_w);
        if (int_factor && coord_mode != InterpolateCoordTransMode::asymmetric &&
            (nearest_mode == InterpolateNearestMode::round_prefer_ceil ||
             nearest_mode == InterpolateNearestMode::round_prefer_floor)) {
            DEBUG_LOG(
                "upsample && int_factor && !asymmetric && (round_prefer_ceil || round_prefer_floor) case is supported");
            return true;
        }
    } else if (scale_h < 1 && scale_w < 1) {
        float down_scale_h = static_cast<float>(inp_shape[index_h]) / out_shape[index_h];
        float down_scale_w = static_cast<float>(inp_shape[index_w]) / out_shape[index_w];
        bool int_factor =
            down_scale_h == static_cast<int>(down_scale_h) && down_scale_w == static_cast<int>(down_scale_w);

        if (int_factor && coord_mode != InterpolateCoordTransMode::align_corners &&
            nearest_mode == InterpolateNearestMode::simple) {
            DEBUG_LOG("!upsample && int_factor && !align_corners && simple case is supported");
            return true;
        }

        if (int_factor && nearest_mode == InterpolateNearestMode::round_prefer_ceil &&
            ((out_shape[index_h] > 1 && out_shape[index_w] > 1) ||
             coord_mode != InterpolateCoordTransMode::half_pixel)) {
            DEBUG_LOG(
                "!upsample && int_factor && round_prefer_ceil && (out_shape > 1 || half_pixel) case is supported");
            return true;
        }
    }
    DEBUG_LOG("ACL Interpolate executor does not support such configuration: coord_mode=",
              static_cast<int>(coord_mode),
              " nearest_mode=",
              static_cast<int>(nearest_mode),
              " upsample=",
              is_upsample,
              " scale_h=",
              scale_h,
              " scale_w=",
              scale_w);
    return false;
}

bool ov::intel_cpu::ACLInterpolateExecutorBuilder::isSupported(const ov::intel_cpu::InterpolateAttrs& interpolateAttrs,
                                                               const std::vector<MemoryDescPtr>& srcDescs,
                                                               const std::vector<MemoryDescPtr>& dstDescs) const {
    if (srcDescs[0]->getShape().getDims().size() != 4u) {
        DEBUG_LOG("ACL Interpolate does not support src shape rank: ", srcDescs[0]->getShape().getDims().size());
        return false;
    }

    auto& pads_begin = interpolateAttrs.padBegin;
    auto& pads_end = interpolateAttrs.padEnd;

    if (!std::all_of(pads_begin.begin(),
                     pads_begin.end(),
                     [](int i) {
                         return i == 0;
                     }) ||
        !std::all_of(pads_end.begin(), pads_end.end(), [](int i) {
            return i == 0;
        })) {
        DEBUG_LOG("ACL Interpolate does not support padding");
        return false;
    }

    if (interpolateAttrs.antialias ||
        interpolateAttrs.coordTransMode == InterpolateCoordTransMode::tf_half_pixel_for_nn ||
        interpolateAttrs.nearestMode == InterpolateNearestMode::ceil) {
        DEBUG_LOG("ACL Interpolate does not support antialias, tf_half_pixel_for_nn, ceil modes");
        return false;
    }

    if (interpolateAttrs.mode == InterpolateMode::cubic || interpolateAttrs.mode == InterpolateMode::bilinear_pillow ||
        interpolateAttrs.mode == InterpolateMode::bicubic_pillow) {
        DEBUG_LOG("ACL Interpolate does not support cubic, bilinear_pillow, bicubic_pillow modes");
        return false;
    }

    if (interpolateAttrs.shapeCalcMode == InterpolateShapeCalcMode::scales &&
        one_of(interpolateAttrs.coordTransMode,
               InterpolateCoordTransMode::half_pixel,
               InterpolateCoordTransMode::asymmetric) &&
        one_of(interpolateAttrs.mode, InterpolateMode::linear, InterpolateMode::linear_onnx)) {
        DEBUG_LOG("ACL Interpolate does not support scales mode with linear/linear_onnx and half_pixel/asymmetric");
        return false;
    }

    if (interpolateAttrs.mode == InterpolateMode::nearest &&
        !isSupportedConfiguration(interpolateAttrs, srcDescs, dstDescs)) {
        DEBUG_LOG("ACL Interpolate isSupportedConfiguration method fails for nearest mode");
        return false;
    }

    if (interpolateAttrs.coordTransMode == InterpolateCoordTransMode::pytorch_half_pixel) {
        DEBUG_LOG("ACL Interpolate does not support pytorch_half_pixel mode");
        return false;
    }
    return true;
}
