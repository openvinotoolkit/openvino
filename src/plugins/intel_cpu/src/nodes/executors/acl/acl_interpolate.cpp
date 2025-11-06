// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_interpolate.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/NEON/functions/NEScale.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "acl_utils.hpp"
#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "openvino/core/except.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"

bool ov::intel_cpu::AclInterpolateExecutor::update(const MemoryArgs& memory) {
    std::vector<MemoryDescPtr> srcDescs{memory.at(ARG_SRC)->getDescPtr()};
    std::vector<MemoryDescPtr> dstDescs{memory.at(ARG_DST)->getDescPtr()};
    acl_coord = arm_compute::SamplingPolicy::TOP_LEFT;
    const auto& out_shape = dstDescs[0]->getShape().getDims();

    static const size_t index_h = 2;
    static const size_t index_w = 3;
    if ((aclInterpolateAttrs.coordTransMode == ov::intel_cpu::InterpolateCoordTransMode::pytorch_half_pixel &&
         out_shape[index_h] > 1 && out_shape[index_w] > 1) ||
        aclInterpolateAttrs.coordTransMode == ov::intel_cpu::InterpolateCoordTransMode::half_pixel) {
        acl_coord = arm_compute::SamplingPolicy::CENTER;
    }

    switch (aclInterpolateAttrs.mode) {
    case ov::intel_cpu::InterpolateMode::linear:
    case ov::intel_cpu::InterpolateMode::linear_onnx:
        acl_policy = arm_compute::InterpolationPolicy::BILINEAR;
        break;
    case ov::intel_cpu::InterpolateMode::nearest:
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
                                     aclInterpolateAttrs.coordTransMode == ov::intel_cpu::InterpolateCoordTransMode::align_corners,
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
                                         aclInterpolateAttrs.coordTransMode == ov::intel_cpu::InterpolateCoordTransMode::align_corners,
                                         getAclDataLayoutByMemoryDesc(srcDescs[0])));
    });
    return true;
}

void ov::intel_cpu::AclInterpolateExecutor::execute(const MemoryArgs& memory) {
    srcTensor.allocator()->import_memory(const_cast<void*>(memory.at(ARG_SRC)->getData()));
    dstTensor.allocator()->import_memory(memory.at(ARG_DST)->getData());

    acl_scale->run();

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
}

static bool isSupportedConfiguration(
    const ov::intel_cpu::InterpolateAttrs& interpolateAttrs,
    const std::vector<ov::intel_cpu::MemoryDescPtr>& srcDescs,
    const std::vector<ov::intel_cpu::MemoryDescPtr>& dstDescs) {
    OPENVINO_ASSERT(srcDescs[0]->getShape().getDims().size() == 4);

    const auto& inp_shape = srcDescs[0]->getShape().getDims();
    const auto& out_shape = dstDescs[0]->getShape().getDims();

    static const size_t index_h = 2;
    static const size_t index_w = 3;
    float scale_h = static_cast<float>(out_shape[index_h]) / inp_shape[index_h];
    float scale_w = static_cast<float>(out_shape[index_w]) / inp_shape[index_w];
    bool is_upsample = scale_h > 1 && scale_w > 1;

    const auto& coord_mode = interpolateAttrs.coordTransMode;
    const auto& nearest_mode = interpolateAttrs.nearestMode;

    if (coord_mode == ov::intel_cpu::InterpolateCoordTransMode::align_corners &&
        nearest_mode == ov::intel_cpu::InterpolateNearestMode::round_prefer_ceil) {
        DEBUG_LOG("InterpolateCoordTransMode::align_corners with InterpolateNearestMode::round_prefer_ceil supported");
        return true;
    }

    if (coord_mode == ov::intel_cpu::InterpolateCoordTransMode::half_pixel &&
        (nearest_mode == ov::intel_cpu::InterpolateNearestMode::simple ||
         nearest_mode == ov::intel_cpu::InterpolateNearestMode::round_prefer_ceil)) {
        DEBUG_LOG("InterpolateCoordTransMode half_pixel is not supported for InterpolateNearestMode simple and "
                  "round_prefer_ceil");
        return false;
    }

    if (coord_mode == ov::intel_cpu::InterpolateCoordTransMode::asymmetric &&
        (nearest_mode == ov::intel_cpu::InterpolateNearestMode::simple ||
         nearest_mode == ov::intel_cpu::InterpolateNearestMode::floor)) {
        DEBUG_LOG("asymmetric && (simple || floor) mode with upsample: ", is_upsample);
        return is_upsample;
    }

    if (is_upsample) {
        bool int_factor = (scale_h == std::floor(scale_h)) && (scale_w == std::floor(scale_w));
        if (int_factor && coord_mode != ov::intel_cpu::InterpolateCoordTransMode::asymmetric &&
            (nearest_mode == ov::intel_cpu::InterpolateNearestMode::round_prefer_ceil ||
             nearest_mode == ov::intel_cpu::InterpolateNearestMode::round_prefer_floor)) {
            DEBUG_LOG(
                "upsample && int_factor && !asymmetric && (round_prefer_ceil || round_prefer_floor) case is supported");
            return true;
        }
    } else if (scale_h < 1 && scale_w < 1) {
        float down_scale_h = static_cast<float>(inp_shape[index_h]) / out_shape[index_h];
        float down_scale_w = static_cast<float>(inp_shape[index_w]) / out_shape[index_w];
        bool int_factor = (down_scale_h == std::floor(down_scale_h)) && (down_scale_w == std::floor(down_scale_w));

        if (int_factor && coord_mode != ov::intel_cpu::InterpolateCoordTransMode::align_corners &&
            nearest_mode == ov::intel_cpu::InterpolateNearestMode::simple) {
            DEBUG_LOG("!upsample && int_factor && !align_corners && simple case is supported");
            return true;
        }

        if (int_factor && nearest_mode == ov::intel_cpu::InterpolateNearestMode::round_prefer_ceil &&
            ((out_shape[index_h] > 1 && out_shape[index_w] > 1) ||
             coord_mode != ov::intel_cpu::InterpolateCoordTransMode::half_pixel)) {
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

bool ov::intel_cpu::AclInterpolateExecutor::supports(const ov::intel_cpu::InterpolateConfig& config) {
    const auto& interpolateAttrs = config.attrs;
    std::vector<MemoryDescPtr> srcDescs{config.descs.at(ARG_SRC)};
    std::vector<MemoryDescPtr> dstDescs{config.descs.at(ARG_DST)};
    if (srcDescs[0]->getShape().getDims().size() != 4U) {
        DEBUG_LOG("ACL Interpolate does not support src shape rank: ", srcDescs[0]->getShape().getDims().size());
        return false;
    }

    const auto& pads_begin = interpolateAttrs.padBegin;
    const auto& pads_end = interpolateAttrs.padEnd;

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
        interpolateAttrs.coordTransMode == ov::intel_cpu::InterpolateCoordTransMode::tf_half_pixel_for_nn ||
        interpolateAttrs.nearestMode == ov::intel_cpu::InterpolateNearestMode::ceil) {
        DEBUG_LOG("ACL Interpolate does not support antialias, tf_half_pixel_for_nn, ceil modes");
        return false;
    }

    if (interpolateAttrs.mode == ov::intel_cpu::InterpolateMode::cubic ||
        interpolateAttrs.mode == ov::intel_cpu::InterpolateMode::bilinear_pillow ||
        interpolateAttrs.mode == ov::intel_cpu::InterpolateMode::bicubic_pillow) {
        DEBUG_LOG("ACL Interpolate does not support cubic, bilinear_pillow, bicubic_pillow modes");
        return false;
    }

    if (interpolateAttrs.shapeCalcMode == ov::intel_cpu::InterpolateShapeCalcMode::scales &&
        any_of(interpolateAttrs.coordTransMode,
               ov::intel_cpu::InterpolateCoordTransMode::half_pixel,
               ov::intel_cpu::InterpolateCoordTransMode::asymmetric) &&
        any_of(interpolateAttrs.mode,
               ov::intel_cpu::InterpolateMode::linear,
               ov::intel_cpu::InterpolateMode::linear_onnx)) {
        DEBUG_LOG("ACL Interpolate does not support scales mode with linear/linear_onnx and half_pixel/asymmetric");
        return false;
    }

    if (interpolateAttrs.mode == ov::intel_cpu::InterpolateMode::nearest &&
        !isSupportedConfiguration(interpolateAttrs, srcDescs, dstDescs)) {
        DEBUG_LOG("ACL Interpolate isSupportedConfiguration method fails for nearest mode");
        return false;
    }

    if (interpolateAttrs.coordTransMode == ov::intel_cpu::InterpolateCoordTransMode::pytorch_half_pixel) {
        DEBUG_LOG("ACL Interpolate does not support pytorch_half_pixel mode");
        return false;
    }
    return true;
}
