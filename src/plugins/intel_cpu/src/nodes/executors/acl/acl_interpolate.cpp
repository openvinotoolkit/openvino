// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_interpolate.hpp"
#include "acl_utils.hpp"

static arm_compute::TensorShape interpolateShapeCast(const ov::intel_cpu::VectorDims& dims) {
    arm_compute::TensorShape tensorShape;
    for (std::size_t i = 0; i < dims.size(); ++i) {
        tensorShape.set(dims.size() - i - 1, dims[i], false);
    }
    if (tensorShape.num_dimensions() == 0) {
        tensorShape.set(0, 1, false);
        tensorShape.set_num_dimensions(1);
    }
    return tensorShape;
}

bool ov::intel_cpu::ACLInterpolateExecutor::init(const InterpolateAttrs &interpolateAttrs,
                                                 const std::vector <MemoryDescPtr> &srcDescs,
                                                 const std::vector <MemoryDescPtr> &dstDescs,
                                                 const dnnl::primitive_attr &attr) {
    InterpolateExecutor::init(interpolateAttrs, srcDescs, dstDescs, attr);
    aclInterpolateAttrs = interpolateAttrs;
    auto& coord_mode = aclInterpolateAttrs.coordTransMode;
    auto& inter_mode = aclInterpolateAttrs.mode;
    acl_coord = arm_compute::SamplingPolicy::TOP_LEFT;
    auto& out_shape = dstDescs[0]->getShape().getDims();

    if ((coord_mode == InterpolateCoordTransMode::pytorch_half_pixel && out_shape[2] > 1 && out_shape[3] > 1) ||
        coord_mode == InterpolateCoordTransMode::half_pixel) {
        acl_coord = arm_compute::SamplingPolicy::CENTER;
    }

    switch (inter_mode) {
        case InterpolateMode::linear:
        case InterpolateMode::linear_onnx:
            acl_policy = arm_compute::InterpolationPolicy::BILINEAR;
            break;
        case InterpolateMode::nearest:
            acl_policy = arm_compute::InterpolationPolicy::NEAREST_NEIGHBOR;
            break;
        default:
            return false;
    }

    auto srcDims = srcDescs[0]->getShape().getStaticDims();
    auto dstDims = dstDescs[0]->getShape().getStaticDims();
    auto srcTensorInfo = arm_compute::TensorInfo(interpolateShapeCast(srcDims), 1,
                                                 precisionToAclDataType(srcDescs[0]->getPrecision()),
                                                 getAclDataLayoutByMemoryDesc(srcDescs[0]));
    auto dstTensorInfo = arm_compute::TensorInfo(interpolateShapeCast(dstDims), 1,
                                                 precisionToAclDataType(dstDescs[0]->getPrecision()),
                                                 getAclDataLayoutByMemoryDesc(dstDescs[0]));

    if (!arm_compute::NEScale::validate(&srcTensorInfo,
                                        &dstTensorInfo,
                                        arm_compute::ScaleKernelInfo(acl_policy,
                                                                     arm_compute::BorderMode::REPLICATE,
                                                                     arm_compute::PixelValue(),
                                                                     acl_coord,
                                                                     false,
                                                                     coord_mode == InterpolateCoordTransMode::align_corners)))
        return false;

    srcTensor.allocator()->init(srcTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    acl_scale = std::make_unique<arm_compute::NEScale>();
    acl_scale->configure(&srcTensor, &dstTensor, arm_compute::ScaleKernelInfo(acl_policy,
                                                                              arm_compute::BorderMode::REPLICATE,
                                                                              arm_compute::PixelValue(),
                                                                              acl_coord,
                                                                              false,
                                                                              aclInterpolateAttrs.coordTransMode == InterpolateCoordTransMode::align_corners));
    return true;
}

void ov::intel_cpu::ACLInterpolateExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) {
    auto in_ptr_ = padPreprocess(src, dst);
    srcTensor.allocator()->import_memory(const_cast<void *>(reinterpret_cast<const void *>(in_ptr_)));
    dstTensor.allocator()->import_memory(dst[0]->GetPtr());

    acl_scale->run();

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
}

bool ov::intel_cpu::ACLInterpolateExecutorBuilder::isSupportedConfiguration(
        const ov::intel_cpu::InterpolateAttrs &interpolateAttrs, const std::vector<MemoryDescPtr> &srcDescs,
        const std::vector<MemoryDescPtr> &dstDescs) {
    auto& inp_shape = srcDescs[0]->getShape().getDims();
    auto& out_shape = dstDescs[0]->getShape().getDims();

    float scale_h = static_cast<float>(out_shape[2]) / inp_shape[2];
    float scale_w = static_cast<float>(out_shape[3]) / inp_shape[3];
    bool is_upsample = scale_h > 1 && scale_w > 1;

    auto& coord_mode = interpolateAttrs.coordTransMode;
    auto& nearest_mode = interpolateAttrs.nearestMode;

    if (coord_mode == InterpolateCoordTransMode::asymmetric &&
        nearest_mode == InterpolateNearestMode::floor) {
        return is_upsample;
    }

    if (coord_mode == InterpolateCoordTransMode::align_corners &&
        nearest_mode == InterpolateNearestMode::round_prefer_ceil) {
        return true;
    }

    if (coord_mode == InterpolateCoordTransMode::half_pixel &&
        (nearest_mode == InterpolateNearestMode::simple || nearest_mode == InterpolateNearestMode::round_prefer_ceil)) {
        return false;
    }

    if (coord_mode == InterpolateCoordTransMode::asymmetric &&
        (nearest_mode == InterpolateNearestMode::simple || nearest_mode == InterpolateNearestMode::floor)) {
        return is_upsample;
    }

    if (is_upsample) {
        bool int_factor = scale_h == static_cast<int>(scale_h) && scale_w == static_cast<int>(scale_w);
        if (int_factor && coord_mode != InterpolateCoordTransMode::asymmetric &&
            (nearest_mode == InterpolateNearestMode::round_prefer_ceil
             || nearest_mode == InterpolateNearestMode::round_prefer_floor)) {
            return true;
        }
    } else if (scale_h < 1 && scale_w < 1) {
        float down_scale_h = static_cast<float>(inp_shape[2]) / out_shape[2];
        float down_scale_w = static_cast<float>(inp_shape[3]) / out_shape[3];
        bool int_factor = down_scale_h == static_cast<int>(down_scale_h) && down_scale_w == static_cast<int>(down_scale_w);

        if (int_factor && coord_mode != InterpolateCoordTransMode::align_corners &&
            nearest_mode == InterpolateNearestMode::simple) {
            return true;
        }

        if (int_factor && nearest_mode == InterpolateNearestMode::round_prefer_ceil &&
            ((out_shape[2] > 1 && out_shape[3] > 1) || coord_mode != InterpolateCoordTransMode::half_pixel)) {
            return true;
        }
    }
    return false;
}

bool ov::intel_cpu::ACLInterpolateExecutorBuilder::isSupported(const ov::intel_cpu::InterpolateAttrs &interpolateAttrs,
                                                               const std::vector<MemoryDescPtr> &srcDescs,
                                                               const std::vector<MemoryDescPtr> &dstDescs) const {
    if (srcDescs[0]->getShape().getDims().size() != 4) {
        return false;
    }

    auto& pads_begin = interpolateAttrs.padBegin;
    auto& pads_end   = interpolateAttrs.padEnd;

    if (!std::all_of(pads_begin.begin(), pads_begin.end(), [](int i){return i == 0;}) ||
        !std::all_of(pads_end.begin(), pads_end.end(), [](int i){return i == 0;})) {
        return false;
    }

    auto& nearest_mode = interpolateAttrs.nearestMode;
    auto& coord_mode   = interpolateAttrs.coordTransMode;
    if (interpolateAttrs.antialias ||
        coord_mode == InterpolateCoordTransMode::tf_half_pixel_for_nn ||
        nearest_mode == InterpolateNearestMode::ceil) {
        return false;
    }

    if (interpolateAttrs.mode == InterpolateMode::cubic) {
        return false;
    }

    if (interpolateAttrs.mode == InterpolateMode::nearest &&
        !isSupportedConfiguration(interpolateAttrs, srcDescs, dstDescs)) {
        return false;
    }

    if (coord_mode == InterpolateCoordTransMode::pytorch_half_pixel) {
        return false;
    }
    return true;
}
