// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executors/interpolate.hpp"
#include "executors/interpolate_list.hpp"
#include "node.h"

#define MAX_INPUT_INTERPOLATE 8

namespace ov::intel_cpu::node {

class InterpolateRefExecutor : public InterpolateExecutorBase {
public:
    InterpolateRefExecutor(const InterpolateAttrs &interpAttrs,
                           const VectorDims &srcDims,
                           const VectorDims &dstDims,
                           const std::vector<float> &_dataScales)
            : InterpolateExecutorBase(interpAttrs, srcDims, dstDims, _dataScales),
              antialias(interpAttrs.antialias),
              dataScales(_dataScales),
              refInterpAttrs(interpAttrs) {}

    void exec(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_) override;

private:
    void
    NNRef(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int ID, int IH, int IW, int OD, int OH, int OW);

    void linearOnnxRef(const uint8_t *in_ptr_,
                       uint8_t *out_ptr_,
                       int B,
                       int C,
                       int ID,
                       int IH,
                       int IW,
                       int OD,
                       int OH,
                       int OW);

    void cubicRef(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int IH, int IW, int OH, int OW);

    void linearInterpolation(const uint8_t *in_ptr_,
                             uint8_t *out_ptr_,
                             int B,
                             int C,
                             int ID,
                             int IH,
                             int IW,
                             float fx,
                             float fy,
                             float fz,
                             int OD,
                             int OH,
                             int OW,
                             int kernel_width,
                             bool antialias);

    void pillowRef(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int IH, int IW, int OH, int OW);

    void
    pillowRefNCHWAsNHWC(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int IH, int IW, int OH, int OW);

    static float getValue(const uint8_t *base, size_t offset, ov::element::Type prec);

    static void setValue(uint8_t *base, size_t offset, float value, ov::element::Type prec);

private:
    bool antialias;
    std::vector<float> dataScales;
    InterpolateAttrs refInterpAttrs;
};

class Interpolate : public Node {
public:
    static constexpr size_t DATA_ID = 0;
    static constexpr size_t TARGET_SHAPE_ID = 1;
    static constexpr size_t SCALES_ID = 2;
    static constexpr size_t AXES_ID = 3;
    static constexpr size_t SIZE_OR_SCALE_ID_V11 = 1;
    static constexpr size_t AXES_ID_V11 = 2;
    static constexpr int CUBIC_GRID_LEN = 4;
    static constexpr float PILLOW_BILINEAR_WINDOW_SCALE = 1.0f;
    static constexpr float PILLOW_BICUBIC_WINDOW_SCALE = 2.0f;

public:
    Interpolate(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;
    bool canBeInPlace() const override {
        return false;
    }
    bool canFuse(const NodePtr& node) const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    bool needShapeInfer() const override;
    bool needPrepareParams() const override;
    void prepareParams() override;

    inline int get_scale_id() const;
    inline int get_axis_id() const;

private:
    bool is_version11 = true;
    InterpolateAttrs interpAttrs;
    size_t dataRank = 0;
    std::shared_ptr<InterpolateExecutorBase> execPtr = nullptr;

    void setPostOps(dnnl::primitive_attr& attr, const VectorDims& dims);

    static VectorDims getPaddedInputShape(const VectorDims& srcDims,
                                          const std::vector<int>& padBegin,
                                          const std::vector<int>& padEnd);
    std::vector<float> getScales(const VectorDims& srcDimPad, const VectorDims& dstDim);
    static size_t getSpatialDimsNum(const Dim rank);

    bool hasPad = false;

    bool isAxesSpecified = false;
    std::vector<int> axes;
    std::vector<float> scales;
    bool isScaleConstant = false;

    // 6 ptrs for each quantization, 2 ptrs for each depth_wise
    std::vector<const void*> postOpsDataPtrs;

    std::vector<float> lastScales;
    std::vector<int32_t> lastSizes;

    VectorDims lastOutputDims;

    bool canUseAclExecutor = false;
    std::shared_ptr<InterpolateExecutor> aclExecPtr = nullptr;
};

} // namespace ov::intel_cpu::node


