// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executors/interpolate.hpp"
#include "executors/interpolate_list.hpp"
#include "node.h"

#define MAX_INPUT_INTERPOLATE 8

namespace ov {
namespace intel_cpu {
namespace node {

struct jit_interpolate_config_params {
    InterpolateLayoutType layout;
    InterpolateMode mode;
    ov::element::Type src_prc;
    ov::element::Type dst_prc;
    int src_data_size;
    int dst_data_size;
    int indices_size;
    int spatial_dim_size;
    int C, ID, IH, IW, OD, OH, OW;
    // for pillow
    int filterLenX;
    int filterLenY;
    int* bound;
};

struct jit_interpolate_call_args {
    const void *src_ptr[MAX_INPUT_INTERPOLATE];
    const void *weight_ptr[MAX_INPUT_INTERPOLATE];
    const int *index;
    void *dst;
    size_t work_amount;
    size_t oc_off;
    //ptr to array of post op inputs pointers (flat list)
    const void* post_op_data;
};

struct jit_uni_interpolate_kernel {
    void (*ker_)(const jit_interpolate_call_args *);

    void operator()(const jit_interpolate_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_interpolate_kernel(jit_interpolate_config_params jcp, const dnnl_primitive_attr &attr) : ker_(nullptr), jcp_(jcp), attr_(attr) {}
    virtual ~jit_uni_interpolate_kernel() {}

    virtual void create_ker() = 0;

    jit_interpolate_config_params jcp_;
    const dnnl_primitive_attr &attr_;
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
    Interpolate(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override;
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
    // Some FEs or preprocessing step resize spatial dimension for tensor with NHWC layout memory,
    // but imported as planar layout[abcd] with axis[1,2] for convenience. In this case, for pillow modes without pad for now,
    // nhwc layout path and the kernel(nhwc layout executor) can be used for this planar layout and axis settings(NCHWAsNHWC is true) to get higher perf with
    // 1. logical shape alignment [abcd-nhwc] to [adbc-nchw].
    // 2. axis alignment [1,2] to [2,3].
    // 3. config planar layout support and treated it as channel_first layout.
    bool NCHWAsNHWC = false;
    size_t dataRank = 0;

    class InterpolateExecutorBase {
        public:
            InterpolateExecutorBase(const InterpolateAttrs& interpAttrs,
                                const VectorDims &srcDims,
                                const VectorDims &dstDims,
                                const std::vector<float> &dataScales);

            virtual void exec(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_) = 0;
            virtual ~InterpolateExecutorBase() = default;
            VectorDims getSrcDimPad5d() const { return srcDimPad5d; }

        private:
            void buildTblNN(const VectorDims& srcDimPad5d, const VectorDims& dstDim5d, const std::vector<float>& dataScales,
                            InterpolateLayoutType layout, InterpolateNearestMode nearestMode);
            void buildTblLinearOnnx(const VectorDims& srcDimPad5d, const VectorDims& dstDim5d, const std::vector<float>& dataScales,
                                    InterpolateLayoutType layout);
            void buildTblLinear(const VectorDims& srcDimPad5d, const VectorDims& dstDim5d, const std::vector<float>& dataScales, int kernel_width,
                                bool antialias);
            void buildTblCubic(const VectorDims& srcDimPad5d, const VectorDims& dstDim5d, const std::vector<float>& dataScales, float cubicCoeff,
                               InterpolateLayoutType layout);
            void buildTblPillow(const VectorDims& srcDimPad5d, const VectorDims& dstDim5d, const std::vector<float>& dataScales,
                                float cubicCoeff, InterpolateLayoutType layout);

            float coordTransToInput(int outCoord, float scale, int inShape, int outShape) const;
            int nearestRound(float origin, bool isDownsample, InterpolateNearestMode nearestMode) const;
            void linearOnnxCF(int outCoord, float scale, int inShape, int outShape, int& index0, int& index1, float& weight0, float& weight1);
            std::vector<float> getCubicCoeffs(float mantissa, float a);
            static float getPillowBilinearCoeffs(float m);
            static float getPillowBicubicCoeffs(float m);
            inline void create_pillow_working_buf(InterpolateLayoutType layout);

        protected:
            InterpolateMode mode;
            InterpolateCoordTransMode coordTransMode;
            InterpolateLayoutType configured_for_layout;
            VectorDims srcDimPad5d, dstDim5d;
            ov::element::Type inputPrec, outputPrec;
            size_t srcDataSize, dstDataSize;
            int spatialDimSize;
            size_t dataRank;
            std::vector<int> auxTable;
            std::vector<uint8_t> pillow_working_buf;
            size_t m_threads_num = 0lu;
    };
    std::shared_ptr<InterpolateExecutorBase> execPtr = nullptr;

    class InterpolateJitExecutor : public InterpolateExecutorBase {
        public:
            InterpolateJitExecutor(const InterpolateAttrs& interpAttrs,
                                   const VectorDims &srcDims,
                                   const VectorDims &dstDims,
                                   const std::vector<float> &dataScales,
                                   const dnnl::primitive_attr &attr);

            void exec(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_) override;

        private:
            // nearest neighbor
            void NNPlanar(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_,
                int B, int C, int ID, int IH, int IW, int OD, int OH, int OW);
            void NNCGathered(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_,
                int B, int C, int ID, int IH, int IW, int OD, int OH, int OW);

            // onnx linear
            void linearOnnxPlanar(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_,
                int B, int C, int ID, int IH, int IW, int OD, int OH, int OW);
            void linearOnnxCGathered(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_,
                int B, int C, int ID, int IH, int IW, int OD, int OH, int OW);

            // cubic
            void cubicPlanar(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_,
                int B, int C, int IH, int IW, int OH, int OW);
            void cubicCGathered(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_,
                int B, int C, int IH, int IW, int OH, int OW);

            // pillow bilinear and pillow bicubic
            void pillowCGathered(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_,
                int B, int C, int IH, int IW, int OH, int OW);

        private:
            std::shared_ptr<jit_uni_interpolate_kernel> interpolateKernel = nullptr;
    };

    class InterpolateRefExecutor : public InterpolateExecutorBase {
        public:
            InterpolateRefExecutor(const InterpolateAttrs& interpAttrs,
                                   const VectorDims &srcDims,
                                   const VectorDims &dstDims,
                                   const std::vector<float> &_dataScales) :
                InterpolateExecutorBase(interpAttrs, srcDims, dstDims, _dataScales),
                antialias(interpAttrs.antialias), dataScales(_dataScales) {}

            void exec(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_) override;

        private:
            void NNRef(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int ID, int IH, int IW, int OD, int OH, int OW);
            void linearOnnxRef(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int ID, int IH, int IW, int OD, int OH, int OW);

            void cubicRef(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int IH, int IW, int OH, int OW);
            void linearInterpolation(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int ID, int IH, int IW,
                                      float fx, float fy, float fz, int OD, int OH, int OW, int kernel_width, bool antialias);
            void pillowRef(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int IH, int IW, int OH, int OW);

            static float getValue(const uint8_t *base, size_t offset, ov::element::Type prec);
            static void setValue(uint8_t *base, size_t offset, float value, ov::element::Type prec);

        private:
            bool antialias;
            std::vector<float> dataScales;
    };

    void setPostOps(dnnl::primitive_attr &attr, const VectorDims &dims);

    static VectorDims getPaddedInputShape(const VectorDims &srcDims, const std::vector<int> &padBegin, const std::vector<int> &padEnd);
    std::vector<float> getScales(const VectorDims &srcDimPad, const VectorDims &dstDim);
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

    std::string errorPrefix;

    bool canUseAclExecutor = false;
    std::shared_ptr<InterpolateExecutor> aclExecPtr = nullptr;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
