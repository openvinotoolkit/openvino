// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <precision_utils.h>
#include <ie_blob.h>

#include <iomanip> // std::setw

#include <vpu/utils/ie_helpers.hpp>
#include <graph_transformer/include/vpu/model/data_desc.hpp>

typedef std::map<std::string, std::string> ParamsStruct;

typedef float (*eltwise_kernel)(float a, float b, float c);

struct param_size {
    size_t x;
    size_t y;
};

struct param_size_3d {
    size_t x;
    size_t y;
    size_t z;
};

struct paddings4 {
    size_t left;
    size_t top;
    size_t right;
    size_t bottom;
};

struct tensor_test_params {
    size_t n;
    size_t c;
    size_t h;
    size_t w;
    friend std::ostream& operator<<(std::ostream& os, tensor_test_params const& tst)
    {
        return os << "tensor (" << tst.n
                  << ", " << tst.c
                  << ", " << tst.h
                  << ", " << tst.w
                  << ")";
    }
    InferenceEngine::SizeVector asVector() const { return {n,c,h,w};}
};

struct tensor_test_params_3d {
    size_t n;
    size_t c;
    size_t d;
    size_t h;
    size_t w;
    friend std::ostream& operator<<(std::ostream& os, tensor_test_params_3d const& tst)
    {
        return os << "tensor (" << tst.n
                  << ", " << tst.c
                  << ", " << tst.d
                  << ", " << tst.h
                  << ", " << tst.w
                  << ")";
    }
    InferenceEngine::SizeVector asVector() const { return {n,c,d,h,w};}
};

/* Wrappers to gen subnets:
 reference function signature should have following structure:
    input blob,
    output blob,
    pointer to weights (if they are required)
    weights number (if pointer to weights is set)
    bias number (if pointer to weights is set)
    other parameters

*/
static inline void PrintTo(const param_size& sz, std::ostream* os) {
    *os << "{" << std::setw(2) << sz.x << ", " << std::setw(2) << sz.y << "}";
};

template <typename DataType>
class IReduceKernel
{
public:
    virtual void init() = 0;
    virtual void accumulate(const DataType& val) = 0;
    virtual DataType result() const = 0;
    virtual DataType copy(const DataType& val) const = 0;
};

void ref_innerproduct_wrap(const InferenceEngine::Blob::Ptr src,
                      InferenceEngine::Blob::Ptr dst,
                      const uint16_t *weights,
                      size_t weightsSize,
                      const uint16_t *biases,
                      size_t biasSize,
                      const ParamsStruct& params);

void ref_ReLU_wrap(const InferenceEngine::Blob::Ptr inTensor,
                   InferenceEngine::Blob::Ptr outTensor,
                   const ParamsStruct& params);

void ref_Clamp_wrap(const InferenceEngine::Blob::Ptr inTensor,
                   InferenceEngine::Blob::Ptr outTensor,
                   const ParamsStruct& params);

void ref_pooling_wrap(const InferenceEngine::Blob::Ptr src,
                    InferenceEngine::Blob::Ptr dst,
                    const ParamsStruct& params);

void ref_copy_wrap(const InferenceEngine::Blob::Ptr src,
                   InferenceEngine::Blob::Ptr dst,
                   const ParamsStruct& params);

void ref_convolution_wrap(const InferenceEngine::Blob::Ptr src,
                          InferenceEngine::Blob::Ptr dst,
                          const uint16_t *weights,
                          size_t weightsSize,
                          const uint16_t *biases,
                          size_t biasSize,
                          const ParamsStruct& params);

void ref_deconvolution_wrap(const InferenceEngine::Blob::Ptr src,
                          InferenceEngine::Blob::Ptr dst,
                          const uint16_t *weights,
                          size_t weightsSize,
                          const uint16_t *biases,
                          size_t biasSize,
                          const ParamsStruct& params);

void ref_tanh_wrap(const InferenceEngine::Blob::Ptr inTensor,
                   InferenceEngine::Blob::Ptr outTensor,
                   const ParamsStruct& params);

void ref_sigmoid_wrap(const InferenceEngine::Blob::Ptr src,
                      InferenceEngine::Blob::Ptr dst,
                      const ParamsStruct& params);

void ref_PReLU_wrap(const InferenceEngine::Blob::Ptr src,
                    InferenceEngine::Blob::Ptr dst,
                    const uint16_t *weights,
                    size_t weightsSize,
                    const uint16_t *biases,
                    size_t biasSize,
                    const ParamsStruct& params);

void ref_RegionYolo_wrap(InferenceEngine::Blob::Ptr inTensor,
              InferenceEngine::Blob::Ptr outTensor,
              const ParamsStruct& params);

void ref_reshape_wrap(const InferenceEngine::Blob::Ptr src,
                      InferenceEngine::Blob::Ptr dst,
                      const ParamsStruct& params);

void ref_permute_wrap(const InferenceEngine::Blob::Ptr src,
                 InferenceEngine::Blob::Ptr dst,
                 const ParamsStruct& params);

void ref_log_wrap(const InferenceEngine::Blob::Ptr& src,
                  InferenceEngine::Blob::Ptr& dst,
                  const ParamsStruct& params);

void ref_exp_wrap(const InferenceEngine::Blob::Ptr& src,
                  InferenceEngine::Blob::Ptr& dst,
                  const ParamsStruct& params);

void ref_convert_wrap(const InferenceEngine::Blob::Ptr src,
                      InferenceEngine::Blob::Ptr dst,
                      const ParamsStruct& params);

/* Original functions*/

void ref_innerproduct(const InferenceEngine::Blob::Ptr src,
                      InferenceEngine::Blob::Ptr dst,
                      const uint16_t *weights,
                      size_t weightsSize,
                      const uint16_t *biases,
                      size_t biasSize,
                      uint32_t OC);

void ref_convolution(const InferenceEngine::Blob::Ptr src,
                     InferenceEngine::Blob::Ptr dst,
                     const InferenceEngine::ie_fp16* weights_data,
                     const InferenceEngine::ie_fp16* bias_data,
                     param_size kernel,
                     param_size stride,
                     param_size pad,
                     size_t group,
                     param_size dilation = {1, 1});

void ref_maxPooling(const InferenceEngine::Blob::Ptr src,
                    InferenceEngine::Blob::Ptr dst,
                    param_size kernel,
                    param_size stride,
                    param_size pad,
                    bool exclude_pad = false);

void ref_avgPooling(const InferenceEngine::Blob::Ptr src,
                    InferenceEngine::Blob::Ptr dst,
                    param_size kernel,
                    param_size stride,
                    param_size pad,
                    bool exclude_pad = false);

void ref_ReLU(const InferenceEngine::Blob::Ptr inTensor,
              InferenceEngine::Blob::Ptr outTensor,
              float negative_slope);

void ref_copy(const InferenceEngine::Blob::Ptr src,
              InferenceEngine::Blob::Ptr dst);

void ref_tanh(const InferenceEngine::Blob::Ptr src,
              InferenceEngine::Blob::Ptr dst);

void ref_sigmoid(const InferenceEngine::Blob::Ptr src,
                 InferenceEngine::Blob::Ptr dst);

void ref_PReLU(const InferenceEngine::Blob::Ptr src,
               InferenceEngine::Blob::Ptr dst,
               const uint16_t *weights,
               size_t weightsSize);

void ref_eltwise(const InferenceEngine::Blob::Ptr src1,
                const InferenceEngine::Blob::Ptr src2,
                const InferenceEngine::Blob::Ptr src3,
                InferenceEngine::Blob::Ptr dst,
                eltwise_kernel fun, std::vector<float> coeff);

void ref_RegionYolo(const InferenceEngine::Blob::Ptr src,
                    InferenceEngine::Blob::Ptr dst,
                    int coords,
                    int classes,
                    int num,
                    int maskSize,
                    int doSoftMax);

template <typename T>
void ref_Permute(const InferenceEngine::Blob::Ptr src,
                 InferenceEngine::Blob::Ptr dst,
                 std::vector<size_t> order);

void ref_softMax(const InferenceEngine::Blob::Ptr& src,
                  InferenceEngine::Blob::Ptr& dst,
                  int axis);
void ref_reshape(const InferenceEngine::Blob::Ptr src,
                 InferenceEngine::Blob::Ptr dst);

void ref_Clamp(const InferenceEngine::Blob::Ptr inTensor,
              InferenceEngine::Blob::Ptr outTensor,
              float min,
              float max);

void ref_log(const InferenceEngine::Blob::Ptr& src,
             InferenceEngine::Blob::Ptr& dst);

void ref_exp(const InferenceEngine::Blob::Ptr& src,
             InferenceEngine::Blob::Ptr& dst);

void ref_gather(const InferenceEngine::Blob::Ptr& srcIdx,
                const InferenceEngine::Blob::Ptr& srcDct,
                const InferenceEngine::Blob::Ptr& dst,
                const                        int  axis);

void ref_scatter_elements_update(InferenceEngine::Blob::Ptr& input,
                                 InferenceEngine::Blob::Ptr& indices,
                                 InferenceEngine::Blob::Ptr& updates,
                                                  const int  axis,
                                 InferenceEngine::Blob::Ptr& output);

template<typename DataType>
void ref_reduce(const InferenceEngine::Blob::Ptr& src,
                const InferenceEngine::Blob::Ptr& axes,
                InferenceEngine::Blob::Ptr& dst,
                int keep_dims,
                vpu::LayoutPreference layoutPreference,
                IReduceKernel<DataType>* op);

void ref_topk(const InferenceEngine::Blob::Ptr& srcValues,
              const InferenceEngine::Blob::Ptr& srcK,
              InferenceEngine::Blob::Ptr dstValues,
              InferenceEngine::Blob::Ptr dstIndices,
              int axis,
              const std::string& mode,
              const std::string& sort);

void ref_strided_slice(const InferenceEngine::Blob::Ptr& src,
                       InferenceEngine::Blob::Ptr& dst,
                       InferenceEngine::SizeVector &out_dims,
                       const std::vector<int32_t>& begin,
                       const std::vector<int32_t>& end,
                       const std::vector<int32_t>& stride,
                       const InferenceEngine::SizeVector& begin_mask,
                       const InferenceEngine::SizeVector& end_mask);

struct ExpDetectionOutputParams {
    float   deltas_weights[4];
    float   max_delta_log_wh;
    float   nms_threshold;
    float   score_threshold;
    int32_t max_detections_per_image;       // int
    int32_t num_classes;                    // int
    int32_t post_nms_count;                 // int
    int32_t class_agnostic_box_regression;  // bool
};

void ref_expDetectionOutput(const InferenceEngine::Blob::Ptr srcBoxes,   // [numRois][4]
                            const InferenceEngine::Blob::Ptr srcDeltas,  // [numRois]([numClasses][4])
                            const InferenceEngine::Blob::Ptr srcScores,  // [numRois][numClasses]
                            const InferenceEngine::Blob::Ptr srcIMinfo,  // [2]
                            InferenceEngine::Blob::Ptr dstBoxes,         // [maxDetections][4]
                            InferenceEngine::Blob::Ptr dstClasses,       // [maxDetections]
                            InferenceEngine::Blob::Ptr dstScores,        // [maxDetections]
                            const int numRois,
                            const int numClasses,
                            const int maxDetections,
                            const ExpDetectionOutputParams& layerParams);

void ref_ROIFeatureExtractor(std::vector<InferenceEngine::Blob::Ptr> inputs,
                             InferenceEngine::Blob::Ptr output,
                             InferenceEngine::Blob::Ptr output_rois,
                             std::vector<int> pyramid_scales,
                             int sampling_ratio,
                             int pooled_height,
                             int pooled_width);

void ref_ROIAlign(InferenceEngine::Blob::Ptr feature_map,
                  InferenceEngine::Blob::Ptr rois,
                  InferenceEngine::Blob::Ptr batch_indices,
                  InferenceEngine::Blob::Ptr output,
                  const int sampling_ratio,
                  const int pooled_h,
                  const int pooled_w,
                  const int num_rois,
                  const float spatial_scale,
                  const std::string mode);

void ref_convert(const InferenceEngine::Blob::Ptr &src,
                 InferenceEngine::Blob::Ptr &dst);

void ref_Split(const InferenceEngine::Blob::Ptr src,
               const InferenceEngine::BlobMap& dst,
               const int axis);

void ref_ExpPriorGridGenerator(std::vector<InferenceEngine::Blob::Ptr> inputs,
                               std::vector<InferenceEngine::Blob::Ptr> output,
                               int grid_w,
                               int grid_h,
                               float stride_w,
                               float stride_h);

void ref_ExpGenerateProposals(std::vector<InferenceEngine::Blob::Ptr> inputs,
                              std::vector<InferenceEngine::Blob::Ptr> output,
                              float min_size_,
                              float nms_threshold_,
                              int post_nms_topn_,
                              int pre_nms_topn_);

void ref_ExpTopKROIs(std::vector<InferenceEngine::Blob::Ptr> inputs,
                     std::vector<InferenceEngine::Blob::Ptr> output,
                     int max_rois);

void ref_nonZero(const InferenceEngine::Blob::Ptr& src,
                 InferenceEngine::Blob::Ptr& outIndices,
                 InferenceEngine::Blob::Ptr& outDims);

static constexpr char const PRELU_PARAM[] = "channel_shared";
