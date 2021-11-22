// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <memory>
#include <string>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNDeconvolutionNode : public MKLDNNNode {
public:
    MKLDNNDeconvolutionNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                          const std::vector<MemoryDescPtr>& outputDesc) override;
    void createPrimitive() override;
    void filterSupportedPrimitiveDescriptors() override;
    void filterSupportedDescriptors();
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

    size_t descInputNumbers(MKLDNNDescriptor desc) override {
        return static_cast<size_t>(getParentEdges().size());
    }

    std::shared_ptr<MemoryDesc> getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;
    std::shared_ptr<MemoryDesc> getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;

    const mkldnn::memory& getWeights() const;

    InferenceEngine::Precision getRuntimePrecision() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    bool canFuse(const MKLDNNNodePtr& node) const override;

    const InferenceEngine::SizeVector& getWeightDims() { return weightDims; }
    const std::vector<ptrdiff_t>& getStride() { return stride; }

    void prepareParams() override;
    void executeDynamicImpl(mkldnn::stream strm) override;
    bool needShapeInfer() const override;
    std::vector<VectorDims> shapeInfer() const override;
    VectorDims deconvShapeInfer(const VectorDims &inDims) const;
    void initPadding(const std::shared_ptr<ngraph::Node> op);

private:
    bool withGroups = false;
    bool isDW = false;
    bool isInt8 = false;
    bool autoPad = false;
    bool withOutputShape = false;
    size_t groupNum = 1;
    size_t IC;
    size_t OC;
    std::vector<ptrdiff_t> kernel;
    std::vector<ptrdiff_t> stride;
    std::vector<ptrdiff_t> dilation;
    std::vector<ptrdiff_t> paddingL;
    std::vector<ptrdiff_t> paddingR;
    mutable std::vector<int32_t> outSpatialDims;
    VectorDims weightDims;

    AttrPtr pAttr;

    mkldnn::reorder reorderSrc;
    mkldnn::reorder reorderWgh;
    mkldnn::reorder reorderDst;
    MKLDNNMemoryPtr srcPlanarMemPtr;
    MKLDNNMemoryPtr wghPlanarMemPtr;
    MKLDNNMemoryPtr dstPlanarMemPtr;

    mkldnn::primitive_attr attr;
    void setPostOps(mkldnn::primitive_attr &attr, const VectorDims &dims);

    void initPaddingR(const Shape &inShape, const Shape &outShape);

    using DefaultDeconvDescs = std::pair<std::shared_ptr<mkldnn::convolution_backward_data::desc>,
                                         std::shared_ptr<mkldnn::convolution_forward::primitive_desc>>;
    DefaultDeconvDescs createDescriptorInternalDefault(const mkldnn::memory::desc& in_candidate,
                                                       const mkldnn::memory::desc& wgh_candidate,
                                                       const mkldnn::memory::desc& out_candidate,
                                                       mkldnn::algorithm alg) const;
    std::shared_ptr<mkldnn::deconvolution_forward::desc> createDescriptorInternalInt8(const mkldnn::memory::desc& in_candidate,
                                                                                      const mkldnn::memory::desc& wgh_candidate,
                                                                                      const mkldnn::memory::desc& out_candidate,
                                                                                      mkldnn::algorithm alg) const;
    std::shared_ptr<MKLDNNDescriptor> createMkldnnDeconvDesc(const mkldnn::memory::desc& srcDesc,
                                                             const mkldnn::memory::desc& wghDesc,
                                                             const mkldnn::memory::desc& dstDesc,
                                                             bool isWinograd) const;

    void createDeconvPrim(std::shared_ptr<MKLDNNDescriptor> desc,
                          MKLDNNMemoryPtr srcMemPtr,
                          MKLDNNMemoryPtr wghMemPtr,
                          MKLDNNMemoryPtr dstMemPtr,
                          AttrPtr attr,
                          impl_desc_type selectedImpl,
                          bool forceGemm = false);

    std::string errorPrefix;

    bool canBeExecutedInInt8() const;
    InferenceEngine::Blob::Ptr createWeiBlobAsIO(InferenceEngine::SizeVector dims);
};

}  // namespace MKLDNNPlugin

