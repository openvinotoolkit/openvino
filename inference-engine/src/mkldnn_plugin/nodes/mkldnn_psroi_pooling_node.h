// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNPSROIPoolingNode : public MKLDNNNode {
public:
    MKLDNNPSROIPoolingNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {};
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    size_t outputDim = 0;
    size_t groupSize = 0;
    float spatialScale = 0;
    size_t pooledHeight = 0;
    size_t pooledWidth = 0;
    size_t spatialBinsX = 0;
    size_t spatialBinsY = 0;
    std::string mode = "";

    int channels = 0;
    int height = 0;
    int width = 0;

    int nn = 0;
    int nc = 0;
    int nh = 0;
    int nw = 0;

    // for Deformable PSROIPolling
    bool noTrans;
    int partSize;
    float transStd;

    std::string errorPrefix;

    void unpackParams(const BlockedMemoryDesc& srcDesc, const BlockedMemoryDesc& dstDesc,
                      int& hInputStride, int& wInputStride,
                      int& hOutputStride, int& wOutputStride,
                      int& inBlockSize, int& outBlockSize,
                      int& outBlockCount,
                      unsigned long& inputChannelsPadding, unsigned long& outputChannelsPadding);

    template <typename inputType, typename outputType>
    void executeAverage(const inputType *srcData, outputType *dstData, const float *bottomRois,
                        const int n, const int roiBatchInd,
                        const BlockedMemoryDesc& srcDesc, const BlockedMemoryDesc& dstDesc);

    template <typename inputType, typename outputType>
    void executeBilinear(const inputType *srcData, outputType *dstData, const float *bottomRois,
                         const int currentRoi, const int roiBatchInd,
                         const BlockedMemoryDesc& srcDesc, const BlockedMemoryDesc& dstDesc);

    template <typename inputType, typename outputType>
    void executeBilinearDeformable(const inputType *srcData, outputType *dstData, const float *bottomRois,
                                   const float *bottomTrans, const int numClasses, const int channelsEachClass,
                                   const int currentRoi, const int roiBatchInd);

    template <typename inputType, typename outputType>
    void executeSpecified();

    template<typename T>
    struct PSROIPoolingExecute;
};

}  // namespace MKLDNNPlugin
