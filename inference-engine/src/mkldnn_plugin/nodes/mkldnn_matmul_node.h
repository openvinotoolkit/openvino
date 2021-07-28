// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNMatMulNode : public MKLDNNNode {
public:
    MKLDNNMatMulNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void initOptimalPrimitiveDescriptor() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    int getMaxBatch() override;

    InferenceEngine::Precision getRuntimePrecision() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    float alpha = 1.f;
    float beta = 0.f;
    bool transposeA = false;
    bool transposeB = false;

    int xAxis = 0;
    int yAxis = 0;

    std::vector<int> aOffsets;
    std::vector<int> bOffsets;
    std::vector<int> cOffsets;

    InferenceEngine::Precision runtimePrecision;

    template<typename T0, typename T1> inline void process_data();

    std::string errorPrefix;

    struct {
        MKLDNNMemoryPtr src0_mem_ptr = nullptr;
        MKLDNNMemoryPtr src1_mem_ptr = nullptr;
        MKLDNNMemoryPtr dst_mem_ptr = nullptr;

        char transa = 'N';
        char transb = 'N';

        int MB1 = 1;
        int MB2 = 1;

        int M = 0;
        int N = 0;
        int K = 0;

        int lda = 0;
        int ldb = 0;
        int ldc = 0;

        int shift1 = 0;
        int shift2 = 0;

        size_t ndims = 0;
    } params;
};

}  // namespace MKLDNNPlugin

