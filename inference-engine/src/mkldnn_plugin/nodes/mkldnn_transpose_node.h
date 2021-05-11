// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <vector>
#include <utility>
#include <map>
#include <memory>
#include "common/permute_kernel.h"

namespace MKLDNNPlugin {

class MKLDNNTransposeNode : public MKLDNNNode {
public:
    MKLDNNTransposeNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

    const InferenceEngine::SizeVector& getOrder() const {
        return order;
    }

private:
    InferenceEngine::SizeVector order;
    InferenceEngine::Precision prec;

    typedef std::function<void(int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr)> transposeImpl;
    typedef std::function<bool(int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr)> isApplicable;
    struct TransposeImpl {
        TransposeImpl(transposeImpl f0, isApplicable f1): execute(std::move(f0)), isValidParams(std::move(f1)) {}

        transposeImpl execute;
        isApplicable isValidParams;
    };

    static const std::multimap<InferenceEngine::SizeVector, TransposeImpl> OptimizedCases;
    std::unique_ptr<PermuteKernel> permuteKernel;
};

}  // namespace MKLDNNPlugin

