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
    template<typename T> void optimizedExecute(const int MB, const MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr);

    InferenceEngine::SizeVector order;
    InferenceEngine::Precision prec;
    bool isOptimized = false;

    const std::vector<std::vector<size_t>> optimizedOrders = {
            std::vector<size_t>{0, 3, 1, 2},
            std::vector<size_t>{0, 4, 1, 2, 3},
            std::vector<size_t>{0, 5, 1, 2, 3, 4},
    };

    std::unique_ptr<PermuteKernel> permuteKernel;

    struct TransposeContext {
        MKLDNNTransposeNode* nodePtr;
        MKLDNNMemoryPtr srcMemPtr;
        MKLDNNMemoryPtr dstMemPtr;
        int MB;
    };

    template<typename T>
    struct TransposeOptimizedEmitter {
        void operator()(TransposeContext& ctx) {
            ctx.nodePtr->optimizedExecute<T>(ctx.MB, ctx.srcMemPtr, ctx.dstMemPtr);
        }
    };
};

}  // namespace MKLDNNPlugin

