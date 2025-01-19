// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common/permute_kernel.h"
#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class SpaceToDepth : public Node {
public:
    SpaceToDepth(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

    void prepareParams() override;

    enum Mode { BLOCKS_FIRST = 0, DEPTH_FIRST = 1 };

    struct SpaceToDepthAttrs {
        LayoutType layoutType;
        Mode mode;
        size_t blockSize = 0lu;
        size_t blockStep = 1lu;
        size_t dataSize = 1lu;
        size_t nSpatialDims = 0lu;
        VectorDims srcBlockedDims;
        VectorDims destBlockedDims;
        size_t hash() const;
        bool operator==(const SpaceToDepthAttrs& rhs) const;
    };

protected:
    void executeDynamicImpl(const dnnl::stream& strm) override;

private:
    SpaceToDepthAttrs attrs;

    struct SpaceToDepthExecutor final {
        SpaceToDepthExecutor(const SpaceToDepthAttrs& attrs);
        void exec(const uint8_t* srcData, uint8_t* dstData, const int MB);
        ~SpaceToDepthExecutor() = default;

    private:
        std::unique_ptr<PermuteKernel> permuteKernel;
    };
    using executorPtr = std::shared_ptr<SpaceToDepthExecutor>;
    executorPtr execPtr = nullptr;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
