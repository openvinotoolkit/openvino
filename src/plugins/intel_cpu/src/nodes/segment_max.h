// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class SegmentMax : public Node {
public:
    SegmentMax(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;
    bool needPrepareParams() const override;
    void executeDynamicImpl(const dnnl::stream& strm) override;
    bool needShapeInfer() const override;

private:
    template <class OV_DATA_TYPE>
    void executeImpl();

    template <class T>
    struct SegmentMaxExecute;

    ov::op::FillMode fillMode;
    std::vector<int32_t> lastSegmentIds;
    std::vector<int32_t> lastNumSegments;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
