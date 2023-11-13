// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <string>
#include <memory>
#include <vector>
#include <dnnl_extension_utils.h>

namespace ov {
namespace intel_cpu {
namespace node {

class AdaptivePooling : public Node {
public:
    AdaptivePooling(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    int spatialDimsCount;
    mutable std::vector<Dim> spatialDimsValue = {};
    ov::element::Type precision = ov::element::f32;
    inline void setBinBorders(size_t *startPtr, size_t *endPtr, size_t idx, size_t inputLength, size_t outputLength);

    std::string errorPrefix;

protected:
    bool needShapeInfer() const override;
    bool needPrepareParams() const override { return false; };
    void executeDynamicImpl(dnnl::stream strm) override;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
