// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class ROIAlignRotated : public Node {
public:
    ROIAlignRotated(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;
    bool needPrepareParams() const override;
    void executeDynamicImpl(dnnl::stream strm) override;

private:
    template <ov::element::Type_t OV_TYPE>
    void executeImpl();

    int pooledH;
    int pooledW;
    int samplingRatio;
    float spatialScale;
    bool clockwiseMode;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
