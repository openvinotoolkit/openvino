// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "cpu_types.h"
#include "graph_context.h"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

class AdaptivePooling : public Node {
public:
    AdaptivePooling(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    int spatialDimsCount;
    mutable std::vector<Dim> spatialDimsValue;
    ov::element::Type precision = ov::element::f32;
    static inline void setBinBorders(size_t* startPtr,
                                     size_t* endPtr,
                                     size_t idx,
                                     size_t inputLength,
                                     size_t outputLength);

protected:
    bool needShapeInfer() const override;
    bool needPrepareParams() const override {
        return false;
    };
    void executeDynamicImpl(const dnnl::stream& strm) override;
};

}  // namespace ov::intel_cpu::node
