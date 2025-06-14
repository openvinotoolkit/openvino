// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov::intel_cpu::node {

class SegmentMax : public Node {
public:
    SegmentMax(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override;
    [[nodiscard]] bool needPrepareParams() const override;
    void executeDynamicImpl(const dnnl::stream& strm) override;
    [[nodiscard]] bool needShapeInfer() const override;

private:
    template <class OV_DATA_TYPE>
    void executeImpl();

    template <class T>
    struct SegmentMaxExecute;

    ov::op::FillMode fillMode;
    std::vector<int32_t> lastSegmentIds;
    std::vector<int32_t> lastNumSegments;
};

}  // namespace ov::intel_cpu::node
