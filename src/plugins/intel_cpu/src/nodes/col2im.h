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
#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"

namespace ov::intel_cpu::node {

class Col2Im : public Node {
public:
    Col2Im(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override;
    [[nodiscard]] bool needPrepareParams() const override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

private:
    template <class OV_DATA_TYPE, class OV_INDEX_TYPE>
    void executeImpl();

    template <typename T>
    struct Col2ImExecute;

    ov::Strides strides;
    ov::Strides dilations;
    ov::Shape padsBegin;
    ov::Shape padsEnd;
};

}  // namespace ov::intel_cpu::node
