// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "graph_context.h"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

class SparseFillEmptyRows : public Node {
public:
    SparseFillEmptyRows(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    bool created() const override;
    bool needPrepareParams() const override;
    bool isExecutable() const override;
    void executeDynamicImpl(const dnnl::stream& strm) override;
    void execute(const dnnl::stream& strm) override;

private:
    template <typename T>
    void executeImpl();

    template <typename T>
    struct SparseFillEmptyRowsExecute;
};

}  // namespace ov::intel_cpu::node
