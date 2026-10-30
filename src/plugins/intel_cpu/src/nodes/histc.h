// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

class Histc : public Node {
public:
    Histc(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override {}
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override;
    [[nodiscard]] bool needPrepareParams() const override {
        return false;
    }
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    // Empty input still produces a zero-filled histogram.
    [[nodiscard]] bool isExecutable() const override {
        return true;
    }
    [[nodiscard]] bool neverExecute() const override {
        return false;
    }

private:
    template <class T>
    void executeImpl();

    double m_min = 0.0;
    double m_max = 0.0;
};

}  // namespace ov::intel_cpu::node
