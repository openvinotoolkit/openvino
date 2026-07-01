// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

class Bincount : public Node {
public:
    Bincount(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

    void prepareParams() override;

    [[nodiscard]] bool needShapeInfer() const override {
        return false;
    }

    // The output size depends on the "data" values (via minlength) and not only on its shape, so the node must
    // still be executed even when the "data" input is empty (e.g. minlength > 0 yields a non-empty output for a
    // zero-element input). The default Node::isExecutable() would skip execute() for empty inputs, leaving the
    // dynamic output shape unresolved.
    [[nodiscard]] bool isExecutable() const override {
        return isDynamicNode() || Node::isExecutable();
    }

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    static constexpr size_t INPUT_DATA_PORT = 0;
    static constexpr size_t INPUT_WEIGHTS_PORT = 1;
    static constexpr size_t OUTPUT_PORT = 0;

    size_t get_output_size() const;

    int64_t m_minlength = 0;
    bool m_has_weights = false;

    ov::element::Type m_data_precision;
    ov::element::Type m_weights_precision;
    ov::element::Type m_output_precision;
};

}  // namespace ov::intel_cpu::node
