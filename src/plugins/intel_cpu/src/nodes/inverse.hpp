// Copyright (C) 2018-2025 Intel Corporation
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

class Inverse : public Node {
public:
    Inverse(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;

    [[nodiscard]] bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    void prepareParams() override;

    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;
    [[nodiscard]] bool canBeInPlace() const override {
        return false;
    }

private:
    /// Inverse params
    bool m_adjoint = false;

    /// Shape inference
    static constexpr size_t INPUT_PORT = 0LU;
    static constexpr size_t OUTPUT_PORT = 0LU;
    bool m_const_input = false;

    /// General algorithm variables
    ov::element::Type m_input_precision = ov::element::f32;

    size_t m_side = 0;
    size_t m_side_squared = 0;
    size_t m_batches_count = 0;

    // Helper functions
    void inverse();

    void lu_decomposition(const float* data,
                          std::vector<float>& L,
                          std::vector<float>& U,
                          std::vector<size_t>& P,
                          size_t b) const;

    void lu_solve(float* output, std::vector<float>& L, std::vector<float>& U, std::vector<size_t>& P, size_t b) const;
};

}  // namespace ov::intel_cpu::node
