// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class Inverse : public Node {
public:
    Inverse(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;

    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    void prepareParams() override;

    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override;
    bool canBeInPlace() const override {
        return false;
    }

private:
    /// Inverse params
    bool m_adjoint = false;

    /// Shape inference
    static constexpr size_t INPUT_PORT = 0lu;
    static constexpr size_t OUTPUT_PORT = 0lu;
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
                          size_t b);

    void lu_solve(float* output, std::vector<float>& L, std::vector<float>& U, std::vector<size_t>& P, size_t b);
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
