// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/expressions/buffer_expression.hpp"

namespace ov::intel_cpu {

class RepackedWeightsBufferExpression : public snippets::lowered::BufferExpression {
    friend class snippets::lowered::ExpressionFactory;

public:
    OPENVINO_RTTI("RepackedWeightsBufferExpression", "0", BufferExpression)
    RepackedWeightsBufferExpression() = default;

    void validate() const override;
    void init_allocation_size(const std::shared_ptr<snippets::lowered::LoopManager>& loop_manager,
                              size_t allocation_rank) override;

private:
    RepackedWeightsBufferExpression(const std::shared_ptr<ov::Node>& n,
                                    const std::shared_ptr<snippets::IShapeInferSnippetsFactory>& factory);

    snippets::lowered::ExpressionPtr clone() const override;
};

class CompensationsBufferExpression : public snippets::lowered::BufferExpression {
    friend class snippets::lowered::ExpressionFactory;

public:
    OPENVINO_RTTI("CompensationsBufferExpression", "0", BufferExpression)
    CompensationsBufferExpression() = default;

    void validate() const override;
    void init_allocation_size(const std::shared_ptr<snippets::lowered::LoopManager>& loop_manager,
                              size_t allocation_rank) override;

private:
    CompensationsBufferExpression(const std::shared_ptr<ov::Node>& n,
                                  const std::shared_ptr<snippets::IShapeInferSnippetsFactory>& factory);

    snippets::lowered::ExpressionPtr clone() const override;
};

}  // namespace ov::intel_cpu
