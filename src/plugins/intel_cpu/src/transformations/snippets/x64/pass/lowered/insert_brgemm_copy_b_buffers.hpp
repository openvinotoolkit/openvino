// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/pass.hpp"

#include "snippets/lowered/expressions/buffer_expression.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {

/**
 * @interface InsertBrgemmCopyBBuffers
 * @brief Insert Buffers after BrgemmCopyB with algorithm of allocation size calculation which
 *        distinguishes with common algorithm
 * @ingroup snippets
 */
class InsertBrgemmCopyBBuffers: public snippets::lowered::pass::RangedPass {
public:
    InsertBrgemmCopyBBuffers() = default;
    OPENVINO_RTTI("InsertBrgemmCopyBBuffers", "Pass");
    bool run(snippets::lowered::LinearIR& linear_ir, snippets::lowered::LinearIR::constExprIt begin, snippets::lowered::LinearIR::constExprIt end) override;

private:
    class RepackedWeightsBufferExpression : public snippets::lowered::BufferExpression {
        friend class snippets::lowered::ExpressionFactory;
    public:
        OPENVINO_RTTI("RepackedWeightsBufferExpression", "0", BufferExpression)
        RepackedWeightsBufferExpression() = default;

        void validate() const override;

        void init_allocation_size(const std::shared_ptr<snippets::lowered::LoopManager>& loop_manager, size_t allocation_rank) override;

    private:
        RepackedWeightsBufferExpression(const RepackedWeightsBufferExpression& other);
        RepackedWeightsBufferExpression(const std::shared_ptr<ov::Node>& n, const std::shared_ptr<snippets::IShapeInferSnippetsFactory>& factory);

        snippets::lowered::ExpressionPtr clone() const override;
    };

    class CompensationsBufferExpression : public snippets::lowered::BufferExpression {
        friend class snippets::lowered::ExpressionFactory;
    public:
        OPENVINO_RTTI("CompensationsBufferExpression", "0", BufferExpression)
        CompensationsBufferExpression() = default;

        void validate() const override;

        void init_allocation_size(const std::shared_ptr<snippets::lowered::LoopManager>& loop_manager, size_t allocation_rank) override;

    private:
        CompensationsBufferExpression(const CompensationsBufferExpression& other);
        CompensationsBufferExpression(const std::shared_ptr<ov::Node>& n, const std::shared_ptr<snippets::IShapeInferSnippetsFactory>& factory);

        snippets::lowered::ExpressionPtr clone() const override;
    };
};

}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov
