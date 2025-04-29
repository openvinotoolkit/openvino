// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets_helpers.hpp"

/* The file contains graphs with different MLP-patterns:
 * Skeleton of MLP-pattern is:
 *       Input
 *         |
 *      MatMul
 *         |
 *      Eltwise
 *         |
 *      MatMul
 *         |
 *      Output
 */

namespace ov::test::snippets {

/* Graph:
 *       Input
 *         |
 *        FQ
 *         |
 *      MatMul
 *         |
 *      Eltwise
 *         |
 *        ...
 *         |
 *      Eltwise
 *         |
 *        FQ
 *         |
 *      MatMul
 *         |
 *      Output
 */
class MLPSeqTypeRelaxedFunction : public SnippetsFunctionBase {
public:
    explicit MLPSeqTypeRelaxedFunction(const std::vector<PartialShape>& inputShapes, const std::vector<ov::element::Type>& precisions, size_t num_layers)
        : SnippetsFunctionBase(inputShapes), precisions(precisions), num_layers(num_layers) {
        OPENVINO_ASSERT(!precisions.empty(), "Precisions vector is empty");
    }

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;

    const std::vector<ov::element::Type> precisions;
    const size_t num_layers;
};

class MLPSeqFunction : public SnippetsFunctionBase {
public:
    explicit MLPSeqFunction(const std::vector<PartialShape>& inputShapes, const std::vector<ov::element::Type>& precisions, size_t num_layers)
        : SnippetsFunctionBase(inputShapes), precisions(precisions), num_layers(num_layers) {
        OPENVINO_ASSERT(!precisions.empty(), "Precisions vector is empty");
    }

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;

    const std::vector<ov::element::Type> precisions;
    const size_t num_layers;
};

}  // namespace ov::test::snippets
