// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets_helpers.hpp"

/* The file contains graphs with different MLP sequential patterns:
 * Skeleton of MLP sequential pattern is:
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

class MLPSeqFunction : public SnippetsFunctionBase {
public:
    explicit MLPSeqFunction(const std::vector<PartialShape>& inputShapes,
                            const std::vector<ov::element::Type>& precisions,
                            size_t num_hidden_layers,
                            size_t hidden_matmul_size)
        : SnippetsFunctionBase(inputShapes),
          precisions(precisions),
          num_hidden_layers(num_hidden_layers),
          hidden_matmul_size(hidden_matmul_size) {
        OPENVINO_ASSERT(precisions.size() == 1, "MLPSeqFunction expects only precision");
    }

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;

    const std::vector<ov::element::Type> precisions;
    const size_t num_hidden_layers, hidden_matmul_size;
};

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

class MLPSeqQuantizedFunction : public MLPSeqFunction {
public:
    explicit MLPSeqQuantizedFunction(const std::vector<PartialShape>& inputShapes,
                                     const std::vector<ov::element::Type>& precisions,
                                     size_t num_hidden_layers,
                                     size_t hidden_matmul_size)
        : MLPSeqFunction(inputShapes, precisions, num_hidden_layers, hidden_matmul_size) {}

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

class MLPSeqQuantizedTypeRelaxedFunction : public MLPSeqFunction {
public:
    explicit MLPSeqQuantizedTypeRelaxedFunction(const std::vector<PartialShape>& inputShapes,
                                                const std::vector<ov::element::Type>& precisions,
                                                size_t num_hidden_layers,
                                                size_t hidden_matmul_size)
        : MLPSeqFunction(inputShapes, precisions, num_hidden_layers, hidden_matmul_size) {}

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};

}  // namespace ov::test::snippets
