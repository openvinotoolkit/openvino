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

class MLPSeqFunction : public SnippetsFunctionBase {
public:
    explicit MLPSeqFunction(const std::vector<PartialShape>& inputShapes,
                            const std::vector<ov::element::Type>& precisions,
                            size_t num_input_nodes,
                            size_t num_hidden_layers)
        : SnippetsFunctionBase(inputShapes),
          precisions(precisions),
          num_input_nodes(num_input_nodes),
          num_hidden_layers(num_hidden_layers) {
        OPENVINO_ASSERT(!precisions.empty(), "Precisions vector is empty");
    }

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;

    const std::vector<ov::element::Type> precisions;
    const size_t num_input_nodes, num_hidden_layers;
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

class MLPSeqQuantizedFunction : public SnippetsFunctionBase {
public:
    explicit MLPSeqQuantizedFunction(const std::vector<PartialShape>& inputShapes,
                                                const std::vector<ov::element::Type>& precisions,
                                                size_t num_input_nodes,
                                                size_t num_hidden_layers)
        : SnippetsFunctionBase(inputShapes),
          precisions(precisions),
          num_input_nodes(num_input_nodes),
          num_hidden_layers(num_hidden_layers) {
        OPENVINO_ASSERT(!precisions.empty(), "Precisions vector is empty");
    }

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;

    const std::vector<ov::element::Type> precisions;
    const size_t num_input_nodes, num_hidden_layers;
};

class MLPSeqQuantizedTypeRelaxedFunction : public SnippetsFunctionBase {
public:
    explicit MLPSeqQuantizedTypeRelaxedFunction(const std::vector<PartialShape>& inputShapes,
                                                const std::vector<ov::element::Type>& precisions,
                                                size_t num_input_nodes,
                                                size_t num_hidden_layers)
        : SnippetsFunctionBase(inputShapes),
          precisions(precisions),
          num_input_nodes(num_input_nodes),
          num_hidden_layers(num_hidden_layers) {
        OPENVINO_ASSERT(!precisions.empty(), "Precisions vector is empty");
    }

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;

    const std::vector<ov::element::Type> precisions;
    const size_t num_input_nodes, num_hidden_layers;
};

}  // namespace ov::test::snippets
