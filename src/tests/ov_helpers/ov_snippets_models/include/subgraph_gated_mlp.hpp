// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets_helpers.hpp"

#include "common_test_utils/test_enums.hpp"

/* The file contains graphs with different GatedMLP:
 * Skeleton of GatedMLP pattern is:
 *        Pattern:
 *            Input
 *           /      \
 * FullyConnected  FullyConnected
 *          |         |
 *      Activation   /
 *          \       /
 *           Multiply
 *              |
 *         FullyConnected
 */

namespace ov::test::snippets {

class GatedMLPFunction : public SnippetsFunctionBase {
public:
    // TODO: Add more (INT8_ASYM, INT8_SYM, INT4 etc)
    enum class WeightFormat {
        FP32,
        FP16,
    };

    explicit GatedMLPFunction(const std::vector<PartialShape>& input_shapes,
                              const std::vector<Shape>& weights_shapes,
                              WeightFormat wei_format,
                              utils::ActivationTypes act_type)
        : SnippetsFunctionBase(input_shapes),
          m_weights_shapes(weights_shapes),
          m_wei_format(wei_format),
          m_act_type(act_type) {
        OPENVINO_ASSERT(input_shapes.size() == 1, "MLPFunction expects 1 input shape");
        OPENVINO_ASSERT(weights_shapes.size() == 3, "MLPFunction expects 3 weights shapes");
    }

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;

    std::shared_ptr<ov::Node> makeWeights(const Shape& shape, int32_t seed = 1) const;
    std::shared_ptr<ov::Node> makeFC(const ov::Output<ov::Node>& output, const Shape& shape, int32_t seed = 1) const;
    std::shared_ptr<ov::Node> makeFC(const ov::Output<ov::Node>& output, const ov::Output<ov::Node>& weights) const;

    const std::vector<Shape> m_weights_shapes {};
    const WeightFormat m_wei_format = {};
    const utils::ActivationTypes m_act_type = {};
};

std::ostream& operator<<(std::ostream& os, GatedMLPFunction::WeightFormat type);

}  // namespace ov::test::snippets
