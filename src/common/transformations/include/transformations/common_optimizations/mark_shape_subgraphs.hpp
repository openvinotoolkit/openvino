// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MarkEntireShapeSubgraphs;
class TRANSFORMATIONS_API MarkConstantsInShapeSubgraphs;
class TRANSFORMATIONS_API MarkDividesInShapeSubgraphs;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief MarkEntireShapeSubgraphs marks entirely
 * all shape subgraphs starting from precision-sensitive inputs and ending at
 * the ShapeOf node as disabled for FP16 compression.
 */
class ov::pass::MarkEntireShapeSubgraphs : public ModelPass {
public:
    OPENVINO_RTTI("MarkEntireShapeSubgraphs", "0");
    MarkEntireShapeSubgraphs();
    bool run_on_model(const std::shared_ptr<ov::Model>& f) override;

protected:
    std::function<void(ov::Node*)> m_markup_func;
};

/**
 * @ingroup ie_transformation_common_api
 * @brief MarkConstantsInShapeSubgraphs marks the constants
 * inside of all shape subgraphs starting from precision-sensitive inputs and ending at
 * the ShapeOf node as disabled for FP16 compression.
 */
class ov::pass::MarkConstantsInShapeSubgraphs : public MarkEntireShapeSubgraphs {
public:
    OPENVINO_RTTI("MarkConstantsInShapeSubgraphs", "0");
    MarkConstantsInShapeSubgraphs();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief MarkDividesInShapeSubgraphs marks the Divide layers
 * inside of all shape subgraphs starting from precision-sensitive input and ending at
 * the ShapeOf node as disabled for ConvertDivide transformation.
 */
class ov::pass::MarkDividesInShapeSubgraphs : public MarkEntireShapeSubgraphs {
public:
    OPENVINO_RTTI("MarkDividesInShapeSubgraphs", "0");
    MarkDividesInShapeSubgraphs();
};
