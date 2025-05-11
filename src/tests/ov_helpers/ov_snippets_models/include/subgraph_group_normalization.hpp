// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {

/* Graph with example shape propogation:
 * Graph between Reshape[1,2,1,8] and Reshape[1,2,2,4] is a MVN
 *       Parameter[1,4,2,2], group_num is 2
 *           |
 *       Reshape[1,2,1,8]
 *        |     |
 *        | ReduceSum[1,2,1,1] Scalar
 *        |     |              /
 *        |    Multiply[1,2,1,1]
 *        |     /
 *       Substract[1,2,1,8]
 *        |     |
 *        |  PowerStatic[1,2,1,8]
 *        |     |
 *        |  ReduceSum[1,2,1,1]
 *        |     |
 *        |  FMA(Multiply+Add)[1,2,1,1]
 *        |     |
 *        |    sqrt[1,2,1,1]
 *        |     |
 *        |  PowerStatic[1,2,1,1]
 *        |     /
 *       Multiply[1,2,1,8] Parameter[4] Parameter[4]
 *               |              |             |
 *        Reshape[1,2,2,4]  Reshape[1,2,2,1] Reshape[1,2,2,1]
 *                       \         |         /
 *                            \    |    /
 *                            FMA(Multiply+Add)[1,2,2,4]
 *                                 |
 *                              Reshape[1,4,2,2]
 *                                 |
 *                               Result[1,4,2,2]
 */
class GroupNormalizationFunction : public SnippetsFunctionBase {
public:
    explicit GroupNormalizationFunction(const std::vector<PartialShape>& inputShapes, const size_t& numGroup, const float& eps)
        : SnippetsFunctionBase(inputShapes), num_groups(numGroup), epsilon(eps)  {
        OPENVINO_ASSERT(input_shapes.size() == 3, "Got invalid number of input shapes");
    }

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
    std::shared_ptr<ov::Model> initLowered() const override;

private:
    size_t num_groups;
    float epsilon;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
