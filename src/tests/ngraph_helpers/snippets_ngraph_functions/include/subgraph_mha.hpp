// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ngraph.hpp"
#include "snippets_helpers.hpp"


/* The file contains graphs with different MHA-patterns:
 * Skeleton on MHA-pattern is:
 *              \     /
 *              MatMul0
 *                 |
 * Eltwise/Select/Reshape/FakeQuantize
 *                 |
 *              Softmax
 *                 |
 * Eltwise/Select/Reshape/FakeQuantize
 *                  \      /
 *                   MatMul1
 */

namespace ov {
namespace test {
namespace snippets {

/* Graph:
 *       Transpose1[0,2,3,1]  Constant
 *                     \       /
 * Transpose0[0,2,1,3] Multiply [with_mul = true]
 *              \     /
 *              MatMul0
 *                 \   /
 *                  Add
 *                Reshape0
 *                Softmax
 *                Reshape1  Transpose2[0,2,1,3]
 *                    \      /
 *                     MatMul1
 *                   Transpose3[0,2,1,3]
 */
class MHAFunction : public SnippetsFunctionBase {
public:
    explicit MHAFunction(const std::vector<PartialShape>& inputShapes, bool with_mul = true)
        : SnippetsFunctionBase(inputShapes), with_mul(with_mul) {
        NGRAPH_CHECK(input_shapes.size() == 4, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;

    bool with_mul = true;
};

/* Graph:
 *       Transpose1[0,2,1,3]  Constant
 *                     \       /
 * Transpose0[0,2,1,3] Multiply
 *              \     /
 *              MatMul0 [transposed_b = true]
 *                 \   /
 *                  Add
 *                Reshape0
 *                Softmax
 *                Reshape1  Transpose2[0,2,1,3]
 *                    \      /
 *                     MatMul1
 *                   Transpose3[0,2,1,3]
 */
class MHAMatMul0TransposeFunction : public SnippetsFunctionBase {
public:
    explicit MHAMatMul0TransposeFunction(const std::vector<PartialShape>& inputShapes)
            : SnippetsFunctionBase(inputShapes) {
        NGRAPH_CHECK(input_shapes.size() == 4, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};

/* Graph:
 *             Transpose1[0,2,3,1]  Constant
 *                           \       /
 *       Transpose0[0,2,1,3] Multiply
 *     \               \     /
 * Broadcast  Scalar   MatMul0
 *       \      |      /
 *           Select
 *          Reshape0
 *          Softmax
 *          Reshape1  Transpose2[0,2,1,3]
 *              \      /
 *               MatMul1
 *             Transpose3[0,2,1,3]
 */
class MHASelectFunction : public SnippetsFunctionBase {
public:
    explicit MHASelectFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        NGRAPH_CHECK(input_shapes.size() == 6, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

/* Graph:
 *             Constant
 *        \      /
 *        Multiply
 *    \     /
 *    MatMul0
 *       |
 *    Softmax
 *        \      /
 *         MatMul1
 *           |
 *       Transpose3[0,2,1,3]
 */
class MHAWOTransposeOnInputsFunction : public SnippetsFunctionBase {
public:
    explicit MHAWOTransposeOnInputsFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        NGRAPH_CHECK(input_shapes.size() == 3, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
