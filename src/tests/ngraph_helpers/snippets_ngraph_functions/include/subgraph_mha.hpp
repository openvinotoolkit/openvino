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
    explicit MHAFunction(const std::vector<PartialShape>& inputShapes, const std::vector<ov::element::Type>& precisions, bool with_mul = true)
        : SnippetsFunctionBase(inputShapes), with_mul(with_mul), precisions(precisions) {
        OPENVINO_ASSERT(input_shapes.size() == 4, "Got invalid number of input shapes");
        OPENVINO_ASSERT(precisions.size() == 4, "Got invalid number of input precisions");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;

    bool with_mul = true;
    std::vector<ov::element::Type> precisions;
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
    explicit MHAMatMul0TransposeFunction(const std::vector<PartialShape>& inputShapes, const std::vector<ov::element::Type>& precisions)
            : SnippetsFunctionBase(inputShapes), precisions(precisions) {
        OPENVINO_ASSERT(input_shapes.size() == 4, "Got invalid number of input shapes");
        OPENVINO_ASSERT(precisions.size() == 4, "Got invalid number of input precisions");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;

    std::vector<ov::element::Type> precisions;
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
    explicit MHASelectFunction(const std::vector<PartialShape>& inputShapes, const std::vector<ov::element::Type>& precisions)
        : SnippetsFunctionBase(inputShapes), precisions(precisions) {
        OPENVINO_ASSERT(input_shapes.size() == 6, "Got invalid number of input shapes");
        OPENVINO_ASSERT(precisions.size() == 6, "Got invalid number of input precisions");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;

    std::vector<ov::element::Type> precisions;
};

// Only for tokenization tests since boolean type->u8
// Without Transposes
class MHASelectSplitMFunction : public SnippetsFunctionBase {
public:
    explicit MHASelectSplitMFunction(const std::vector<PartialShape>& inputShapes, const std::vector<Shape>& reshapes)
            : SnippetsFunctionBase(inputShapes), reshapes(reshapes) {
        OPENVINO_ASSERT(input_shapes.size() == 5, "Got invalid number of input shapes");
        OPENVINO_ASSERT(reshapes.size() == 6, "Got invalid number of input precisions");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;

    std::vector<Shape> reshapes;
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
        OPENVINO_ASSERT(input_shapes.size() == 3, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

/* Graph:
 *    \     /
 *    MatMul0
 *       |
 *    Softmax
 *        \      /
 *         MatMul1
 *           |
 */
class MHAWOTransposeFunction : public SnippetsFunctionBase {
public:
    explicit MHAWOTransposeFunction(const std::vector<PartialShape>& inputShapes, const std::vector<ov::element::Type>& precisions)
        : SnippetsFunctionBase(inputShapes), precisions(precisions) {
        OPENVINO_ASSERT(input_shapes.size() == 3, "Got invalid number of input shapes");
        OPENVINO_ASSERT(precisions.size() == 3, "Got invalid number of input precisions");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;

     std::vector<ov::element::Type> precisions;
};

class MHAWOTransposeSplitMFunction : public MHAWOTransposeFunction {
public:
    explicit MHAWOTransposeSplitMFunction(const std::vector<PartialShape>& inputShapes, const std::vector<ov::element::Type>& precisions,
                                          const std::vector<Shape>& reshapes)
            : MHAWOTransposeFunction(inputShapes, precisions), reshapes(reshapes) {
        OPENVINO_ASSERT(reshapes.size() == 4, "Got invalid number of Reshape shapes");
    }
protected:
    std::shared_ptr<ov::Model> initReference() const override;

    std::vector<ov::Shape> reshapes;
};

/* Graph:
 * Transpose0[0,2,1,3] Transpose1[0,2,3,1]
 *              \     /
 *              MatMul0
 *            FakeQuantize i8
 *                 \   /
 *                  Add
 *                Reshape0
 *                Softmax
 *                Reshape1  Transpose2[0,2,1,3]
 *                    \      /
 *                     MatMul1
 *                   FakeQuantize i8
 *                  Transpose3[0,2,1,3]
 */
class MHAFQAfterMatMulFunction : public SnippetsFunctionBase {
public:
    explicit MHAFQAfterMatMulFunction(const std::vector<PartialShape>& inputShapes)
            : SnippetsFunctionBase(inputShapes) {
        OPENVINO_ASSERT(input_shapes.size() == 4, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

/* Graph:
 *   FakeQuantize i8      FakeQuantize i8
 * Transpose0[0,2,1,3] Transpose1[0,2,3,1]
 *              \     /
 *              MatMul0
 *            FakeQuantize i8
 *                 \   /
 *                  Add
 *                Reshape0
 *                Softmax
 *                Reshape1   FakeQuantize i8
 *            FakeQuantize u8 Transpose2[0,2,1,3]
 *                    \      /
 *                     MatMul1
 *                  FakeQuantize i8
 *                  Transpose3[0,2,1,3]
 */
class MHAINT8MatMulFunction : public SnippetsFunctionBase {
public:
    explicit MHAINT8MatMulFunction(const std::vector<PartialShape>& inputShapes)
            : SnippetsFunctionBase(inputShapes) {
        OPENVINO_ASSERT(input_shapes.size() == 4, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

/* Graph:
 *   FakeQuantize i8      Reshape1
 *       Reshape0       Transpose1[0,2,3,1]
 * Transpose0[0,2,1,3] FakeQuantize i8
 *              \     /
 *              MatMul0
 *                 \   /
 *                  Add        Reshape2
 *                Softmax   Transpose2[0,2,1,3]
 *                    \      /
 *                     MatMul1
 *                  FakeQuantize i8
 *                  Transpose3[0,2,1,3]
 *                    Reshape3
 * Note: Reshapes are tosplit Tokenization between FQs and deq Mul and MHA since Snippets::Ignore_Callback may be enabled
 */
class MHAQuantMatMul0Function : public SnippetsFunctionBase {
public:
    explicit MHAQuantMatMul0Function(const std::vector<PartialShape>& inputShapes)
            : SnippetsFunctionBase(inputShapes) {
        NGRAPH_CHECK(input_shapes.size() == 4, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};


/* Graph:
 *                                          Constant
 *   FakeQuantize u8      FakeQuantize u8   Convert
 * Transpose0[0,2,1,3] Transpose1[0,2,3,1]  Multiply
 *                  \             \         /
 *                   \              Multiply
 *                    \         FakeQuantize f32
 *                     \         /
 *                       MatMul0
 *                     FakeQuantize f32   FakeQuantize u8
 *                                   \     /
 *                                     Add
 *                                   Softmax  Transpose2[0,2,1,3]
 *                                       \      /
 *                                        MatMul1
 *                                     FakeQuantize u8
 *                                     Transpose3[0,2,1,3]
 * Note: Check a lot of different FQ (the both quantized and floating) - buffers with different size and precision
 */
class MHAFQFunction : public SnippetsFunctionBase {
public:
    explicit MHAFQFunction(const std::vector<PartialShape>& inputShapes)
            : SnippetsFunctionBase(inputShapes) {
        OPENVINO_ASSERT(input_shapes.size() == 4, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

// Only for tokenization! The graph is after LPT: contains TypeRelaxed ops
/* Graph:
 *   FakeQuantize i8      FakeQuantize i8
 * Transpose0[0,2,1,3] Transpose1[0,2,3,1]
 *              \     /
 *              MatMul0
 *            FakeQuantize i8
 *                 \   /
 *                  Add
 *                  Mul (DeQuantize)
 *                Reshape0
 *                Softmax
 *                Reshape1   FakeQuantize i8
 *            FakeQuantize u8 Transpose2[0,2,1,3]
 *                    \      /
 *                     MatMul1
 *                  FakeQuantize i8
 *                  Transpose3[0,2,1,3]
 */
class MHAINT8MatMulTypeRelaxedFunction : public SnippetsFunctionBase {
public:
    explicit MHAINT8MatMulTypeRelaxedFunction(const std::vector<PartialShape>& inputShapes)
            : SnippetsFunctionBase(inputShapes) {
        OPENVINO_ASSERT(input_shapes.size() == 4, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};

/* Graph:
 * Transpose0[0,2,1,3] Transpose1[0,2,3,1]
 *              \     /
 *              MatMul0
 *                 \
 *                Multiply
 *                  Add
 *                Softmax   Transpose2[0,2,1,3]
 *                    \      /
 *                     MatMul1
 *                   Transpose3[0,2,1,3]
 */
class MHAMulAddFunction : public SnippetsFunctionBase {
public:
    explicit MHAMulAddFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        OPENVINO_ASSERT(input_shapes.size() == 3, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

/* Graph:
 *       Transpose/Parameter
 *    \     /
 *    MatMul0 [transposed_b = true/false]
 *       |
 *    Softmax
 *        \      /
 *         MatMul1
 *           |
 */
class MHATransposedInputFunction : public SnippetsFunctionBase {
public:
    explicit MHATransposedInputFunction(const std::vector<PartialShape>& inputShapes, bool transposed_b = false,
                                        std::vector<int64_t> order = {})
        : SnippetsFunctionBase(inputShapes), m_transposed_b(transposed_b), m_order(order) {
        OPENVINO_ASSERT(input_shapes.size() == 3, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;

    bool m_transposed_b = false;
    std::vector<int64_t> m_order = {};
};

/* Graph:
 *           input0   input1
 *              \     /
 *              MatMul0   input2
 *                 |        | 
 *              Reshape  Reshape (optional)
 *                 |    /
 *              Eltwise1  input3
 *                 |     /
 *              Eltwise2
 *                 |
 *              Reshape
 *                 |
 *              Softmax
 *                 |       input4
 *                  \      /
 *                   MatMul1
 */
class MHAWithExtractedReshapeFunction : public SnippetsFunctionBase {
public:
    explicit MHAWithExtractedReshapeFunction(const std::vector<PartialShape>& inputShapes, const bool add_2nd_reshape)
        : SnippetsFunctionBase(inputShapes), add_2nd_reshape(add_2nd_reshape) {
        NGRAPH_CHECK(input_shapes.size() == 5, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
private:
    bool add_2nd_reshape = false;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
