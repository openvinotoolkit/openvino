// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "./snippets_helpers.hpp"

/* This file contains definitions of relatively simple functions (models) that will be used
 * to test snippets-specific behavior. All the functions are expected to be direct descendants of
 * SnippetsFunctionBase, so their constructors take only one (inputShapes) argument.
 */

namespace ov {
namespace test {
namespace snippets {
/// The most trivial graph, just one Add.
/// Tokenized simply by starting subgraph.
// in1   in2
//    Add
//   Result
class AddFunction : public SnippetsFunctionBase {
public:
    explicit AddFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        OPENVINO_ASSERT(input_shapes.size() == 2, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};

/// One unary operation: Exp
/// Tokenized simply by starting subgraph.
//    in1
//    Exp
//   Result
class ExpFunction : public SnippetsFunctionBase {
public:
    explicit ExpFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        OPENVINO_ASSERT(input_shapes.size() == 1, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

/// Two unary operations: Exp + Power(x, -1) (aka Reciprocal)
/// Tokenized simply by starting subgraph.
//    in1
//    Exp
//  Power(x, -1)
//   Result
class ExpReciprocalFunction : public SnippetsFunctionBase {
public:
    explicit ExpReciprocalFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        OPENVINO_ASSERT(input_shapes.size() == 1, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

/// Like AddSinh but with a constant second input (and no sinh on in)
//   in1       in2
//        Add
//      Result
// todo: remove Sinh once "no subgraph after input" limitation is relaxed
class AddConstFunction : public SnippetsFunctionBase {
public:
    explicit AddConstFunction(const std::vector<PartialShape>& inputShapes, const PartialShape& constShape) :
        SnippetsFunctionBase(inputShapes), m_const_shape(constShape) {
        OPENVINO_ASSERT(input_shapes.size() == 1, "Got invalid number of input shapes");
        OPENVINO_ASSERT(m_const_shape.is_static(), "Const shape must be static shape");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    PartialShape m_const_shape;
//    std::shared_ptr<ov::Model> initReference() const override;
};
// Function is to check for different model precision
/// Like AddSinhConst but with a Roll instead of Sinh because Roll is movement operation which
//  supports different precisions but Sinh supports only FP32 in CPU Plugin
//   in1
//   Roll     Const
//        Add
//      Result
// The function is needed to check different input element types (model precision change)
class AddRollConstFunction : public AddConstFunction {
public:
    explicit AddRollConstFunction(const std::vector<PartialShape>& inputShapes, const PartialShape& constShape) :
        AddConstFunction(inputShapes, constShape) {
        OPENVINO_ASSERT(input_shapes.size() == 1, "Got invalid number of input shapes");
        OPENVINO_ASSERT(m_const_shape[0].is_static(), "Const shape must be static shape");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
//    std::shared_ptr<ov::Model> initReference() const override;
};
/// Simple Eltwise graph fully convertible to Subgraph.
/// Tokenized simply by attaching eltwises.
// in1   in2
//    Add
//   /   Subtract
//  Multiply
//   Result
class EltwiseFunction : public SnippetsFunctionBase {
public:
    explicit EltwiseFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        OPENVINO_ASSERT(input_shapes.size() == 2, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};
/// Simple Eltwise graph fully convertible to Subgraph.
/// Tokenized simply by attaching eltwises.
// in1   in2   in3   Scalar
//    Add      Multiply
//      Subtract
//       Result
class EltwiseThreeInputsFunction : public SnippetsFunctionBase {
public:
    explicit EltwiseThreeInputsFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        OPENVINO_ASSERT(input_shapes.size() == 3, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};
/// Eltwise graph with 9 inputs and 2 outputs.
/// Needed to test for a max number of inputs+outputs allowed.
// in1   in2   in3 ... in9
// ........................
//    Subtract        Power
//          \     Sinh
//          Result
class EltwiseMaxNumParamsFunction : public SnippetsFunctionBase {
public:
    explicit EltwiseMaxNumParamsFunction(const std::vector<PartialShape>& inputShapes) :
            SnippetsFunctionBase(inputShapes) {
        OPENVINO_ASSERT(input_shapes.size() == 9, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};
/// MatMul with two eltwise branches joined with Add just before the Result.
/// Tokenized by attaching eltwises to separate subgraphs, and then joining them together.
//                   in1   in2
//                     MatMul
//  [Eltwise sequence 1]   [Eltwise sequence 2]
//                      Add
//                     Result
class MatMulEltwiseBranchesFunction : public SnippetsFunctionBase {
public:
    explicit MatMulEltwiseBranchesFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
            OPENVINO_ASSERT(input_shapes.size() == 2, "Got invalid number of input shapes");
            OPENVINO_ASSERT(input_shapes[0].size() == 4 && input_shapes[1].size() == 4,
                         "Only 4D input shapes are currently supported by this test");
            // todo:
            //  Note that single-element constant are not supported by the test, since they'll be converted
            //  to snippets::op::Scalar. So a more comlex logics is required to produce reference function.
            OPENVINO_ASSERT(input_shapes[0][1] == input_shapes[1][1], "Channel dimensions must be equal and != 1");
            OPENVINO_ASSERT(input_shapes[0].is_static() && input_shapes[1].is_static(), "This test supports only static shapes");
    }

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};
/// Add with HSwish and Log  joined Multiply.
/// Log is not tokenizable, so two Subgraphs are created to avoid loop introduction: Add+HSwish and Multiply.
//     in1   in2
//        Add
//  HSwish   Log
//      Multiply
//       Result
class EltwiseLogLoopFunction : public SnippetsFunctionBase {
public:
    explicit EltwiseLogLoopFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
            OPENVINO_ASSERT(input_shapes.size() == 2, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};
/// 2 results.
/// So we have 2 subgraphs - Snippets don't support subgraphs with many results
/// Also Output tensors have names to check correct copying output names
//    in1    in2
//        Add
//  Relu   Result
//  HSwish
//  Result
class EltwiseTwoResultsFunction : public SnippetsFunctionBase {
public:
    explicit EltwiseTwoResultsFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
            OPENVINO_ASSERT(input_shapes.size() == 2, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};
/// Two different Input and Outputs.
/// This function is to check correct Broadcasting
//        in1       in2
//       HSwish      /
//  Result      Add
//              Relu
//              Sin
//             Result
class TwoInputsAndOutputsFunction : public SnippetsFunctionBase {
public:
    explicit TwoInputsAndOutputsFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        OPENVINO_ASSERT(input_shapes.size() == 2, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};
/// Two different Input and Outputs.
/// This function is to check correct Broadcasting
//        in1       in2
//       HSwish      /
//  Result      Add
//              Relu
//              Sin
//             Result
class TwoInputsAndOutputsWithReversedOutputsFunction : public SnippetsFunctionBase {
public:
    explicit TwoInputsAndOutputsWithReversedOutputsFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        OPENVINO_ASSERT(input_shapes.size() == 2, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};
/// Verify Select
//   in0     in1    in2
//     \      |     /
//         Select
//         Result
class SelectFunction : public SnippetsFunctionBase {
public:
    explicit SelectFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        OPENVINO_ASSERT(input_shapes.size() == 3, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};
/// Verify Broadcast in passes
//   in0     in1
// Broadcast  |
//     \     /
//        Add
//       Result
class BroadcastAddFunction : public SnippetsFunctionBase {
public:
    explicit BroadcastAddFunction(const std::vector<PartialShape>& inputShapes, const PartialShape& targetShape)
        : SnippetsFunctionBase(inputShapes), m_target_shape(targetShape) {
        OPENVINO_ASSERT(input_shapes.size() == 2, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;

    PartialShape m_target_shape;
};

/// Verify Select + Broadcast
//   in0     in1    in2
// Broadcast  |      |
//     \      |     /
//         Select
//         Result
class BroadcastSelectFunction : public SelectFunction {
public:
    explicit BroadcastSelectFunction(const std::vector<PartialShape>& inputShapes, const PartialShape& targetShape)
            : SelectFunction(inputShapes), m_target_shape(targetShape) {}
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;

    PartialShape m_target_shape;
};

/// graph use case to cover duplicated subgraphs edge elimination in such as in yolo-v3
//              subgraph/node
//                /     |
//           subgraph  subgraph
//                \     /
//              eltwise node
class EdgeReplaceFunction : public SnippetsFunctionBase {
public:
    explicit EdgeReplaceFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        OPENVINO_ASSERT(input_shapes.size() == 1, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};
}  // namespace snippets
}  // namespace test
}  // namespace ov
