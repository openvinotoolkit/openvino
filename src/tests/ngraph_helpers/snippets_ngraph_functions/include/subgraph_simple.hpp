// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ngraph.hpp"
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
    explicit AddFunction(const std::vector<Shape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};
/// Add separated from inputs by Sin to WA CPU-specific disabling after inputs.
/// Works because Sinh is not supported by tokenization yet.
/// Tokenized simply by starting subgraph.
//   in1       in2
//   Sin       Sinh
//        Add
//      Result
class AddSinhFunction : public SnippetsFunctionBase {
public:
    explicit AddSinhFunction(const std::vector<Shape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};
/// Like AddSinh but with a constant second input (and no sinh on in)
//   in1       in2
//   Sin       Sinh
//        Add
//      Result
class AddSinhConstFunction : public SnippetsFunctionBase {
public:
    explicit AddSinhConstFunction(const std::vector<Shape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        NGRAPH_CHECK(input_shapes.size() == 1, "Got invalid number of input shapes");
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
    explicit EltwiseFunction(const std::vector<Shape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
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
    explicit EltwiseThreeInputsFunction(const std::vector<Shape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        NGRAPH_CHECK(input_shapes.size() == 3, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};
/// EltwiseFunctionThreeInputs with Sinh after inputs to to WA CPU-specific disabling after inputs
/// See AddSinh for details.
class EltwiseThreeInputsSinhFunction : public SnippetsFunctionBase {
public:
    explicit EltwiseThreeInputsSinhFunction(const std::vector<Shape>& inputShapes) :
        SnippetsFunctionBase(inputShapes) {
        NGRAPH_CHECK(input_shapes.size() == 3, "Got invalid number of input shapes");
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
    explicit MatMulEltwiseBranchesFunction(const std::vector<Shape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
            NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
            NGRAPH_CHECK(input_shapes[0].size() == 4 && input_shapes[1].size() == 4,
                         "Only 4D input shapes are currently supported by this test");
            // todo:
            //  Note that single-element constant are not supported by the test, since they'll be converted
            //  to snippets::op::Scalar. So a more comlex logics is required to produce reference function.
            NGRAPH_CHECK(input_shapes[0][1] == input_shapes[1][1], "Channel dimensions must be equal and != 1");
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
    explicit EltwiseLogLoopFunction(const std::vector<Shape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
            NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
