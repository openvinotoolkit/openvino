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
/// Minimal graph to test MatMul support
/// Works because Sinh is not supported by tokenization yet.
/// Tokenized simply by starting subgraph,
//   in1        in2
//   Sinh       Sinh
//        Matmul
//         Result
// todo: remove Sinh once "no subgraph after input" limitation is relaxed
class MatMulSinhFunction : public SnippetsFunctionBase {
public:
    explicit MatMulSinhFunction(const std::vector<PartialShape>& inputShapes)
    : SnippetsFunctionBase(inputShapes) {
        NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};

// As same as MatMulSinhFunction but with biases
class MatMulBiasSinhFunction : public SnippetsFunctionBase {
public:
    explicit MatMulBiasSinhFunction(const std::vector<PartialShape>& inputShapes)
            : SnippetsFunctionBase(inputShapes) {
        NGRAPH_CHECK(input_shapes.size() == 3, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

/// Minimal graph to test MatMul+Transpose combinations. Transpose location is specified via the position argument:
/// 0 - before the first MatMul input; 1 - before the second MatMul input; 2 - after the MatMul output.
/// Tokenized simply by starting subgraph,
//   in1        in2
//   Sinh       Sinh
//   Transpose  /
//         Matmul
//         Result
// todo: remove Sinh once "no subgraph after input" limitation is relaxed
class Transpose0213MatMulSinhFunction : public SnippetsFunctionBase {
public:
    explicit Transpose0213MatMulSinhFunction(const std::vector<PartialShape>& inputShapes, size_t position = 0,
                                             bool insert_guard = true)
    : SnippetsFunctionBase(inputShapes), transpose_position(position), insert_guard(insert_guard)  {
        NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
        NGRAPH_CHECK(input_shapes[0].rank().get_length() == 4 && input_shapes[1].rank().get_length() == 4,
                     "Only rank 4 input shapes are supported by this test");
        NGRAPH_CHECK(transpose_position >=0 && transpose_position <= 2, "Got invalid transpose position");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    size_t transpose_position;
    bool insert_guard; // true if Sinh ops should be inserted after inputs
};

class TransposeMatMulSinhFunction : public SnippetsFunctionBase {
public:
    explicit TransposeMatMulSinhFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

class TransposeMatMulBiasSinhFunction : public SnippetsFunctionBase {
public:
    explicit TransposeMatMulBiasSinhFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        NGRAPH_CHECK(input_shapes.size() == 3, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

class TransposeMulMatMulBiasSinhFunction : public SnippetsFunctionBase {
public:
    explicit TransposeMulMatMulBiasSinhFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        NGRAPH_CHECK(input_shapes.size() == 4, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
