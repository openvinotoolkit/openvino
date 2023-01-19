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
/// Tokenized simply by starting subgraph,
//   in1        in2
//        Matmul
//         Result
// todo: remove  once "no subgraph after input" limitation is relaxed
class MatMulFunction : public SnippetsFunctionBase {
public:
    explicit MatMulFunction(const std::vector<PartialShape>& inputShapes)
    : SnippetsFunctionBase(inputShapes) {
        NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};

// As same as MatMulFunction but with biases
class MatMulBiasFunction : public SnippetsFunctionBase {
public:
    explicit MatMulBiasFunction(const std::vector<PartialShape>& inputShapes)
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
// Transpose  /
//         Matmul
//         Result
class Transpose0213MatMulFunction : public SnippetsFunctionBase {
public:
    explicit Transpose0213MatMulFunction(const std::vector<PartialShape>& inputShapes, size_t position = 0)
    : SnippetsFunctionBase(inputShapes), transpose_position(position)  {
        NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
        NGRAPH_CHECK(input_shapes[0].rank().get_length() == 4 && input_shapes[1].rank().get_length() == 4,
                     "Only rank 4 input shapes are supported by this test");
        NGRAPH_CHECK(transpose_position >=0 && transpose_position <= 2, "Got invalid transpose position");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    size_t transpose_position;
};

class TransposeMatMulFunction : public SnippetsFunctionBase {
public:
    explicit TransposeMatMulFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

class TransposeMatMulBiasFunction : public SnippetsFunctionBase {
public:
    explicit TransposeMatMulBiasFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        NGRAPH_CHECK(input_shapes.size() == 3, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

class TransposeMulMatMulBiasFunction : public SnippetsFunctionBase {
public:
    explicit TransposeMulMatMulBiasFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        NGRAPH_CHECK(input_shapes.size() == 4, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
