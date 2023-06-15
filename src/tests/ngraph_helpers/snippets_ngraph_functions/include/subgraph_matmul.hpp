// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ngraph.hpp"
#include "./snippets_helpers.hpp"
#include "snippets/utils.hpp"

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
class MatMulFunction : public SnippetsFunctionBase {
public:
    explicit MatMulFunction(const std::vector<PartialShape>& inputShapes, const std::vector<ov::element::Type>& precisions)
    : SnippetsFunctionBase(inputShapes), precisions(precisions) {
        NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
        validate_precisions(precisions);
    }
    static void validate_precisions(const std::vector<ov::element::Type>& precisions) {
        NGRAPH_CHECK(precisions.size() == 2, "Got invalid number of input element types");
        const bool is_f32 = ov::snippets::utils::everyone_is(element::f32, precisions[0], precisions[1]);
        const bool is_int8 = ov::snippets::utils::one_of(precisions[0], element::i8, element::u8) && precisions[1] == element::i8;
        const bool is_bf16 = ov::snippets::utils::everyone_is(element::bf16, precisions[0], precisions[1]);
        NGRAPH_CHECK(is_f32 || is_bf16 || is_int8, "Invalid precisions");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;

    std::vector<ov::element::Type> precisions;
};

class FQMatMulFunction : public SnippetsFunctionBase {
public:
    explicit FQMatMulFunction(const std::vector<PartialShape>& inputShapes, int pos = -1) : SnippetsFunctionBase({inputShapes[0]}), pos(pos)  {
        NGRAPH_CHECK(inputShapes.size() == 2, "Got invalid number of input shapes");
        NGRAPH_CHECK(pos >=-1 && pos <= 2, "Got invalid transpose position");
        const_shape = inputShapes[1];
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;

    ov::PartialShape const_shape;
    int pos = -1;
};

// As same as MatMulFunction but with biases
class MatMulBiasFunction : public SnippetsFunctionBase {
public:
    explicit MatMulBiasFunction(const std::vector<PartialShape>& inputShapes, const std::vector<ov::element::Type>& precisions)
            : SnippetsFunctionBase(inputShapes), precisions(precisions) {
        NGRAPH_CHECK(input_shapes.size() == 3, "Got invalid number of input shapes");
        MatMulFunction::validate_precisions(precisions);
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;

    std::vector<ov::element::Type> precisions;
};

//  Quantized MatMul
//       FQ[I8]
//        Add
class MatMulBiasQuantizedFunction : public SnippetsFunctionBase {
public:
    explicit MatMulBiasQuantizedFunction(const std::vector<PartialShape>& inputShapes, const std::vector<ov::element::Type>& precisions)
            : SnippetsFunctionBase(inputShapes), precisions(precisions) {
        NGRAPH_CHECK(input_shapes.size() == 3, "Got invalid number of input shapes");
        MatMulFunction::validate_precisions(precisions);
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;

    std::vector<ov::element::Type> precisions;
};

//  Quantized MatMul  FQ[I8]
//       FQ[U8]    Reshape  <- To have only one sequence in Subgraph: MatMuL->FQ[U8]->MatMul->FQ[I8]
//            \     /
//             MatMul
//             FQ[I8]
class MatMulsQuantizedFunction : public SnippetsFunctionBase {
public:
    explicit MatMulsQuantizedFunction(const std::vector<PartialShape>& inputShapes, const std::vector<ov::element::Type>& precisions)
            : SnippetsFunctionBase(inputShapes), precisions(precisions) {
        NGRAPH_CHECK(input_shapes.size() == 3, "Got invalid number of input shapes");
        MatMulFunction::validate_precisions(precisions);
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;

    std::vector<ov::element::Type> precisions;
};

/// Minimal graph to test MatMul+Transpose combinations. Transpose location is specified via the position argument:
/// 0 - before the first MatMul input; 1 - before the second MatMul input; 2 - after the MatMul output.
/// Tokenized simply by starting subgraph,
//   in1        in2
//   Transpose  /
//         Matmul
//         Result
class Transpose0213MatMulFunction : public SnippetsFunctionBase {
public:
    explicit Transpose0213MatMulFunction(const std::vector<PartialShape>& inputShapes, const std::vector<ov::element::Type>& precisions,
                                         size_t position = 0)
    : SnippetsFunctionBase(inputShapes), transpose_position(position), precisions(precisions)  {
        NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
        NGRAPH_CHECK(input_shapes[0].rank().get_length() == 4 && input_shapes[1].rank().get_length() == 4,
                     "Only rank 4 input shapes are supported by this test");
        NGRAPH_CHECK(transpose_position >=0 && transpose_position <= 2, "Got invalid transpose position");
        MatMulFunction::validate_precisions(precisions);
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    size_t transpose_position;
    std::vector<ov::element::Type> precisions;
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

//  Quantized MatMul  FQ[I8]
//       Softmax    Reshape  <- To have only one sequence in Subgraph: MatMuL->Softmax>FQ[U8]->MatMul->FQ[I8]
//        FQ[U8]     /
//             MatMul
//             FQ[I8]
class MatMulsQuantizedSoftmaxFunction : public SnippetsFunctionBase {
public:
    explicit MatMulsQuantizedSoftmaxFunction(const std::vector<PartialShape>& inputShapes, const std::vector<ov::element::Type>& precisions)
            : SnippetsFunctionBase(inputShapes), precisions(precisions) {
        NGRAPH_CHECK(input_shapes.size() == 3, "Got invalid number of input shapes");
        MatMulFunction::validate_precisions(precisions);
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;

    std::vector<ov::element::Type> precisions;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
