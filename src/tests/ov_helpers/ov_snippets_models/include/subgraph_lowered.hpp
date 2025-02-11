// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets_helpers.hpp"
#include "subgraph_simple.hpp"
#include "subgraph_converts.hpp"
#include "subgraph_matmul.hpp"
#include "subgraph_softmax.hpp"

/* This file provides lowered representations (after the generate() was called) for some simple functions.
 * This is required to test snippets lowering and optimization passes. All the functions are expected to be direct
 * descendants of SnippetsFunctionCustomizable (defined here) and one of the SnippetsFunctionBase derived classes
 * (declared in subgraph_simple.hpp). Note that the corresponding SnippetsFunctionBase child should use virtual inheritance
 * from SnippetsFunctionBase (typically "virtual public") to avoid creation of two internal copies of SnippetsFunctionBase.
 */

namespace ov {
namespace test {
namespace snippets {

class AddFunctionLoweredBroadcast : public AddFunction {
public:
    explicit AddFunctionLoweredBroadcast(const std::vector<PartialShape>& inputShapes, const std::vector<Shape>& broadcastShapes) :
        AddFunction(inputShapes), broadcast_shapes{broadcastShapes} {
        OPENVINO_ASSERT(input_shapes.size() == broadcast_shapes.size(),
                     "Broadcast shapes should have the same size as input_shapes");
    }

protected:
    std::shared_ptr<ov::Model> initLowered() const override;

private:
    std::vector<Shape> broadcast_shapes;
};

class EltwiseThreeInputsLoweredFunction : public EltwiseThreeInputsFunction {
public:
    explicit EltwiseThreeInputsLoweredFunction(const std::vector<PartialShape>& inputShapes, const std::vector<Shape>& broadcastShapes) :
            EltwiseThreeInputsFunction(inputShapes), broadcast_shapes{broadcastShapes} {
        OPENVINO_ASSERT(input_shapes.size() == broadcast_shapes.size(),
                     "Broadcast shapes should have the same size as input_shapes");
        OPENVINO_ASSERT(input_shapes[0].is_static() && input_shapes[1].is_static() && input_shapes[2].is_static(),
                     "Broadcast shapes should have the same size as input_shapes");
    }

protected:
    std::shared_ptr<ov::Model> initLowered() const override;
private:
    std::vector<Shape> broadcast_shapes;
};

class Transpose0213MatMulLoweredFunction : public Transpose0213MatMulFunction {
public:
    explicit Transpose0213MatMulLoweredFunction(const std::vector<PartialShape>& inputShapes, size_t position = 0) :
            Transpose0213MatMulFunction(inputShapes, std::vector<ov::element::Type>{ov::element::f32, ov::element::f32}, MatMulType::MatMul, position) {
    }
protected:
    std::shared_ptr<ov::Model> initLowered() const override;
};

class BroadcastAddLoweredFunction : public BroadcastAddFunction {
public:
    explicit BroadcastAddLoweredFunction(const std::vector<PartialShape>& inputShapes, const PartialShape& targetShape) :
            BroadcastAddFunction(inputShapes, targetShape) {}

protected:
    std::shared_ptr<ov::Model> initLowered() const override;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov

