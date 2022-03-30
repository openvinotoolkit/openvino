// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ngraph.hpp"
#include "snippets_helpers.hpp"
#include "subgraph_simple.hpp"

/* This file provides lowered representations (after the generate() was calles) for some simple functions.
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
    explicit AddFunctionLoweredBroadcast(const std::vector<Shape>& inputShapes, const std::vector<Shape>& broadcastShapes) :
        AddFunction(inputShapes), broadcast_shapes{broadcastShapes} {
        NGRAPH_CHECK(input_shapes.size() == broadcast_shapes.size(),
                     "Broadcast shapes should have the same size as input_shapes");
    }

protected:
    std::shared_ptr<ov::Model> initLowered() const override;

private:
    std::vector<Shape> broadcast_shapes;
};

class EltwiseThreeInputsLoweredFunction : public EltwiseThreeInputsFunction {
public:
    explicit EltwiseThreeInputsLoweredFunction(const std::vector<Shape>& inputShapes, const std::vector<Shape>& broadcastShapes) :
            EltwiseThreeInputsFunction(inputShapes), broadcast_shapes{broadcastShapes} {
        NGRAPH_CHECK(input_shapes.size() == broadcast_shapes.size(),
                     "Broadcast shapes should have the same size as input_shapes");
    }

protected:
    std::shared_ptr<ov::Model> initLowered() const override;
private:
    std::vector<Shape> broadcast_shapes;;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov

