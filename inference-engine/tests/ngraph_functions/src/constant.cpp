// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"


namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeConstant(const element::Type &type, const std::vector<size_t> &shape,
                                   const std::vector<float> &data, bool random) {
    std::shared_ptr<ngraph::Node> weightsNode;

#define makeNode(TYPE) \
        case TYPE: \
            weightsNode = std::make_shared<ngraph::opset1::Constant>( \
                    type, shape, \
                    random ? NGraphFunctions::Utils::generateVector<TYPE>(ngraph::shape_size(shape)) : \
                             NGraphFunctions::Utils::castVector<float, ngraph::helpers::nGraphTypesTrait<TYPE>::value_type >(data)); \
            break;
    switch (type) {
        case ngraph::element::Type_t::bf16:
            weightsNode = std::make_shared<ngraph::opset1::Constant>(
                    type, shape,
                    random ? NGraphFunctions::Utils::generateBF16Vector(ngraph::shape_size(shape)) :
                    NGraphFunctions::Utils::castVector<float, ngraph::bfloat16>(data));
            break;
        case ngraph::element::Type_t::f16:
            weightsNode = std::make_shared<ngraph::opset1::Constant>(
                    type, shape,
                    random ? NGraphFunctions::Utils::generateF16Vector(ngraph::shape_size(shape)) :
                    NGraphFunctions::Utils::castVector<float, ngraph::float16>(data));
            break;
        makeNode(ngraph::element::Type_t::f32);
        makeNode(ngraph::element::Type_t::f64);
        makeNode(ngraph::element::Type_t::i8);
        makeNode(ngraph::element::Type_t::i16);
        makeNode(ngraph::element::Type_t::i32);
        makeNode(ngraph::element::Type_t::i64);
        makeNode(ngraph::element::Type_t::u8);
        makeNode(ngraph::element::Type_t::u16);
        makeNode(ngraph::element::Type_t::u32);
        makeNode(ngraph::element::Type_t::u64);
#undef makeNode
        default:
            throw std::runtime_error("Unhandled precision");
    }
    return weightsNode;
}
}  // namespace builder
}  // namespace ngraph