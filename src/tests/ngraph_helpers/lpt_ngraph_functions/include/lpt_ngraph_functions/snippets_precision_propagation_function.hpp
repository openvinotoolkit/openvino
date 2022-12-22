//// Copyright (C) 2023 Intel Corporation
//// SPDX-License-Identifier: Apache-2.0
////
//
//#pragma once
//
//#include <memory>
//#include <ngraph/ngraph.hpp>
//#include <low_precision/layer_transformation.hpp>
//
//#include "elementwise_function.hpp"
//#include "lpt_ngraph_functions/common/builders.hpp"
//#include "lpt_ngraph_functions/common/convolution.hpp"
//#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
//#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
//
//namespace ngraph {
//namespace builder {
//namespace subgraph {
//
//class SnippetsPrecisionPropagationFunction {
//public:
//    static std::shared_ptr<ngraph::Function> get(
//        const ngraph::element::Type precision1,
//        const ngraph::PartialShape& inputShape1,
//        const ngraph::element::Type precision2,
//        const ngraph::PartialShape& inputShape2);
//};
//
//}  // namespace subgraph
//}  // namespace builder
//}  // namespace ngraph
