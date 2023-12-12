// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "op/bitwise_and.hpp"
#include "default_opset.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
using namespace ov::op;

namespace ngraph {
	namespace onnx_import {
		namespace op {
			namespace set_1 {
				OutputVector bitwise_and(const Node& node) {
					const Output<ngraph::Node> a = node.get_ng_inputs().at(0);
					const Output<ngraph::Node> b = node.get_ng_inputs().at(1);

					NGRAPH_CHECK(a.get_element_type() != ov::element::bf16 && b.get_element_type() != ov::element::bf16,
						"The input data bfloat16 isn't supported in opset 12");

					return { std::make_shared<v13::BitwiseAnd>(a, b) };

				}

			}  // namespace set_1

		}  // namespace op

	}  // namespace onnx_import

}  // namespace ngraph
