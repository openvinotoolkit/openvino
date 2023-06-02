// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/opsets/opset8.hpp>

int main() {
    //! [ngraph:graph]
    // _____________    _____________
    // | Parameter |    | Parameter |
    // |   data1   |    |   data2   |
    // |___________|    |___________|
    //         |            |
    // data1_t |            | data2_t
    //          \          /
    //           \        /
    //            \      /
    //         ____\____/____
    //         |   Concat   |
    //         |   concat   |
    //         |____________|
    //               |
    //               | concat_t
    //               |
    //        _______|_______
    //        |    Result   |
    //        |    result   |
    //        |_____________|
    auto data1 = std::make_shared<ov::opset8::Parameter>(ov::element::i64, ov::Shape{1, 3, 2, 2});
    data1->set_friendly_name("data1");        // operation name
    data1->output(0).set_names({"data1_t"});  // tensor names
    auto data2 = std::make_shared<ov::opset8::Parameter>(ov::element::i64, ov::Shape{1, 2, 2, 2});
    data2->set_friendly_name("data2");        // operation name
    data2->output(0).set_names({"data2_t"});  // tensor names

    auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{data1, data2}, 1);
    concat->set_friendly_name("concat");        // operation name
    concat->output(0).set_names({"concat_t"});  // tensor name

    auto result = std::make_shared<ov::opset8::Result>(concat);
    result->set_friendly_name("result");  // operation name

    auto f = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{data1, data2}, "function_name");
    //! [ngraph:graph]
    return 0;
}
