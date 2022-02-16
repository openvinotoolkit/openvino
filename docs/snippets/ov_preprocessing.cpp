// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/runtime/core.hpp>
//! [ov:preprocess:include]
#include <openvino/core/preprocess/pre_post_process.hpp>
//! [ov:preprocess:include]

int main() {
    std::string model_path;
    std::string input_name;
    //! [ov:preprocess:create]
    using namespace ov;
    using namespace ov::preprocess;
    Core core;
    auto model = core.read_model(model_path);
    PrePostProcessor ppp(model);
    //! [ov:preprocess:create]

    //! [ov:preprocess:tensor]
    auto& input = ppp.input(input_name);
    input.tensor()
            .set_element_type(element::u8)
            .set_shape({1, 480, 640, 3})
            .set_layout("NHWC")
            .set_color_format(ColorFormat::BGR);
    //! [ov:preprocess:tensor]
    //! [ov:preprocess:model]
    // `model's input` already `knows` it's shape and data type, no need to specify them here
    input.model().set_layout("NCHW");
    //! [ov:preprocess:model]
    //! [ov:preprocess:steps]
    input.preprocess()
            .convert_element_type(element::f32)
            .convert_color(ColorFormat::RGB)
            .resize(ResizeAlgorithm::RESIZE_LINEAR)
            .mean({100.5, 101, 101.5})
            .scale({50., 51., 52.});
            // .convert_layout("NCHW"); // Not needed, such conversion will be added implicitly
    //! [ov:preprocess:steps]
    //! [ov:preprocess:build]
    std::cout << "Dump preprocessor: " << ppp << std::endl;
    model = ppp.build();
    //! [ov:preprocess:build]
    return 0;
}
