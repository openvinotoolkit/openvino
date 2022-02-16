// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/runtime/core.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>

int main() {
    std::string model_path;
    std::string input_name;
    //! [ov:preprocess:create]
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(model_path);
    ov::preprocess::PrePostProcessor ppp(model);
    //! [ov:preprocess:create]

    //! [ov:preprocess:tensor]
    ov::preprocess::InputInfo& input = ppp.input(input_name);
    input.tensor()
            .set_element_type(ov::element::u8)
            .set_shape({1, 480, 640, 3})
            .set_layout("NHWC")
            .set_color_format(ov::preprocess::ColorFormat::BGR);
    //! [ov:preprocess:tensor]
    //! [ov:preprocess:model]
    // `model's input` already `knows` it's shape and data type, no need to specify them here
    input.model().set_layout("NCHW");
    //! [ov:preprocess:model]
    //! [ov:preprocess:steps]
    input.preprocess()
            .convert_element_type(ov::element::f32)
            .convert_color(ov::preprocess::ColorFormat::RGB)
            .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR)
            .mean({100.5, 101, 101.5})
            .scale({50., 51., 52.});
            // .convert_layout("NCHW"); // Not needed, such conversion will be added implicitly
    //! [ov:preprocess:steps]
    //! [ov:preprocess:build]
    std::cout << "Dump preprocessor: " << ppp << std::endl;
    model = ppp.build();
    //! [ov:preprocess:build]
    OPENVINO_ASSERT(model, "Model is invalid");
    return 0;
}
