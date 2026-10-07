// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/openvino.hpp>

#include "memory_test.hpp"
#include "common.hpp"


std::vector<std::string> test_samples() {
    return {
        "start",
        "compile_model",
        "fill_inputs",
        "inference"
    };
}


void do_test(memory_tests::Context &test) {
    test.sample("start");

    ov::Core core;
    auto model = core.read_model(test.model_path);
    std::map<size_t, ov::PartialShape> input_shapes;
    for (auto input: model->inputs()) {
        if (input.get_partial_shape().is_static()) {
            continue;
        }
        input_shapes.insert({input.get_index(), get_default_shape(model->input().get_partial_shape())});
    }
    model->reshape(input_shapes);
    ov::CompiledModel compiled_model = core.compile_model(model, test.device);
    test.sample("compile_model");

    auto ireq = compiled_model.create_infer_request();
    for (auto input: compiled_model.inputs()) {
        ireq.set_tensor(input, {input});
    }
    test.sample("fill_inputs");

    ireq.infer();
    test.sample("inference");
}
