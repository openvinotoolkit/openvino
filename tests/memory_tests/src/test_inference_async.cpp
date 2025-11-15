// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/openvino.hpp>

#include "memory_test.hpp"


void do_test(TestContext &test) {
    test.sample("start");

    ov::Core core;
    ov::CompiledModel model = core.compile_model(test.model_path, test.device);
    test.sample("compile_model");

    auto ireq = model.create_infer_request();
    for (auto input: model.inputs()) {
        ireq.set_tensor(input, {input});
    }
    test.sample("fill_inputs");

    ireq.start_async();
    ireq.wait();
    test.sample("inference");
}
