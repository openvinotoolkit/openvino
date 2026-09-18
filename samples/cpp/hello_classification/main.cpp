// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// clang-format off
#include "openvino/openvino.hpp"

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/classification_results.h"
#include "samples/slog.hpp"
#include "format_reader_ptr.h"
// clang-format on

/**
 * @brief Main with support Unicode paths, wide strings
 */
int tmain(int argc, tchar* argv[]) {
    const std::string model_path = argc > 1
        ? std::string(argv[1])
        : "/opt/home/opipikin/workspace/models/TinyLlama-1.1B-Chat-v1.0/openvino_model.xml";
    const std::string device = "CPU";

    using clock = std::chrono::high_resolution_clock;
    auto ms_since = [](clock::time_point start) {
        return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(clock::now() - start).count();
    };

    ov::Core core;

    auto t0 = clock::now();
    std::shared_ptr<ov::Model> model = core.read_model(model_path);
    std::cout << "read_model:          " << ms_since(t0) << " ms" << std::endl;

    t0 = clock::now();
    ov::CompiledModel compiled_model = core.compile_model(model, device);
    std::cout << "compile_model:       " << ms_since(t0) << " ms" << std::endl;

    t0 = clock::now();
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    std::cout << "create_infer_request:" << ms_since(t0) << " ms" << std::endl;

    t0 = clock::now();
    // fill every input with zeros, replacing dynamic dimensions with 1
    for (const auto& input : compiled_model.inputs()) {
        ov::Shape shape;
        for (const auto& dim : input.get_partial_shape())
            shape.push_back(dim.is_static() ? dim.get_length() : 1);
        infer_request.set_tensor(input, ov::Tensor(input.get_element_type(), shape));
    }
    std::cout << "fill_input_data:     " << ms_since(t0) << " ms" << std::endl;

    t0 = clock::now();
    infer_request.infer();
    std::cout << "infer:               " << ms_since(t0) << " ms" << std::endl;

    return EXIT_SUCCESS;
}
