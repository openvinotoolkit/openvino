// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "openvino/frontend/gguf/extension/genai.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/openvino.hpp"

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 5) {
            std::cerr << "Usage: gguf_extension_runner EXTENSION_LIBRARY MODEL.gguf [TOKENS] [--stateful]\n";
            return 2;
        }
        const bool stateful = argc >= 4 && std::string(argv[argc - 1]) == "--stateful";
        const auto argument_count = argc - (stateful ? 1 : 0);
        if (argument_count > 4)
            throw std::invalid_argument("Expected --stateful after TOKENS");
        const auto tokens = argument_count == 4 ? std::stoul(argv[3]) : 3;
        if (!tokens)
            throw std::invalid_argument("TOKENS must be positive");
        std::shared_ptr<ov::frontend::FrontEnd> frontend = std::make_shared<ov::frontend::gguf::FrontEnd>();
        frontend->add_extension(std::string(argv[1]));
        if (stateful) {
            frontend->add_extension(std::make_shared<ov::frontend::gguf::GenAIExtension>());
        }
        auto model = frontend->convert(frontend->load(std::string(argv[2])));
        auto compiled = ov::Core{}.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32));
        auto request = compiled.create_infer_request();
        for (const auto& input : compiled.inputs()) {
            const auto name = input.get_names().empty() ? input.get_node()->get_friendly_name() : input.get_any_name();
            auto shape = input.get_partial_shape();
            if (shape.rank().is_dynamic())
                throw std::runtime_error("The example runner requires a known input rank");
            ov::Shape concrete;
            for (const auto& dimension : shape)
                concrete.push_back(dimension.is_static() ? dimension.get_length() : tokens);
            if (name == "input_ids" || name == "position_ids" || name == "attention_mask" || name == "beam_idx")
                concrete[0] = 1;
            ov::Tensor tensor(input.get_element_type(), concrete);
            if (tensor.get_element_type() == ov::element::f32)
                std::fill_n(tensor.data<float>(),
                            tensor.get_size(),
                            name.find("mask") != std::string::npos ? 0.f : 1.f);
            else if (tensor.get_element_type() == ov::element::i32)
                std::fill_n(tensor.data<int32_t>(), tensor.get_size(), 0);
            else if (tensor.get_element_type() == ov::element::i64) {
                const int64_t value = name == "token_len_per_seq" ? static_cast<int64_t>(tokens)
                                      : name == "attention_mask"  ? 1
                                                                  : 0;
                std::fill_n(tensor.data<int64_t>(), tensor.get_size(), value);
            } else
                throw std::runtime_error("Unsupported example input type for " + name);
            if (name == "position_ids" && tensor.get_element_type() == ov::element::i64)
                for (size_t i = 0; i < tensor.get_size(); ++i)
                    tensor.data<int64_t>()[i] = i;
            request.set_tensor(input, tensor);
        }
        request.infer();
        for (const auto& output : compiled.outputs()) {
            auto tensor = request.get_tensor(output);
            if (tensor.get_element_type() == ov::element::f32)
                for (size_t i = 0; i < tensor.get_size(); ++i)
                    if (!std::isfinite(tensor.data<const float>()[i]))
                        throw std::runtime_error("Non-finite output");
            std::cout << (output.get_names().empty() ? "output" : output.get_any_name()) << " " << tensor.get_shape()
                      << " " << tensor.get_element_type();
            if (tensor.get_element_type() == ov::element::f32 && tensor.get_size())
                std::cout << " first=" << tensor.data<const float>()[0];
            std::cout << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
