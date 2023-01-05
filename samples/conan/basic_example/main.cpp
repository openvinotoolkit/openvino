// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <openvino/openvino.hpp>

int main(int argc, char *argv[])
{
    try
    {
        // -------- Get OpenVINO runtime version --------
        std::cout << ov::get_openvino_version() << std::endl;

        // -------- Parsing and validation of input arguments --------
        if (argc != 2)
        {
            std::cout << "Usage : " << argv[0] << " <path_to_model>" << std::endl;
            return EXIT_FAILURE;
        }

        // -------- Step 1. Initialize OpenVINO Runtime Core --------
        ov::Core core;

        // -------- Step 2. Read model --------
        const std::string model_path = std::string(argv[1]);
        auto model = core.read_model(model_path);

        std::cout << "model name:" << model->get_friendly_name() << std::endl;

        // -------- Step 3. Print inputs and outputs --------
        std::cout << "Inputs:" << std::endl;
        for (const auto &input : model->inputs())
        {
            std::cout << "name: " << input.get_any_name() << " type:" << input.get_element_type() << std::endl;
        }

        std::cout << "Outputs:" << std::endl;
        for (const auto &output : model->outputs())
        {
            std::cout << "name: " << output.get_any_name() << " type:" << output.get_element_type() << std::endl;
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
