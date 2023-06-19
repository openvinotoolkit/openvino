// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <string>

#include "transformer/model_transformer.hpp"

class Menu {
public:
    class WrongArgumentsExeception : public std::runtime_error {
    public:
        WrongArgumentsExeception(const std::string& message) : std::runtime_error(message) {}
    };

    class FileNameFormatExeception : public std::runtime_error {
    public:
        FileNameFormatExeception(const std::string& message) : std::runtime_error(message) {}
    };

    Menu(int argc, char* argv[]);
    void execute_action();

private:
    static constexpr char kFileNameDelimiter = '.';
    static const std::string kWeightsFileExtension;

    void init(int argc, char* argv[]);
    std::function<void(void)> prepare_gna_specific_transformations(const std::string& configuration_file_path,
                                                                   const std::string& input_model_path,
                                                                   const std::string& output_model_path);
    std::function<void(void)> prepare_app_specific_transformation(const std::string& transformation_name,
                                                                  const std::string& input_model_path,
                                                                  const std::string& output_model_path);
    void run_transformer(const std::string input_model_path,
                         transformation_sample::ModelTransformer& transformer,
                         const std::string& output_model_path);

    void export_model_to_file(std::shared_ptr<ov::Model> model, const std::string& file_path);

    void print_app_defined_transformations();

    std::function<void(void)> m_action;
};
