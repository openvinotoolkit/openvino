// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "menu.hpp"

#include <gflags/gflags.h>

#include <iostream>
#include <memory>
#include <openvino/openvino.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/serialize.hpp>
#include <string>

#include "configuration/transformer_configuration_loader_impl.hpp"
#include "logger/logger.hpp"
#include "passes/app_pass_manager.hpp"
#include "transformer/model_transformer_app.hpp"
#include "transformer/model_transformer_gna.hpp"

using namespace transformation_sample;

static const std::string help_message = "Print helper message.";
DEFINE_bool(h, false, help_message.c_str());

std::string config_message = "Path to .json file with transformer application configuration.";
DEFINE_string(config, "", config_message.c_str());

std::string transformation_message = "Name of application defined transformation to be applied on input_model.";
DEFINE_string(transformation, "", transformation_message.c_str());

std::string input_model_message = "Path to .xml file with input model in IR format.";
DEFINE_string(input_model, "", input_model_message.c_str());

std::string output_model_message = "Path to file to export transformend model with .xml extension.";
DEFINE_string(output_model, "", output_model_message.c_str());

static const std::string list_app_transformations_message = "List application defined transformations.";
DEFINE_bool(list_app_transformations, false, list_app_transformations_message.c_str());

const std::string Menu::kWeightsFileExtension = ".bin";
constexpr char Menu::kFileNameDelimiter;

const std::string example_command_config =
    "-input_model model.xml -output_model transformed_model.xml -config configuration.json";
const std::string example_command_transformation =
    "-input_model model.xml -output_model transformed_model.xml -transformations transformation_name";

Menu::Menu(int argc, char* argv[]) {
    init(argc, argv);
}

static void print_usage() {
    log_info() << std::endl;
    log_info() << "gna_transformer_app [OPTION]" << std::endl;
    log_info() << "Options:" << std::endl;
    log_info() << std::endl;
    log_info() << "    -h                            " << help_message << std::endl;
    log_info() << "    -input_model \"<path>\"       " << input_model_message << std::endl;
    log_info() << "    -output_model \"<path>\"      " << output_model_message << std::endl;
    log_info() << "    -config \"<path>\"            " << config_message << std::endl;
    log_info() << "    -transformation \"<string>\"  " << transformation_message << std::endl;
    log_info() << "    -list_app_transformations     " << list_app_transformations_message << std::endl;
    log_info() << std::endl;
    log_info() << "    commands examples:" << transformation_message << std::endl;
    log_info() << std::endl;

    log_info() << example_command_config << std::endl;
    log_info() << example_command_transformation << std::endl;
}

void Menu::execute_action() {
    if (m_action) {
        m_action();
        log_info() << "Transformation successed" << std::endl;
    }
}

void Menu::init(int argc, char* argv[]) {
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        print_usage();
        return;
    }

    if (FLAGS_list_app_transformations) {
        print_app_defined_transformations();
        return;
    }

    if (FLAGS_input_model.empty()) {
        throw WrongArgumentsExeception("Lack of input_model argument. Please check -h option");
    }

    if (FLAGS_output_model.empty()) {
        throw WrongArgumentsExeception("Lack of output_model argument. Please check -h option");
    }

    if (!FLAGS_config.empty()) {
        m_action = prepare_gna_specific_transformations(FLAGS_config, FLAGS_input_model, FLAGS_output_model);
        return;
    }

    if (!FLAGS_transformation.empty()) {
        m_action = prepare_app_specific_transformation(FLAGS_transformation, FLAGS_input_model, FLAGS_output_model);
        return;
    }

    throw WrongArgumentsExeception("Lack of one of arguments: config or transformation. Please check -h option");
}

std::function<void(void)> Menu::prepare_app_specific_transformation(const std::string& transformation_name,
                                                                    const std::string& input_model_path,
                                                                    const std::string& output_model_path) {
    log_info() << "Chosen Application defined transformation: " << transformation_name << "." << std::endl;

    passes::AppPassManager manager;

    auto pass = manager.get_pass(transformation_name);

    std::shared_ptr<ModelTransformer> transformer = std::make_shared<ModelTransformerApp>(std::move(pass));

    return [this, input_model_path, transformer, output_model_path]() {
        run_transformer(input_model_path, *transformer, output_model_path);
    };
}

std::function<void(void)> Menu::prepare_gna_specific_transformations(const std::string& configuration_file_path,
                                                                     const std::string& input_model_path,
                                                                     const std::string& output_model_path) {
    log_info() << "Chosen GNA transformations specified in configuration file" << std::endl;

    auto configuration = TransformerConfigurationLoaderImpl().parse_configuration(configuration_file_path);
    std::shared_ptr<ModelTransformer> transformer = std::make_shared<ModelTransformerGNA>(configuration);

    return [this, input_model_path, transformer, output_model_path]() {
        run_transformer(input_model_path, *transformer, output_model_path);
    };
}

void Menu::print_app_defined_transformations() {
    auto transformations_names = passes::AppPassManager().available_passes_names();

    log_info() << "Available transformations: " << std::endl;

    for (const auto& transformation_name : transformations_names) {
        log_info() << transformation_name << std::endl;
    }
}

void Menu::run_transformer(const std::string input_model_path,
                           ModelTransformer& transformer,
                           const std::string& output_model_path) {
    ov::Core core;
    auto model = core.read_model(input_model_path);
    transformer.transform(model);
    export_model_to_file(model, output_model_path);
}

void Menu::export_model_to_file(std::shared_ptr<ov::Model> model, const std::string& file_path) {
    size_t dot_position = file_path.find_last_of(kFileNameDelimiter);
    if (dot_position == std::string::npos) {
        throw FileNameFormatExeception("File name should have .xml extension, but received: " + file_path);
    }

    std::string weights_file_name = file_path.substr(0, dot_position);
    weights_file_name.append(kWeightsFileExtension);

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::Serialize>(file_path, weights_file_name);
    manager.run_passes(model);
}
