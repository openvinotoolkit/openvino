// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "intel_npu/npu_private_properties.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"

enum class ModelType { SIMPLE, WEIGHTS_SEPARATION, WEIGHTFUL, DYNAMIC, FILE };

ModelType readModelType() {
    std::string choice;
    std::cout << "Choose model type:\n1 - simple\n2 - weights separation\n3 - weightful\n4 - dynamic\n5 - model from "
                 "file\n";
    if (!(std::cin >> choice)) {
        throw std::runtime_error("Failed to read model type");
    }

    if (choice == "1") {
        return ModelType::SIMPLE;
    }
    if (choice == "2") {
        return ModelType::WEIGHTS_SEPARATION;
    }
    if (choice == "3") {
        return ModelType::WEIGHTFUL;
    }
    if (choice == "4") {
        return ModelType::DYNAMIC;
    }
    if (choice == "5") {
        return ModelType::FILE;
    }
    throw std::runtime_error("Invalid model type; enter 1, 2, 3, 4, or 5");
}

std::string readModelPath() {
    std::string path;
    std::cout << "Enter the path of the model (IR .xml or ONNX): ";
    std::cin >> std::ws;
    if (!std::getline(std::cin, path) || path.empty()) {
        throw std::runtime_error("Failed to read the model path");
    }
    // "Copy as path" in Explorer adds quotes
    if (path.size() >= 2 && path.front() == '"' && path.back() == '"') {
        path = path.substr(1, path.size() - 2);
    }
    return path;
}

bool readWeightless() {
    std::string choice;
    std::cout << "Compile the model as:\n1 - weightful\n2 - weightless\n";
    if (!(std::cin >> choice)) {
        throw std::runtime_error("Failed to read the weightless choice");
    }
    if (choice != "1" && choice != "2") {
        throw std::runtime_error("Invalid choice; enter 1 or 2");
    }
    return choice == "2";
}

bool readYesNo(const std::string& question) {
    std::string choice;
    std::cout << question << " (yes/no): ";
    if (!(std::cin >> choice)) {
        throw std::runtime_error("Failed to read the answer to: " + question);
    }
    std::transform(choice.begin(), choice.end(), choice.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (choice == "yes" || choice == "y") {
        return true;
    }
    if (choice == "no" || choice == "n") {
        return false;
    }
    throw std::runtime_error("Invalid choice; enter yes or no");
}

bool readThroughputMode() {
    return readYesNo("Use THROUGHPUT performance mode?");
}

std::string_view getModelTypeName(ModelType type) {
    switch (type) {
    case ModelType::SIMPLE:
        return "simple";
    case ModelType::WEIGHTS_SEPARATION:
        return "weights separation";
    case ModelType::WEIGHTFUL:
        return "weightful";
    case ModelType::DYNAMIC:
        return "dynamic";
    case ModelType::FILE:
        return "file";
    }
    OPENVINO_THROW("Unknown model type");
}

std::shared_ptr<ov::Model> createModel(ModelType type) {
    // The weightful case reuses this exact model, so sizes of both can be compared.
    if (type == ModelType::WEIGHTS_SEPARATION || type == ModelType::WEIGHTFUL) {
        ov::ParameterVector parameters;
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1});
        parameters.push_back(input);
        auto add = std::make_shared<ov::op::v1::Add>(
            input,
            ov::op::v0::Constant::create(ov::element::f16, ov::Shape{50}, {0.5f}));

        for (std::size_t i = 0; i < 10; ++i) {
            auto intermediate_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1});
            parameters.push_back(intermediate_input);
            auto weights = ov::op::v0::Constant::create(
                ov::element::f16, ov::Shape{10, 10, 20, 50}, {static_cast<float>(i)});
            // The one-shot exporter uses this metadata to extract constants into weight-init schedules.
            weights->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] =
                ov::WeightlessCacheAttribute(weights->get_byte_size(), i, weights->get_element_type());
            add = std::make_shared<ov::op::v1::Add>(add, intermediate_input);
            add = std::make_shared<ov::op::v1::Add>(add, weights);
        }
        return std::make_shared<ov::Model>(ov::OutputVector{add}, parameters);
    }

    if (type == ModelType::DYNAMIC) {
        auto input = std::make_shared<ov::op::v0::Parameter>(
            ov::element::f16,
            ov::PartialShape{1, 16, ov::Dimension(10, 1280), ov::Dimension(10, 1280)});
        input->set_friendly_name("input1");
        input->get_output_tensor(0).set_names({"input1"});
        input->set_layout("NCHW");
        auto max_pool = std::make_shared<ov::op::v1::MaxPool>(input,
                                                              ov::Strides{1, 1},
                                                              ov::Shape{0, 0},
                                                              ov::Shape{0, 0},
                                                              ov::Shape{1, 1},
                                                              ov::op::RoundingType::FLOOR,
                                                              ov::op::PadType::EXPLICIT);
        max_pool->set_friendly_name("MaxPool_2");
        auto result = std::make_shared<ov::op::v0::Result>(max_pool);
        result->set_friendly_name("output");
        result->get_output_tensor(0).set_names({"output"});
        result->set_layout("NCHW");
        return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "MaxPool");
    }

    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto relu = std::make_shared<ov::op::v0::Relu>(input);
    return std::make_shared<ov::Model>(ov::OutputVector{relu}, ov::ParameterVector{input});
}

ov::Shape makeInferenceInputShape(const ov::PartialShape& partialShape, ModelType type, std::size_t inferenceIndex) {
    if (type == ModelType::DYNAMIC) {
        // These non-minimum shapes exercise two VM runs without triggering its small-shape backtracking failure.
        return inferenceIndex == 0 ? ov::Shape{1, 16, 720, 1280} : ov::Shape{1, 16, 720, 720};
    }

    ov::Shape shape;
    shape.reserve(partialShape.size());
    for (const auto& dimension : partialShape) {
        auto length = std::max<std::int64_t>(dimension.get_min_length(), 1);
        if (!dimension.is_static() && inferenceIndex > 0 && length < dimension.get_max_length()) {
            ++length;
        }
        shape.push_back(static_cast<std::size_t>(length));
    }
    return shape;
}

// Number of blobs in an exported NPU model: the entry count of a multi-blob container, or 1 for a single blob.
// Mirrors the container layout of the NPU plugin (metadata.hpp):
// [entries][directory: version (u32), entry count (u64), ...][directory size (u64)]["OVNPUC"]
std::size_t countExportedBlobs(const std::string& exported) {
    constexpr std::string_view containerMagic = "OVNPUC";
    constexpr uint32_t supportedContainerVersion = 1;
    constexpr std::size_t footerSize = sizeof(uint64_t) + containerMagic.size();
    if (exported.size() < footerSize ||
        std::string_view(exported).substr(exported.size() - containerMagic.size()) != containerMagic) {
        return 1;
    }

    uint64_t directorySize = 0;
    std::memcpy(&directorySize, exported.data() + exported.size() - footerSize, sizeof(directorySize));
    if (directorySize < sizeof(uint32_t) + sizeof(uint64_t) || directorySize > exported.size() - footerSize) {
        throw std::runtime_error("Malformed blob container: invalid directory size");
    }
    const char* directory = exported.data() + exported.size() - footerSize - directorySize;

    uint32_t version = 0;
    std::memcpy(&version, directory, sizeof(version));
    if (version != supportedContainerVersion) {
        throw std::runtime_error("Unsupported blob container version " + std::to_string(version));
    }
    uint64_t entryCount = 0;
    std::memcpy(&entryCount, directory + sizeof(version), sizeof(entryCount));
    return static_cast<std::size_t>(entryCount);
}

// Replays the examples of the compiler's compilation target query through the public API: the target list is the
// query without a config, and each narrowing query is what compile_model sends for a target plus its properties.
// Returns false if any scenario fails.
bool interrogateOfflineCompilationTargets(ov::Core& core,
                                          const std::string& device,
                                          const std::vector<std::string>& targets) {
    std::cout << "Offline compilation targets for " << device << " (" << targets.size() << "):" << std::endl;
    for (const auto& target : targets) {
        std::cout << "  " << target << std::endl;
    }

    const auto hasTarget = [&targets](const std::string& target) {
        return std::find(targets.begin(), targets.end(), target) != targets.end();
    };

    std::size_t passed = 0, failed = 0, skipped = 0;
    const auto report = [&](const std::string& status, const std::string& name, const std::string& details) {
        std::cout << "[" << status << "] " << name << (details.empty() ? "" : " - " + details) << std::endl;
        (status == "PASS" ? passed : status == "FAIL" ? failed : skipped)++;
    };

    // Query without a config: every platform the compiler can compile for
    for (const auto& expected : {"3720", "4000", "5010", "5020", "6010", "7010", "8010"}) {
        const std::string name = std::string("enumeration lists ") + expected;
        if (hasTarget(expected)) {
            report("PASS", name, "");
        } else {
            report("SKIP", name, "not available in this compiler build");
        }
    }

    const auto model = createModel(ModelType::SIMPLE);
    // Selects HostCompile_Interpreter, whose blobs carry no compatibility descriptor: a container could not pick
    // between its entries at import, so the compiler has to resolve such a config to a single bundle.
    const auto dynamicModel = createModel(ModelType::DYNAMIC);

    struct Scenario {
        std::string name;
        std::string target;  // empty: no ov::compilation_target is passed
        ov::AnyMap properties;
        std::size_t expectedBlobs;  // 0: compile_model is expected to throw
        std::shared_ptr<ov::Model> scenarioModel = nullptr;  // nullptr: the simple model
    };
    const std::vector<Scenario> scenarios = {
        // Baseline - without a compilation target, a platform alone is a plain compile: one blob, never a container
        {"NPU_PLATFORM=6010 only, no COMPILATION_TARGET",
         "",
         {{ov::intel_npu::platform.name(), std::string("6010")}},
         1},

        // A compilation target: one blob per config bundle the compiler resolves it to
        {"COMPILATION_TARGET=6010", "6010", {}, 2},
        {"COMPILATION_TARGET=6010 PERFORMANCE_HINT=LATENCY",
         "6010",
         {{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::LATENCY}},
         2},
        {"COMPILATION_TARGET=6010 PERFORMANCE_HINT=THROUGHPUT",
         "6010",
         {{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::THROUGHPUT}},
         1},
        {"COMPILATION_TARGET=6010 NPU_TILES=1", "6010", {{ov::intel_npu::tiles.name(), 1}}, 1},
        {"COMPILATION_TARGET=6010 dynamic model (HostCompile_Interpreter)", "6010", {}, 1, dynamicModel},

        // An explicit platform selector is allowed alongside the target only if both name the same platform
        {"COMPILATION_TARGET=6010 with a matching NPU_PLATFORM=6010",
         "6010",
         {{ov::intel_npu::platform.name(), std::string("6010")}},
         2},
        {"COMPILATION_TARGET=6010 with a matching NPU_PLATFORM=NPU6010 (other spelling)",
         "6010",
         {{ov::intel_npu::platform.name(), std::string("NPU6010")}},
         2},
        {"COMPILATION_TARGET=6010 with a conflicting NPU_PLATFORM=5010",
         "6010",
         {{ov::intel_npu::platform.name(), std::string("5010")}},
         0},

        // Rejected targets
        // AUTO_DETECT would tie the offline compile to the attached device
        {"COMPILATION_TARGET=AUTO_DETECT", "AUTO_DETECT", {}, 0},
        {"COMPILATION_TARGET=9999 (a target the compiler cannot compile for)", "9999", {}, 0},
        {"COMPILATION_TARGET=6010 NPU_BOGUS_OPTION (an option no one knows)",
         "6010",
         {{"NPU_BOGUS_OPTION", std::string("1")}},
         0},
        {"COMPILATION_TARGET=3720 NPU_ENABLE_STRIDES_FOR (an option 3720 does not support)",
         "3720",
         {{"NPU_ENABLE_STRIDES_FOR", model->get_parameters().front()->get_friendly_name()}},
         0},
    };

    // The action was read with operator>>, which leaves the rest of its line in the input
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    for (std::size_t index = 0; index < scenarios.size(); ++index) {
        const auto& scenario = scenarios[index];
        auto properties = scenario.properties;
        if (!scenario.target.empty()) {
            properties[ov::compilation_target.name()] = scenario.target;
        }

        std::cout << std::endl << "=== Scenario " << index + 1 << "/" << scenarios.size() << ": " << scenario.name;
        // A pause before each scenario, so each result can be read before the next one's output
        std::cout << " - press ENTER to continue" << std::flush;
        std::getchar();
        std::cout << "    compile_model(" << (scenario.scenarioModel ? "dynamic" : "simple") << " model, \"" << device
                  << "\", {";
        for (auto it = properties.begin(); it != properties.end(); ++it) {
            std::cout << (it == properties.begin() ? "" : ", ") << it->first << ": " << it->second.as<std::string>();
        }
        std::cout << "})" << std::endl;
        std::cout << "    expected: "
                  << (scenario.expectedBlobs == 0 ? "compile_model fails"
                                                  : std::to_string(scenario.expectedBlobs) + " blob(s) exported")
                  << std::endl;

        // An expected failure for an unknown platform does not depend on the target list
        if (scenario.expectedBlobs > 0 && !scenario.target.empty() && !hasTarget(scenario.target)) {
            report("SKIP", scenario.name, "target " + scenario.target + " not available in this compiler build");
            continue;
        }

        try {
            auto compiled = core.compile_model(scenario.scenarioModel ? scenario.scenarioModel : model, device, properties);
            std::stringstream exported;
            compiled.export_model(exported);
            const auto blobs = countExportedBlobs(exported.str());
            if (scenario.expectedBlobs == 0) {
                report("FAIL", scenario.name, "compiled into " + std::to_string(blobs) + " blob(s), expected an error");
            } else if (blobs != scenario.expectedBlobs) {
                report("FAIL",
                       scenario.name,
                       std::to_string(blobs) + " blob(s), expected " + std::to_string(scenario.expectedBlobs));
            } else {
                report("PASS", scenario.name, std::to_string(blobs) + " blob(s)");
            }
        } catch (const std::exception& ex) {
            std::string message = ex.what();
            message = message.substr(0, message.find_last_not_of("\r\n ") + 1);
            message = message.substr(message.find_last_of('\n') + 1);
            if (scenario.expectedBlobs == 0) {
                report("PASS", scenario.name, "compile_model failed as expected: " + message);
            } else {
                report("FAIL", scenario.name, "compile_model failed: " + message);
            }
        }
    }

    std::cout << std::endl
              << "Summary: " << passed << " passed, " << failed << " failed, " << skipped << " skipped" << std::endl;
    return failed == 0;
}

int main(int argc, char* argv[]) {
    const std::string device = argc > 1 ? argv[1] : "NPU";
    try {
        ov::Core core;
        const auto offline_targets = core.get_property(device, ov::offline_compilation_targets);

        std::string action;
        std::cout << "Choose action:\n0 - interrogate offline compilation targets\n1 - compile and run a model\n";
        if (!(std::cin >> action)) {
            throw std::runtime_error("Failed to read action");
        }
        if (action == "0") {
            return interrogateOfflineCompilationTargets(core, device, offline_targets) ? 0 : 1;
        }
        if (action != "1") {
            throw std::runtime_error("Invalid action; enter 0 or 1");
        }

        std::string platform;
        std::cout << "Enter the target platform (e.g. 3720, 4000, 5010, 6010): ";
        if (!(std::cin >> platform)) {
            throw std::runtime_error("Failed to read NPU platform number");
        }

        // Check if target is available in the offline compilation targets
        if (std::find(offline_targets.begin(), offline_targets.end(), platform) == offline_targets.end()) {
            throw std::runtime_error("Platform " + platform + " is not present in the offline compilation targets");
        }

        const auto model_type = readModelType();
        const auto model_name = getModelTypeName(model_type);
        std::cout << "Selected model type scenario: " << model_name << std::endl;
        const auto model = model_type == ModelType::FILE ? core.read_model(readModelPath()) : createModel(model_type);
        ov::AnyMap compile_properties;
        compile_properties.emplace(ov::compilation_target.name(), platform);
        std::size_t standard_blob_size = 0;
        const bool weightless =
            model_type == ModelType::FILE ? readWeightless() : model_type == ModelType::WEIGHTS_SEPARATION;
        if (model_type == ModelType::WEIGHTS_SEPARATION) {
            auto standard_compiled_model = core.compile_model(model, device, compile_properties);
            std::stringstream standard_blob;
            standard_compiled_model.export_model(standard_blob);
            standard_blob_size = standard_blob.str().size();
            std::cout << "Standard model blob size: " << standard_blob_size << " bytes" << std::endl;
        }
        if (weightless) {
            compile_properties.emplace(ov::enable_weightless.name(), true);
        }
        // With THROUGHPUT the compiler resolves a multi-SKU target to a single bundle, so no container is produced
        if (model_type == ModelType::FILE && readThroughputMode()) {
            compile_properties.emplace(ov::hint::performance_mode.name(), ov::hint::PerformanceMode::THROUGHPUT);
        }

        // Compile the model for the selected target
        std::cout << "Compiling " << model_name << " model for " << platform << std::endl;
        auto compiled_model = core.compile_model(model, device, compile_properties);
        std::cout << model_name << " model compiled successfully for " << platform << std::endl;

        const auto compiled_properties = compiled_model.get_property(ov::supported_properties);
        std::string requirements;
        if (std::find(compiled_properties.begin(), compiled_properties.end(), ov::runtime_requirements) !=
            compiled_properties.end()) {
            requirements = compiled_model.get_property(ov::runtime_requirements);
            std::cout << "Runtime requirements: " << requirements << std::endl;
        } else {
            std::cout << "Runtime requirements are unavailable for this compiled model." << std::endl;
        }

        // Export: offline, no device involved
        std::stringstream blob;
        compiled_model.export_model(blob);
        const auto exported_blobs = countExportedBlobs(blob.str());
        std::cout << "Model exported successfully for " << platform << ": "
                  << (exported_blobs > 1 ? "container of " + std::to_string(exported_blobs) + " blobs"
                                         : std::string("single blob"))
                  << ", " << blob.str().size() << " bytes" << std::endl;
        if (model_type == ModelType::WEIGHTS_SEPARATION) {
            const auto weights_separated_blob_size = blob.str().size();
            std::cout << "Weights-separated blob size: " << weights_separated_blob_size << " bytes" << std::endl;
            if (weights_separated_blob_size >= standard_blob_size) {
                throw std::runtime_error("Weights-separated blob is not smaller than the standard model blob");
            }
            std::cout << "Verified: weights separation reduced the exported blob size." << std::endl;
        } else if (model_type == ModelType::WEIGHTFUL) {
            std::cout << "Weightful blob size: " << blob.str().size() << " bytes" << std::endl;
        } else if (model_type == ModelType::FILE) {
            std::cout << (weightless ? "Weightless" : "Weightful") << " blob size: " << blob.str().size()
                      << " bytes" << std::endl;
        }

        // Everything below runs on this host's device: optional, since offline compilation does not need one
        if (!readYesNo("Import the exported model and run inference on this host?")) {
            std::cout << "Skipping import and inference on this host." << std::endl;
            return 0;
        }

        std::cout << std::endl << "=== Import and inference on this host" << std::endl;
        bool hostStepFailed = false;
        const auto reportStep = [&hostStepFailed](const std::string& step, bool ok, const std::string& details) {
            std::cout << "[" << (ok ? "PASS" : "FAIL") << "] " << step << (details.empty() ? "" : " - " + details)
                      << std::endl;
            hostStepFailed = hostStepFailed || !ok;
        };
        const auto lastLine = [](const std::string& message) {
            const auto trimmed = message.substr(0, message.find_last_not_of("\r\n ") + 1);
            return trimmed.substr(trimmed.find_last_of('\n') + 1);
        };

        std::string host_architecture;
        try {
            host_architecture = core.get_property(device, ov::device::architecture);
            reportStep("host device", true, device + " " + host_architecture);
        } catch (const std::exception& ex) {
            reportStep("host device", false, "no " + device + " available on this host: " + lastLine(ex.what()));
        }

        if (!requirements.empty()) {
            try {
                const auto check =
                    core.get_property(device, ov::compatibility_check, {{ov::runtime_requirements.name(), requirements}});
                std::ostringstream checkText;
                checkText << check;
                reportStep("compatibility check", true, checkText.str());
            } catch (const std::exception& ex) {
                reportStep("compatibility check", false, lastLine(ex.what()));
            }
        }

        try {
            blob.seekg(0);
            auto imported_model = weightless ? core.import_model(blob, device, {ov::hint::model(model)})
                                             : core.import_model(blob, device);
            reportStep("import", true, "");

            // Exercise the inference
            try {
                if (model_type == ModelType::DYNAMIC && host_architecture != platform) {
                    throw std::runtime_error("Dynamic inference requires an attached NPU with platform " + platform +
                                             "; detected '" + host_architecture + "'");
                }
                auto infer_request = imported_model.create_infer_request();
                for (std::size_t inferenceIndex = 0; inferenceIndex < 2; ++inferenceIndex) {
                    for (std::size_t inputIndex = 0; inputIndex < model->inputs().size(); ++inputIndex) {
                        const auto input = model->input(inputIndex);
                        const auto input_shape = makeInferenceInputShape(input.get_partial_shape(), model_type, inferenceIndex);
                        ov::Tensor input_tensor(input.get_element_type(), input_shape);
                        std::memset(input_tensor.data(), 0, input_tensor.get_byte_size());
                        infer_request.set_input_tensor(inputIndex, input_tensor);
                    }
                    infer_request.infer();
                    reportStep("inference " + std::to_string(inferenceIndex + 1), true, "");
                }
            } catch (const std::exception& ex) {
                reportStep("inference", false, lastLine(ex.what()));
            }
        } catch (const std::exception& ex) {
            reportStep("import", false, lastLine(ex.what()));
        }

        std::cout << "Import and inference on this host " << (hostStepFailed ? "FAILED" : "succeeded") << std::endl;
        return hostStepFailed ? 1 : 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
