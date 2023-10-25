// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <time.h>

#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// clang-format off
#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gna/properties.hpp>

#include <samples/args_helper.hpp>
#include <samples/slog.hpp>

#include "fileutils.hpp"
#include "speech_sample.hpp"
#include "utils.hpp"
// clang-format on

using namespace ov::preprocess;

/**
 * @brief The entry point for OpenVINO Runtime automatic speech recognition sample
 * @file speech_sample/main.cpp
 * @example speech_sample/main.cpp
 */
int main(int argc, char* argv[]) {
    try {
        // ------------------------------ Get OpenVINO Runtime version ----------------------------------------------
        slog::info << "OpenVINO runtime: " << ov::get_openvino_version() << slog::endl;

        // ------------------------------ Parsing and validation of input arguments ---------------------------------
        if (!parse_and_check_command_line(argc, argv)) {
            return 0;
        }
        BaseFile* file;
        BaseFile* fileOutput;
        ArkFile arkFile;
        NumpyFile numpyFile;
        std::pair<std::string, std::vector<std::string>> input_data;
        if (!FLAGS_i.empty())
            input_data = parse_parameters(FLAGS_i);
        auto extInputFile = fileExt(input_data.first);
        if (extInputFile == "ark") {
            file = &arkFile;
        } else if (extInputFile == "npz") {
            file = &numpyFile;
        } else {
            throw std::logic_error("Invalid input file");
        }
        std::vector<std::string> inputFiles;
        std::vector<uint32_t> numBytesThisUtterance;
        uint32_t numUtterances(0);
        if (!input_data.first.empty()) {
            std::string outStr;
            std::istringstream stream(input_data.first);
            uint32_t currentNumUtterances(0), currentNumBytesThisUtterance(0);
            while (getline(stream, outStr, ',')) {
                std::string filename(fileNameNoExt(outStr) + "." + extInputFile);
                inputFiles.push_back(filename);
                file->get_file_info(filename.c_str(), 0, &currentNumUtterances, &currentNumBytesThisUtterance);
                if (numUtterances == 0) {
                    numUtterances = currentNumUtterances;
                } else if (currentNumUtterances != numUtterances) {
                    throw std::logic_error(
                        "Incorrect input files. Number of utterance must be the same for all input files");
                }
                numBytesThisUtterance.push_back(currentNumBytesThisUtterance);
            }
        }
        size_t numInputFiles(inputFiles.size());

        // --------------------------- Step 1. Initialize OpenVINO Runtime core and read model
        // -------------------------------------
        ov::Core core;
        try {
            const auto& gnaLibraryVersion = core.get_property("GNA", ov::intel_gna::library_full_version);
            slog::info << "Detected GNA Library: " << gnaLibraryVersion << slog::endl;
        } catch (std::exception& e) {
            slog::info << "Cannot detect GNA Library version, exception: " << e.what() << slog::endl;
        }
        slog::info << "Loading model files:" << slog::endl << FLAGS_m << slog::endl;
        uint32_t batchSize = (FLAGS_cw_r > 0 || FLAGS_cw_l > 0 || !FLAGS_bs) ? 1 : (uint32_t)FLAGS_bs;
        std::shared_ptr<ov::Model> model;
        // --------------------------- Processing custom outputs ---------------------------------------------
        const auto output_data = parse_parameters(FLAGS_o);
        const auto reference_data = parse_parameters(FLAGS_r);

        const auto outputs = get_first_non_empty(output_data.second, reference_data.second);

        // ------------------------------ Preprocessing ------------------------------------------------------
        // the preprocessing steps can be done only for loaded network and are not applicable for the imported network
        // (already compiled)
        if (!FLAGS_m.empty()) {
            const auto outputs_with_ports = parse_to_extract_port(outputs);
            model = core.read_model(FLAGS_m);
            for (const auto& output_with_port : outputs_with_ports) {
                auto output = model->add_output(output_with_port.first, output_with_port.second);
                output.set_names({output_with_port.first + ":" + std::to_string(output_with_port.second)});
            }
            check_number_of_inputs(model->inputs().size(), numInputFiles);
            ov::preprocess::PrePostProcessor proc(model);
            const auto& inputs = model->inputs();
            std::map<std::string, std::string> custom_layouts;
            if (!FLAGS_layout.empty()) {
                custom_layouts = parse_input_layouts(FLAGS_layout, inputs);
            }
            for (const auto& input : inputs) {
                const auto& item_name = input.get_any_name();
                auto& in = proc.input(item_name);
                in.tensor().set_element_type(ov::element::f32);
                // Explicitly set inputs layout
                if (custom_layouts.count(item_name) > 0) {
                    in.model().set_layout(ov::Layout(custom_layouts.at(item_name)));
                }
            }
            for (size_t i = 0; i < model->outputs().size(); i++) {
                proc.output(i).tensor().set_element_type(ov::element::f32);
            }
            model = proc.build();
            if (FLAGS_bs) {
                if (FLAGS_layout.empty() &&
                    std::any_of(inputs.begin(), inputs.end(), [](const ov::Output<ov::Node>& i) {
                        return ov::layout::get_layout(i).empty();
                    })) {
                    throw std::logic_error(
                        "-bs option is set to " + std::to_string(FLAGS_bs) +
                        " but model does not contain layout information for any input. Please "
                        "specify it explicitly using -layout option. For example, input1[NCHW], input2[NC] or [NC]");
                } else {
                    ov::set_batch(model, batchSize);
                }
            }
        }
        // ------------------------------ Get Available Devices ------------------------------------------------------
        auto isFeature = [&](const std::string xFeature) {
            return FLAGS_d.find(xFeature) != std::string::npos;
        };
        bool useGna = isFeature("GNA");
        bool useHetero = isFeature("HETERO");
        std::string deviceStr = useHetero && useGna ? "HETERO:GNA,CPU" : FLAGS_d.substr(0, (FLAGS_d.find("_")));
        // -----------------------------------------------------------------------------------------------------
        // --------------------------- Set parameters and scale factors -------------------------------------
        /** Setting parameter for per layer metrics **/
        ov::AnyMap gnaPluginConfig;
        ov::AnyMap genericPluginConfig;
        if (useGna) {
            std::string gnaDevice =
                useHetero ? FLAGS_d.substr(FLAGS_d.find("GNA"), FLAGS_d.find(",") - FLAGS_d.find("GNA")) : FLAGS_d;
            auto parse_gna_device = [&](const std::string& device) -> ov::intel_gna::ExecutionMode {
                ov::intel_gna::ExecutionMode mode;
                std::stringstream ss(device);
                ss >> mode;
                return mode;
            };
            gnaPluginConfig[ov::intel_gna::execution_mode.name()] = gnaDevice.find("_") == std::string::npos
                                                                        ? ov::intel_gna::ExecutionMode::AUTO
                                                                        : parse_gna_device(gnaDevice);
        }
        if (FLAGS_pc) {
            genericPluginConfig.emplace(ov::enable_profiling(true));
        }
        if (FLAGS_q.compare("user") == 0) {
            if (!FLAGS_rg.empty()) {
                std::string errMessage("Custom scale factor can not be set for imported gna model: " + FLAGS_rg);
                throw std::logic_error(errMessage);
            } else {
                auto scale_factors_per_input = parse_scale_factors(model->inputs(), FLAGS_sf);
                if (numInputFiles != scale_factors_per_input.size()) {
                    std::string errMessage("Incorrect command line for multiple inputs: " +
                                           std::to_string(scale_factors_per_input.size()) +
                                           " scale factors provided for " + std::to_string(numInputFiles) +
                                           " input files.");
                    throw std::logic_error(errMessage);
                }
                for (auto&& sf : scale_factors_per_input) {
                    slog::info << "For input " << sf.first << " using scale factor of " << sf.second << slog::endl;
                }
                gnaPluginConfig[ov::intel_gna::scale_factors_per_input.name()] = scale_factors_per_input;
            }
        } else {
            // "static" quantization with calculated scale factor
            if (!FLAGS_rg.empty()) {
                slog::info << "Using scale factor from provided imported gna model: " << FLAGS_rg << slog::endl;
            } else {
                std::map<std::string, float> scale_factors_per_input;
                for (size_t i = 0; i < numInputFiles; i++) {
                    auto inputFileName = inputFiles[i].c_str();
                    std::string name;
                    std::vector<uint8_t> ptrFeatures;
                    uint32_t numArrays(0), numBytes(0), numFrames(0), numFrameElements(0), numBytesPerElement(0);
                    file->get_file_info(inputFileName, 0, &numArrays, &numBytes);
                    ptrFeatures.resize(numBytes);
                    file->load_file(inputFileName,
                                    0,
                                    name,
                                    ptrFeatures,
                                    &numFrames,
                                    &numFrameElements,
                                    &numBytesPerElement);
                    auto floatScaleFactor = scale_factor_for_quantization(ptrFeatures.data(),
                                                                          MAX_VAL_2B_FEAT,
                                                                          numFrames * numFrameElements);
                    slog::info << "Using scale factor of " << floatScaleFactor << " calculated from first utterance."
                               << slog::endl;
                    scale_factors_per_input[strip_name(model->input(i).get_any_name())] = floatScaleFactor;
                }
                gnaPluginConfig[ov::intel_gna::scale_factors_per_input.name()] = scale_factors_per_input;
            }
        }
        gnaPluginConfig[ov::hint::inference_precision.name()] = (FLAGS_qb == 8) ? ov::element::i8 : ov::element::i16;
        const std::unordered_map<std::string, ov::intel_gna::HWGeneration> StringHWGenerationMap{
            {"GNA_TARGET_1_0", ov::intel_gna::HWGeneration::GNA_1_0},
            {"GNA_TARGET_1_0_E", ov::intel_gna::HWGeneration::GNA_1_0_E},
            {"GNA_TARGET_2_0", ov::intel_gna::HWGeneration::GNA_2_0},
            {"GNA_TARGET_3_0", ov::intel_gna::HWGeneration::GNA_3_0},
            {"GNA_TARGET_3_1", ov::intel_gna::HWGeneration::GNA_3_1},
            {"GNA_TARGET_3_5", ov::intel_gna::HWGeneration::GNA_3_5},
            {"GNA_TARGET_3_5_E", ov::intel_gna::HWGeneration::GNA_3_5_E},
            {"GNA_TARGET_3_6", ov::intel_gna::HWGeneration::GNA_3_6},
            {"GNA_TARGET_4_0", ov::intel_gna::HWGeneration::GNA_4_0}};
        auto parse_target = [&](const std::string& target) -> ov::intel_gna::HWGeneration {
            auto hw_target = ov::intel_gna::HWGeneration::UNDEFINED;
            const auto key_iter = StringHWGenerationMap.find(target);
            if (key_iter != StringHWGenerationMap.end()) {
                hw_target = key_iter->second;
            } else if (!target.empty()) {
                slog::warn << "Unsupported target: " << target << slog::endl;
            }
            return hw_target;
        };

        gnaPluginConfig[ov::intel_gna::execution_target.name()] = parse_target(FLAGS_exec_target);
        gnaPluginConfig[ov::intel_gna::compile_target.name()] = parse_target(FLAGS_compile_target);
        gnaPluginConfig[ov::intel_gna::memory_reuse.name()] = !FLAGS_memory_reuse_off;
        gnaPluginConfig[ov::intel_gna::pwl_max_error_percent.name()] = FLAGS_pwl_me;
        gnaPluginConfig[ov::log::level.name()] = FLAGS_log;
        // -----------------------------------------------------------------------------------------------------
        // --------------------------- Write model to file --------------------------------------------------
        // Embedded GNA model dumping (for Intel(R) Speech Enabling Developer Kit)
        if (!FLAGS_we.empty()) {
            gnaPluginConfig[ov::intel_gna::firmware_model_image_path.name()] = FLAGS_we;
        }
        // -----------------------------------------------------------------------------------------------------
        // --------------------------- Step 2. Loading model to the device ------------------------------------------
        if (useGna) {
            if (useHetero) {
                genericPluginConfig.insert(ov::device::properties("GNA", gnaPluginConfig));
            } else {
                genericPluginConfig.insert(std::begin(gnaPluginConfig), std::end(gnaPluginConfig));
            }
        }
        auto t0 = Time::now();
        ms loadTime = std::chrono::duration_cast<ms>(Time::now() - t0);
        slog::info << "Model loading time " << loadTime.count() << " ms" << slog::endl;
        ov::CompiledModel executableNet;
        if (!FLAGS_m.empty()) {
            slog::info << "Loading model to the device " << FLAGS_d << slog::endl;
            executableNet = core.compile_model(model, deviceStr, genericPluginConfig);
        } else {
            slog::info << "Importing model to the device" << slog::endl;
            std::ifstream streamrq(FLAGS_rg, std::ios_base::binary | std::ios_base::in);
            if (!streamrq.is_open()) {
                throw std::runtime_error("Cannot open model file " + FLAGS_rg);
            }
            executableNet = core.import_model(streamrq, deviceStr, genericPluginConfig);
            // loading batch from exported model
            const auto& imported_inputs = executableNet.inputs();
            if (std::any_of(imported_inputs.begin(), imported_inputs.end(), [](const ov::Output<const ov::Node>& i) {
                    return ov::layout::get_layout(i).empty();
                })) {
                slog::warn << "No batch dimension was found at any input, assuming batch to be 1." << slog::endl;
                batchSize = 1;
            } else {
                for (auto& info : imported_inputs) {
                    auto imported_layout = ov::layout::get_layout(info);
                    if (ov::layout::has_batch(imported_layout)) {
                        batchSize = (uint32_t)info.get_shape()[ov::layout::batch_idx(imported_layout)];
                        break;
                    }
                }
            }
        }
        // --------------------------- Exporting gna model using OpenVINO API---------------------
        if (!FLAGS_wg.empty()) {
            slog::info << "Writing GNA Model to file " << FLAGS_wg << slog::endl;
            t0 = Time::now();
            std::ofstream streamwq(FLAGS_wg, std::ios_base::binary | std::ios::out);
            executableNet.export_model(streamwq);
            ms exportTime = std::chrono::duration_cast<ms>(Time::now() - t0);
            slog::info << "Exporting time " << exportTime.count() << " ms" << slog::endl;
            return 0;
        }
        if (!FLAGS_we.empty()) {
            slog::info << "Exported GNA embedded model to file " << FLAGS_we << slog::endl;
            if (!FLAGS_compile_target.empty()) {
                slog::info << "GNA embedded model target: " << FLAGS_compile_target << slog::endl;
            }
            return 0;
        }
        // ---------------------------------------------------------------------------------------------------------
        // --------------------------- Step 3. Create infer request
        // --------------------------------------------------
        std::vector<InferRequestStruct> inferRequests(1);

        for (auto& inferRequest : inferRequests) {
            inferRequest = {executableNet.create_infer_request(), -1, batchSize};
        }
        // --------------------------- Step 4. Configure input & output
        // --------------------------------------------------
        std::vector<ov::Tensor> ptrInputBlobs;
        auto cInputInfo = executableNet.inputs();
        check_number_of_inputs(cInputInfo.size(), numInputFiles);
        if (!input_data.second.empty()) {
            std::vector<std::string> inputNameBlobs = input_data.second;
            if (inputNameBlobs.size() != cInputInfo.size()) {
                std::string errMessage(std::string("Number of network inputs ( ") + std::to_string(cInputInfo.size()) +
                                       " ) is not equal to the number of inputs entered in the -i argument ( " +
                                       std::to_string(inputNameBlobs.size()) + " ).");
                throw std::logic_error(errMessage);
            }
            for (const auto& input : inputNameBlobs) {
                ov::Tensor blob = inferRequests.begin()->inferRequest.get_tensor(input);
                if (!blob) {
                    std::string errMessage("No blob with name : " + input);
                    throw std::logic_error(errMessage);
                }
                ptrInputBlobs.push_back(blob);
            }
        } else {
            for (const auto& input : cInputInfo) {
                ptrInputBlobs.push_back(inferRequests.begin()->inferRequest.get_tensor(input));
            }
        }
        std::vector<std::string> output_name_files;
        std::vector<std::string> reference_name_files;
        size_t count_file = 1;
        if (!output_data.first.empty()) {
            output_name_files = convert_str_to_vector(output_data.first);
            if (output_name_files.size() != outputs.size() && outputs.size()) {
                throw std::logic_error("The number of output files is not equal to the number of network outputs.");
            }
            count_file = output_name_files.size();
            if (executableNet.outputs().size() > 1 && output_data.second.empty() && count_file == 1) {
                throw std::logic_error("-o is ambiguous: the model has multiple outputs but only one file provided "
                                       "without output name specification");
            }
        }
        if (!reference_data.first.empty()) {
            reference_name_files = convert_str_to_vector(reference_data.first);
            if (reference_name_files.size() != outputs.size() && outputs.size()) {
                throw std::logic_error("The number of reference files is not equal to the number of network outputs.");
            }
            count_file = reference_name_files.size();
            if (executableNet.outputs().size() > 1 && reference_data.second.empty() && count_file == 1) {
                throw std::logic_error("-r is ambiguous: the model has multiple outputs but only one file provided "
                                       "without output name specification");
            }
        }
        if (count_file > executableNet.outputs().size()) {
            throw std::logic_error(
                "The number of output/reference files is not equal to the number of network outputs.");
        }
        // -----------------------------------------------------------------------------------------------------
        // --------------------------- Step 5. Do inference --------------------------------------------------------
        std::vector<std::vector<uint8_t>> ptrUtterances;
        const auto effective_outputs_size = outputs.size() ? outputs.size() : executableNet.outputs().size();
        std::vector<std::vector<uint8_t>> vectorPtrScores(effective_outputs_size);
        std::vector<uint16_t> numScoresPerOutput(effective_outputs_size);

        std::vector<std::vector<uint8_t>> vectorPtrReferenceScores(reference_name_files.size());
        std::vector<ScoreErrorT> vectorFrameError(reference_name_files.size()),
            vectorTotalError(reference_name_files.size());
        ptrUtterances.resize(inputFiles.size());
        // initialize memory state before starting
        for (auto&& state : inferRequests.begin()->inferRequest.query_state()) {
            state.reset();
        }
        /** Work with each utterance **/
        for (uint32_t utteranceIndex = 0; utteranceIndex < numUtterances; ++utteranceIndex) {
            std::map<std::string, ov::ProfilingInfo> utterancePerfMap;
            uint64_t totalNumberOfRunsOnHw = 0;
            std::string uttName;
            uint32_t numFrames(0), n(0);
            std::vector<uint32_t> numFrameElementsInput;
            std::vector<uint32_t> numFramesReference(reference_name_files.size()),
                numFrameElementsReference(reference_name_files.size()),
                numBytesPerElementReference(reference_name_files.size()),
                numBytesReferenceScoreThisUtterance(reference_name_files.size());

            /** Get information from input file for current utterance **/
            numFrameElementsInput.resize(numInputFiles);
            for (size_t i = 0; i < inputFiles.size(); i++) {
                std::vector<uint8_t> ptrUtterance;
                auto inputFilename = inputFiles[i].c_str();
                uint32_t currentNumFrames(0), currentNumFrameElementsInput(0), currentNumBytesPerElementInput(0);
                file->get_file_info(inputFilename, utteranceIndex, &n, &numBytesThisUtterance[i]);
                ptrUtterance.resize(numBytesThisUtterance[i]);
                file->load_file(inputFilename,
                                utteranceIndex,
                                uttName,
                                ptrUtterance,
                                &currentNumFrames,
                                &currentNumFrameElementsInput,
                                &currentNumBytesPerElementInput);
                if (numFrames == 0) {
                    numFrames = currentNumFrames;
                } else if (numFrames != currentNumFrames) {
                    std::string errMessage("Number of frames in input files is different: " +
                                           std::to_string(numFrames) + " and " + std::to_string(currentNumFrames));
                    throw std::logic_error(errMessage);
                }
                ptrUtterances[i] = ptrUtterance;
                numFrameElementsInput[i] = currentNumFrameElementsInput;
            }
            int i = 0;
            for (auto& ptrInputBlob : ptrInputBlobs) {
                if (ptrInputBlob.get_size() != numFrameElementsInput[i++] * batchSize) {
                    throw std::logic_error("network input size(" + std::to_string(ptrInputBlob.get_size()) +
                                           ") mismatch to input file size (" +
                                           std::to_string(numFrameElementsInput[i - 1] * batchSize) + ")");
                }
            }

            double totalTime = 0.0;

            for (size_t errorIndex = 0; errorIndex < vectorFrameError.size(); errorIndex++) {
                clear_score_error(&vectorTotalError[errorIndex]);
                vectorTotalError[errorIndex].threshold = vectorFrameError[errorIndex].threshold = MAX_SCORE_DIFFERENCE;
            }

            std::vector<uint8_t*> inputFrame;
            for (auto& ut : ptrUtterances) {
                inputFrame.push_back(&ut.front());
            }
            std::map<std::string, ov::ProfilingInfo> callPerfMap;
            size_t frameIndex = 0;
            uint32_t numFramesFile = numFrames;
            numFrames += FLAGS_cw_l + FLAGS_cw_r;
            uint32_t numFramesThisBatch{batchSize};
            auto t0 = Time::now();
            auto t1 = t0;

            BaseFile* fileReferenceScores;
            std::string refUtteranceName;

            if (!reference_data.first.empty()) {
                /** Read file with reference scores **/
                auto exReferenceScoresFile = fileExt(reference_data.first);
                if (exReferenceScoresFile == "ark") {
                    fileReferenceScores = &arkFile;
                } else if (exReferenceScoresFile == "npz") {
                    fileReferenceScores = &numpyFile;
                } else {
                    throw std::logic_error("Invalid Reference Scores file");
                }
                for (size_t next_output = 0; next_output < count_file; next_output++) {
                    if (fileReferenceScores != nullptr) {
                        fileReferenceScores->get_file_info(reference_name_files[next_output].c_str(),
                                                           utteranceIndex,
                                                           &n,
                                                           &numBytesReferenceScoreThisUtterance[next_output]);
                        vectorPtrReferenceScores[next_output].resize(numBytesReferenceScoreThisUtterance[next_output]);
                        fileReferenceScores->load_file(reference_name_files[next_output].c_str(),
                                                       utteranceIndex,
                                                       refUtteranceName,
                                                       vectorPtrReferenceScores[next_output],
                                                       &numFramesReference[next_output],
                                                       &numFrameElementsReference[next_output],
                                                       &numBytesPerElementReference[next_output]);
                    }
                }
            }

            while (frameIndex <= numFrames) {
                if (frameIndex == numFrames) {
                    if (std::find_if(inferRequests.begin(), inferRequests.end(), [&](InferRequestStruct x) {
                            return (x.frameIndex != -1);
                        }) == inferRequests.end()) {
                        break;
                    }
                }
                bool inferRequestFetched = false;
                /** Start inference loop **/
                for (auto& inferRequest : inferRequests) {
                    if (frameIndex == numFrames) {
                        numFramesThisBatch = 1;
                    } else {
                        numFramesThisBatch =
                            (numFrames - frameIndex < batchSize) ? (numFrames - frameIndex) : batchSize;
                    }

                    /* waits until inference result becomes available */
                    if (inferRequest.frameIndex != -1) {
                        inferRequest.inferRequest.wait();
                        if (inferRequest.frameIndex >= 0)
                            for (size_t next_output = 0; next_output < count_file; next_output++) {
                                const auto output_name = outputs.size() > next_output
                                                             ? outputs[next_output]
                                                             : executableNet.output(next_output).get_any_name();
                                auto dims = executableNet.output(output_name).get_shape();
                                numScoresPerOutput[next_output] = std::accumulate(std::begin(dims),
                                                                                  std::end(dims),
                                                                                  size_t{1},
                                                                                  std::multiplies<size_t>());

                                vectorPtrScores[next_output].resize(numFramesFile * numScoresPerOutput[next_output] *
                                                                    sizeof(float));

                                if (!FLAGS_o.empty()) {
                                    /* Prepare output data for save to file in future */
                                    auto outputFrame = &vectorPtrScores[next_output].front() +
                                                       numScoresPerOutput[next_output] * sizeof(float) *
                                                           (inferRequest.frameIndex) / batchSize;

                                    ov::Tensor outputBlob =
                                        inferRequest.inferRequest.get_tensor(executableNet.output(output_name));
                                    // locked memory holder should be alive all time while access to its buffer happens
                                    auto byteSize = numScoresPerOutput[next_output] * sizeof(float);
                                    std::memcpy(outputFrame, outputBlob.data<float>(), byteSize);
                                }
                                if (!FLAGS_r.empty()) {
                                    /** Compare output data with reference scores **/
                                    ov::Tensor outputBlob =
                                        inferRequest.inferRequest.get_tensor(executableNet.output(output_name));

                                    if (numScoresPerOutput[next_output] / numFrameElementsReference[next_output] ==
                                        batchSize) {
                                        compare_scores(
                                            outputBlob.data<float>(),
                                            &vectorPtrReferenceScores[next_output]
                                                                     [inferRequest.frameIndex *
                                                                      numFrameElementsReference[next_output] *
                                                                      numBytesPerElementReference[next_output]],
                                            &vectorFrameError[next_output],
                                            inferRequest.numFramesThisBatch,
                                            numFrameElementsReference[next_output]);
                                        update_score_error(&vectorFrameError[next_output],
                                                           &vectorTotalError[next_output]);
                                    } else {
                                        throw std::logic_error("Number of output and reference frames does not match.");
                                    }
                                }
                                if (FLAGS_pc) {
                                    // retrieve new counters
                                    get_performance_counters(inferRequest.inferRequest, callPerfMap);
                                    // summarize retrieved counters with all previous
                                    sum_performance_counters(callPerfMap, utterancePerfMap, totalNumberOfRunsOnHw);
                                }
                            }
                        // -----------------------------------------------------------------------------------------------------
                    }
                    if (frameIndex == numFrames) {
                        inferRequest.frameIndex = -1;
                        continue;
                    }
                    ptrInputBlobs.clear();
                    if (input_data.second.empty()) {
                        for (auto& input : cInputInfo) {
                            ptrInputBlobs.push_back(inferRequest.inferRequest.get_tensor(input));
                        }
                    } else {
                        std::vector<std::string> inputNameBlobs = input_data.second;
                        for (const auto& input : inputNameBlobs) {
                            ov::Tensor blob = inferRequests.begin()->inferRequest.get_tensor(input);
                            if (!blob) {
                                std::string errMessage("No blob with name : " + input);
                                throw std::logic_error(errMessage);
                            }
                            ptrInputBlobs.push_back(blob);
                        }
                    }

                    /** Iterate over all the input blobs **/
                    for (size_t i = 0; i < numInputFiles; ++i) {
                        ov::Tensor minput = ptrInputBlobs[i];
                        if (!minput) {
                            std::string errMessage("We expect ptrInputBlobs[" + std::to_string(i) +
                                                   "] to be inherited from Tensor, " +
                                                   "but in fact we were not able to cast input to Tensor");
                            throw std::logic_error(errMessage);
                        }
                        memcpy(minput.data(),
                               inputFrame[i],
                               numFramesThisBatch * numFrameElementsInput[i] * sizeof(float));
                        // Used to infer fewer frames than the batch size
                        if (batchSize != numFramesThisBatch) {
                            memset(minput.data<float>() + numFramesThisBatch * numFrameElementsInput[i],
                                   0,
                                   (batchSize - numFramesThisBatch) * numFrameElementsInput[i]);
                        }
                    }
                    // -----------------------------------------------------------------------------------------------------
                    int index = static_cast<int>(frameIndex) - (FLAGS_cw_l + FLAGS_cw_r);
                    /* Starting inference in asynchronous mode*/
                    inferRequest.inferRequest.start_async();
                    inferRequest.frameIndex = index < 0 ? -2 : index;
                    inferRequest.numFramesThisBatch = numFramesThisBatch;
                    frameIndex += numFramesThisBatch;
                    for (size_t j = 0; j < inputFiles.size(); j++) {
                        if (FLAGS_cw_l > 0 || FLAGS_cw_r > 0) {
                            int idx = frameIndex - FLAGS_cw_l;
                            if (idx > 0 && idx < static_cast<int>(numFramesFile)) {
                                inputFrame[j] += sizeof(float) * numFrameElementsInput[j] * numFramesThisBatch;
                            } else if (idx >= static_cast<int>(numFramesFile)) {
                                inputFrame[j] = &ptrUtterances[j].front() + (numFramesFile - 1) * sizeof(float) *
                                                                                numFrameElementsInput[j] *
                                                                                numFramesThisBatch;
                            } else if (idx <= 0) {
                                inputFrame[j] = &ptrUtterances[j].front();
                            }
                        } else {
                            inputFrame[j] += sizeof(float) * numFrameElementsInput[j] * numFramesThisBatch;
                        }
                    }
                    inferRequestFetched |= true;
                }
                /** Inference was finished for current frame **/
                if (!inferRequestFetched) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
            }
            t1 = Time::now();
            fsec fs = t1 - t0;
            ms d = std::chrono::duration_cast<ms>(fs);
            totalTime += d.count();
            // resetting state between utterances
            for (auto&& state : inferRequests.begin()->inferRequest.query_state()) {
                state.reset();
            }
            // -----------------------------------------------------------------------------------------------------

            // --------------------------- Step 6. Process output
            // -------------------------------------------------------

            /** Show performance results **/
            std::cout << "Utterance " << utteranceIndex << ": " << std::endl;
            std::cout << "Total time in Infer (HW and SW):\t" << totalTime << " ms" << std::endl;
            std::cout << "Frames in utterance:\t\t\t" << numFrames << " frames" << std::endl;
            std::cout << "Average Infer time per frame:\t\t" << totalTime / static_cast<double>(numFrames) << " ms\n"
                      << std::endl;

            if (FLAGS_pc) {
                // print performance results
                print_performance_counters(utterancePerfMap,
                                           frameIndex,
                                           std::cout,
                                           getFullDeviceName(core, FLAGS_d),
                                           totalNumberOfRunsOnHw,
                                           FLAGS_d);
            }

            for (size_t next_output = 0; next_output < count_file; next_output++) {
                if (!FLAGS_o.empty()) {
                    auto exOutputScoresFile = fileExt(output_data.first);
                    if (exOutputScoresFile == "ark") {
                        fileOutput = &arkFile;
                    } else if (exOutputScoresFile == "npz") {
                        fileOutput = &numpyFile;
                    } else {
                        throw std::logic_error("Invalid Reference Scores file");
                    }
                    /* Save output data to file */
                    bool shouldAppend = (utteranceIndex == 0) ? false : true;
                    fileOutput->save_file(output_name_files[next_output].c_str(),
                                          shouldAppend,
                                          uttName,
                                          &vectorPtrScores[next_output].front(),
                                          numFramesFile,
                                          numScoresPerOutput[next_output] / batchSize);
                }
                if (!FLAGS_r.empty()) {
                    // print statistical score error
                    const auto output_name = outputs.size() > next_output
                                                 ? outputs[next_output]
                                                 : executableNet.output(next_output).get_any_name();
                    std::cout << "Output name: " << output_name << std::endl;
                    std::cout << "Number scores per frame: " << numScoresPerOutput[next_output] / batchSize << std::endl
                              << std::endl;
                    print_reference_compare_results(vectorTotalError[next_output], numFrames, std::cout);
                }
            }
        }
    } catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    } catch (...) {
        slog::err << "Unknown/internal exception happened" << slog::endl;
        return 1;
    }
    slog::info << "Execution successful" << slog::endl;
    return 0;
}
