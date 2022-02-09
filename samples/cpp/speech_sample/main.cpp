// Copyright (C) 2018-2022 Intel Corporation
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
#include <gna/gna_config.hpp>

#include <samples/args_helper.hpp>
#include <samples/slog.hpp>

#include "fileutils.hpp"
#include "speech_sample.hpp"
#include "utils.hpp"
// clang-format on

using namespace ov::preprocess;

/**
 * @brief The entry point for inference engine automatic speech recognition sample
 * @file speech_sample/main.cpp
 * @example speech_sample/main.cpp
 */
int main(int argc, char* argv[]) {
    try {
        // ------------------------------ Get Inference Engine version ----------------------------------------------
        slog::info << "OpenVINO runtime: " << ov::get_openvino_version() << slog::endl;

        // ------------------------------ Parsing and validation of input arguments ---------------------------------
        if (!parse_and_check_command_line(argc, argv)) {
            return 0;
        }
        BaseFile* file;
        BaseFile* fileOutput;
        ArkFile arkFile;
        NumpyFile numpyFile;
        auto extInputFile = fileExt(FLAGS_i);
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
        if (!FLAGS_i.empty()) {
            std::string outStr;
            std::istringstream stream(FLAGS_i);
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

        // --------------------------- Step 1. Initialize inference engine core and read model
        // -------------------------------------
        ov::Core core;
        slog::info << "Loading model files:" << slog::endl << FLAGS_m << slog::endl;
        uint32_t batchSize = (FLAGS_cw_r > 0 || FLAGS_cw_l > 0) ? 1 : (uint32_t)FLAGS_bs;
        std::shared_ptr<ov::Model> model;
        std::vector<std::string> outputs;
        std::vector<size_t> ports;
        // --------------------------- Processing custom outputs ---------------------------------------------
        if (!FLAGS_oname.empty()) {
            std::vector<std::string> output_names = convert_str_to_vector(FLAGS_oname);
            for (const auto& output_name : output_names) {
                auto pos_layer = output_name.rfind(":");
                if (pos_layer == std::string::npos) {
                    throw std::logic_error("Output " + output_name + " doesn't have a port");
                }
                outputs.push_back(output_name.substr(0, pos_layer));
                try {
                    ports.push_back(std::stoi(output_name.substr(pos_layer + 1)));
                } catch (const std::exception&) {
                    throw std::logic_error("Ports should have integer type");
                }
            }
        }
        // ------------------------------ Preprocessing ------------------------------------------------------
        // the preprocessing steps can be done only for loaded network and are not applicable for the imported network
        // (already compiled)
        if (!FLAGS_m.empty()) {
            model = core.read_model(FLAGS_m);
            if (!outputs.empty()) {
                for (size_t i = 0; i < outputs.size(); i++) {
                    auto output = model->add_output(outputs[i], ports[i]);
                    output.set_names({outputs[i] + ":" + std::to_string(ports[i])});
                }
            }
            check_number_of_inputs(model->inputs().size(), numInputFiles);
            const ov::Layout tensor_layout{"NC"};
            ov::preprocess::PrePostProcessor proc(model);
            for (int i = 0; i < model->inputs().size(); i++) {
                proc.input(i).tensor().set_element_type(ov::element::f32).set_layout(tensor_layout);
            }
            for (int i = 0; i < model->outputs().size(); i++) {
                proc.output(i).tensor().set_element_type(ov::element::f32);
            }
            model = proc.build();
            ov::set_batch(model, batchSize);
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
            gnaPluginConfig[InferenceEngine::GNAConfigParams::KEY_GNA_DEVICE_MODE] =
                gnaDevice.find("_") == std::string::npos ? "GNA_AUTO" : gnaDevice;
        }
        if (FLAGS_pc) {
            genericPluginConfig.emplace(ov::enable_profiling(true));
        }
        if (FLAGS_q.compare("user") == 0) {
            if (!FLAGS_rg.empty()) {
                slog::warn << "Custom scale factor will be used for imported gna model: " << FLAGS_rg << slog::endl;
            }
            auto scaleFactorInput = parse_scale_factors(FLAGS_sf);
            if (numInputFiles != scaleFactorInput.size()) {
                std::string errMessage(
                    "Incorrect command line for multiple inputs: " + std::to_string(scaleFactorInput.size()) +
                    " scale factors provided for " + std::to_string(numInputFiles) + " input files.");
                throw std::logic_error(errMessage);
            }
            for (size_t i = 0; i < scaleFactorInput.size(); ++i) {
                slog::info << "For input " << i << " using scale factor of " << scaleFactorInput[i] << slog::endl;
                std::string scaleFactorConfigKey = GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_") + std::to_string(i);
                gnaPluginConfig[scaleFactorConfigKey] = scaleFactorInput[i];
            }
        } else {
            // "static" quantization with calculated scale factor
            if (!FLAGS_rg.empty()) {
                slog::info << "Using scale factor from provided imported gna model: " << FLAGS_rg << slog::endl;
            } else {
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
                    std::string scaleFactorConfigKey =
                        GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_") + std::to_string(i);
                    gnaPluginConfig[scaleFactorConfigKey] = std::to_string(floatScaleFactor);
                }
            }
        }
        if (FLAGS_qb == 8) {
            gnaPluginConfig[InferenceEngine::GNAConfigParams::KEY_GNA_PRECISION] = "I8";
        } else {
            gnaPluginConfig[InferenceEngine::GNAConfigParams::KEY_GNA_PRECISION] = "I16";
        }
        gnaPluginConfig[InferenceEngine::GNAConfigParams::KEY_GNA_EXEC_TARGET] = FLAGS_exec_target;
        gnaPluginConfig[InferenceEngine::GNAConfigParams::KEY_GNA_COMPILE_TARGET] = FLAGS_compile_target;
        gnaPluginConfig[GNA_CONFIG_KEY(COMPACT_MODE)] = CONFIG_VALUE(NO);
        IE_SUPPRESS_DEPRECATED_START
        gnaPluginConfig[GNA_CONFIG_KEY(PWL_MAX_ERROR_PERCENT)] = std::to_string(FLAGS_pwl_me);
        IE_SUPPRESS_DEPRECATED_END
        // -----------------------------------------------------------------------------------------------------
        // --------------------------- Write model to file --------------------------------------------------
        // Embedded GNA model dumping (for Intel(R) Speech Enabling Developer Kit)
        if (!FLAGS_we.empty()) {
            IE_SUPPRESS_DEPRECATED_START
            gnaPluginConfig[InferenceEngine::GNAConfigParams::KEY_GNA_FIRMWARE_MODEL_IMAGE] = FLAGS_we;
            IE_SUPPRESS_DEPRECATED_END
        }
        // -----------------------------------------------------------------------------------------------------
        // --------------------------- Step 2. Loading model to the device ------------------------------------------
        if (useGna) {
            genericPluginConfig.insert(std::begin(gnaPluginConfig), std::end(gnaPluginConfig));
        }
        auto t0 = Time::now();
        ms loadTime = std::chrono::duration_cast<ms>(Time::now() - t0);
        slog::info << "Model loading time " << loadTime.count() << " ms" << slog::endl;
        slog::info << "Loading model to the device " << FLAGS_d << slog::endl;
        ov::CompiledModel executableNet;
        if (!FLAGS_m.empty()) {
            slog::info << "Loading model to the device" << slog::endl;
            executableNet = core.compile_model(model, deviceStr, genericPluginConfig);
        } else {
            slog::info << "Importing model to the device" << slog::endl;
            std::ifstream streamrq(FLAGS_rg, std::ios_base::binary | std::ios_base::in);
            if (!streamrq.is_open()) {
                throw std::runtime_error("Cannot open model file " + FLAGS_rg);
            }
            executableNet = core.import_model(streamrq, deviceStr, genericPluginConfig);
        }
        // --------------------------- Exporting gna model using InferenceEngine AOT API---------------------
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
            return 0;
        }
        // ---------------------------------------------------------------------------------------------------------
        // --------------------------- Step 3. Create infer request --------------------------------------------------
        std::vector<InferRequestStruct> inferRequests(1);

        for (auto& inferRequest : inferRequests) {
            inferRequest = {executableNet.create_infer_request(), -1, batchSize};
        }
        // --------------------------- Step 4. Configure input & output
        // --------------------------------------------------
        std::vector<ov::Tensor> ptrInputBlobs;
        auto cInputInfo = executableNet.inputs();
        check_number_of_inputs(cInputInfo.size(), numInputFiles);
        if (!FLAGS_iname.empty()) {
            std::vector<std::string> inputNameBlobs = convert_str_to_vector(FLAGS_iname);
            if (inputNameBlobs.size() != cInputInfo.size()) {
                std::string errMessage(std::string("Number of network inputs ( ") + std::to_string(cInputInfo.size()) +
                                       " ) is not equal to the number of inputs entered in the -iname argument ( " +
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
        if (!FLAGS_o.empty()) {
            output_name_files = convert_str_to_vector(FLAGS_o);
            if (output_name_files.size() != outputs.size() && !outputs.empty()) {
                throw std::logic_error("The number of output files is not equal to the number of network outputs.");
            }
            count_file = output_name_files.empty() ? 1 : output_name_files.size();
        }
        if (!FLAGS_r.empty()) {
            reference_name_files = convert_str_to_vector(FLAGS_r);
            if (reference_name_files.size() != outputs.size() && !outputs.empty()) {
                throw std::logic_error("The number of reference files is not equal to the number of network outputs.");
            }
            count_file = reference_name_files.empty() ? 1 : reference_name_files.size();
        }
        // -----------------------------------------------------------------------------------------------------
        // --------------------------- Step 5. Do inference --------------------------------------------------------
        for (size_t next_output = 0; next_output < count_file; next_output++) {
            std::vector<std::vector<uint8_t>> ptrUtterances;
            std::vector<uint8_t> ptrScores;
            std::vector<uint8_t> ptrReferenceScores;
            ScoreErrorT frameError, totalError;
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
                uint32_t numFramesReference(0), numFrameElementsReference(0), numBytesPerElementReference(0),
                    numBytesReferenceScoreThisUtterance(0);
                auto dims = executableNet.outputs()[0].get_shape();
                const auto numScoresPerFrame =
                    std::accumulate(std::begin(dims), std::end(dims), size_t{1}, std::multiplies<size_t>());
                slog::info << "Number scores per frame : " << numScoresPerFrame << slog::endl;
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
                ptrScores.resize(numFrames * numScoresPerFrame * sizeof(float));
                if (!FLAGS_r.empty()) {
                    /** Read file with reference scores **/
                    BaseFile* fileReferenceScores;
                    auto exReferenceScoresFile = fileExt(FLAGS_r);
                    if (exReferenceScoresFile == "ark") {
                        fileReferenceScores = &arkFile;
                    } else if (exReferenceScoresFile == "npz") {
                        fileReferenceScores = &numpyFile;
                    } else {
                        throw std::logic_error("Invalid Reference Scores file");
                    }
                    std::string refUtteranceName;
                    fileReferenceScores->get_file_info(reference_name_files[next_output].c_str(),
                                                       utteranceIndex,
                                                       &n,
                                                       &numBytesReferenceScoreThisUtterance);
                    ptrReferenceScores.resize(numBytesReferenceScoreThisUtterance);
                    fileReferenceScores->load_file(reference_name_files[next_output].c_str(),
                                                   utteranceIndex,
                                                   refUtteranceName,
                                                   ptrReferenceScores,
                                                   &numFramesReference,
                                                   &numFrameElementsReference,
                                                   &numBytesPerElementReference);
                }
                double totalTime = 0.0;
                std::cout << "Utterance " << utteranceIndex << ": " << std::endl;
                clear_score_error(&totalError);
                totalError.threshold = frameError.threshold = MAX_SCORE_DIFFERENCE;
                auto outputFrame = &ptrScores.front();
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
                            if (inferRequest.frameIndex >= 0) {
                                if (!FLAGS_o.empty()) {
                                    /* Prepare output data for save to file in future */
                                    outputFrame = &ptrScores.front() +
                                                  numScoresPerFrame * sizeof(float) * (inferRequest.frameIndex);

                                    ov::Tensor outputBlob =
                                        inferRequest.inferRequest.get_tensor(executableNet.outputs()[0]);
                                    if (!outputs.empty()) {
                                        outputBlob =
                                            inferRequest.inferRequest.get_tensor(executableNet.output(FLAGS_oname));
                                    }
                                    // locked memory holder should be alive all time while access to its buffer happens
                                    auto byteSize = numScoresPerFrame * sizeof(float);
                                    std::memcpy(outputFrame, outputBlob.data<float>(), byteSize);
                                }
                                if (!FLAGS_r.empty()) {
                                    /** Compare output data with reference scores **/
                                    ov::Tensor outputBlob =
                                        inferRequest.inferRequest.get_tensor(executableNet.outputs()[0]);
                                    if (!FLAGS_oname.empty())
                                        outputBlob =
                                            inferRequest.inferRequest.get_tensor(executableNet.output(FLAGS_oname));
                                    compare_scores(
                                        outputBlob.data<float>(),
                                        &ptrReferenceScores[inferRequest.frameIndex * numFrameElementsReference *
                                                            numBytesPerElementReference],
                                        &frameError,
                                        inferRequest.numFramesThisBatch,
                                        numFrameElementsReference);
                                    update_score_error(&frameError, &totalError);
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
                        // -----------------------------------------------------------------------------------------------------
                        int index = static_cast<int>(frameIndex) - (FLAGS_cw_l + FLAGS_cw_r);
                        for (int i = 0; i < executableNet.inputs().size(); i++) {
                            inferRequest.inferRequest.set_input_tensor(
                                i,
                                ov::Tensor(ov::element::f32, executableNet.inputs()[i].get_shape(), inputFrame[i]));
                        }
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

                if (!FLAGS_o.empty()) {
                    auto exOutputScoresFile = fileExt(FLAGS_o);
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
                                          &ptrScores.front(),
                                          numFramesFile,
                                          numScoresPerFrame);
                }
                /** Show performance results **/
                std::cout << "Total time in Infer (HW and SW):\t" << totalTime << " ms" << std::endl;
                std::cout << "Frames in utterance:\t\t\t" << numFrames << " frames" << std::endl;
                std::cout << "Average Infer time per frame:\t\t" << totalTime / static_cast<double>(numFrames) << " ms"
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
                if (!FLAGS_r.empty()) {
                    // print statistical score error
                    print_reference_compare_results(totalError, numFrames, std::cout);
                }
                std::cout << "End of Utterance " << utteranceIndex << std::endl << std::endl;
                // -----------------------------------------------------------------------------------------------------
            }
        }
        // -----------------------------------------------------------------------------------------------------
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
