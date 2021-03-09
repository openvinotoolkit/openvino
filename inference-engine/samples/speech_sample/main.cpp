// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "speech_sample.hpp"
#include <gflags/gflags.h>
#include <thread>
#include <chrono>
#include <gna/gna_config.hpp>
#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>
#include <inference_engine.hpp>
#include "utils.hpp"

#define MAX_SCORE_DIFFERENCE 0.0001f
#define MAX_VAL_2B_FEAT 16384

using namespace InferenceEngine;

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
typedef std::chrono::duration<float> fsec;

struct InferRequestStruct {
    InferRequest inferRequest;
    int frameIndex;
    uint32_t numFramesThisBatch;
};

void CheckNumberOfInputs(size_t numInputs, size_t numInputFiles) {
    if (numInputs != numInputFiles) {
        throw std::logic_error("Number of network inputs (" + std::to_string(numInputs) + ")"
                               " is not equal to number of  files (" + std::to_string(numInputFiles) + ")");
    }
}

float ScaleFactorForQuantization(void *ptrFloatMemory, float targetMax, uint32_t numElements) {
    float *ptrFloatFeat = reinterpret_cast<float *>(ptrFloatMemory);
    float max = 0.0;
    float scaleFactor;

    for (uint32_t i = 0; i < numElements; i++) {
        if (fabs(ptrFloatFeat[i]) > max) {
            max = fabs(ptrFloatFeat[i]);
        }
    }

    if (max == 0) {
        scaleFactor = 1.0;
    } else {
        scaleFactor = targetMax / max;
    }

    return (scaleFactor);
}

std::vector<std::string> ParseScaleFactors(const std::string& str) {
    std::vector<std::string> scaleFactorInput;

    if (!str.empty()) {
        std::string outStr;
        std::istringstream stream(str);
        int i = 0;
        while (getline(stream, outStr, ',')) {
            auto floatScaleFactor  = std::stof(outStr);
            if (floatScaleFactor <= 0.0f) {
                throw std::logic_error("Scale factor for input #" + std::to_string(i)
                    + " (counting from zero) is out of range (must be positive).");
            }
            scaleFactorInput.push_back(outStr);
            i++;
        }
    } else {
        throw std::logic_error("Scale factor need to be specified via -sf option if you are using -q user");
    }
    return scaleFactorInput;
}

std::vector<std::string> ParseBlobName(std::string str) {
    std::vector<std::string> blobName;
    if (!str.empty()) {
        size_t pos_last = 0;
        size_t pos_next = 0;
        while ((pos_next = str.find(",", pos_last)) != std::string::npos) {
            blobName.push_back(str.substr(pos_last, pos_next - pos_last));
            pos_last = pos_next + 1;
        }
        blobName.push_back(str.substr(pos_last));
    }
    return blobName;
}

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    slog::info << "Parsing input parameters" << slog::endl;

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    bool isDumpMode = !FLAGS_wg.empty() || !FLAGS_we.empty();

    // input not required only in dump mode and if external scale factor provided
    if (FLAGS_i.empty() && (!isDumpMode || FLAGS_q.compare("user") != 0)) {
        if (isDumpMode) {
            throw std::logic_error("In model dump mode either static quantization is used (-i) or user scale"
                                   " factor need to be provided. See -q user option");
        }
        throw std::logic_error("Input file not set. Please use -i.");
    }

    if (FLAGS_m.empty() && FLAGS_rg.empty()) {
        throw std::logic_error("Either IR file (-m) or GNAModel file (-rg) need to be set.");
    }

    if ((!FLAGS_m.empty() && !FLAGS_rg.empty())) {
        throw std::logic_error("Only one of -m and -rg is allowed.");
    }

    std::vector<std::string> supportedDevices = {
            "CPU",
            "GPU",
            "GNA_AUTO",
            "GNA_HW",
            "GNA_SW_EXACT",
            "GNA_SW",
            "GNA_SW_FP32",
            "HETERO:GNA,CPU",
            "HETERO:GNA_HW,CPU",
            "HETERO:GNA_SW_EXACT,CPU",
            "HETERO:GNA_SW,CPU",
            "HETERO:GNA_SW_FP32,CPU",
            "MYRIAD"
    };

    if (std::find(supportedDevices.begin(), supportedDevices.end(), FLAGS_d) == supportedDevices.end()) {
        throw std::logic_error("Specified device is not supported.");
    }

    uint32_t batchSize = (uint32_t) FLAGS_bs;
    if ((batchSize < 1) || (batchSize > 8)) {
        throw std::logic_error("Batch size out of range (1..8).");
    }

    /** default is a static quantization **/
    if ((FLAGS_q.compare("static") != 0) && (FLAGS_q.compare("dynamic") != 0) && (FLAGS_q.compare("user") != 0)) {
        throw std::logic_error("Quantization mode not supported (static, dynamic, user).");
    }

    if (FLAGS_q.compare("dynamic") == 0) {
        throw std::logic_error("Dynamic quantization not yet supported.");
    }

    if (FLAGS_qb != 16 && FLAGS_qb != 8) {
        throw std::logic_error("Only 8 or 16 bits supported.");
    }

    if (FLAGS_nthreads <= 0) {
        throw std::logic_error("Invalid value for 'nthreads' argument. It must be greater that or equal to 0");
    }

    if (FLAGS_cw_r < 0) {
        throw std::logic_error("Invalid value for 'cw_r' argument. It must be greater than or equal to 0");
    }

    if (FLAGS_cw_l < 0) {
        throw std::logic_error("Invalid value for 'cw_l' argument. It must be greater than or equal to 0");
    }

    if (FLAGS_pwl_me < 0.0 || FLAGS_pwl_me > 100.0) {
        throw std::logic_error("Invalid value for 'pwl_me' argument. It must be greater than 0.0 and less than 100.0");
    }

    return true;
}

/**
 * @brief The entry point for inference engine automatic speech recognition sample
 * @file speech_sample/main.cpp
 * @example speech_sample/main.cpp
 */
int main(int argc, char *argv[]) {
    try {
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        auto isFeature = [&](const std::string xFeature) { return FLAGS_d.find(xFeature) != std::string::npos; };

        bool useGna = isFeature("GNA");
        bool useHetero = isFeature("HETERO");
        std::string deviceStr =
                useHetero && useGna ? "HETERO:GNA,CPU" : FLAGS_d.substr(0, (FLAGS_d.find("_")));
        uint32_t batchSize = (FLAGS_cw_r > 0 || FLAGS_cw_l > 0) ? 1 : (uint32_t) FLAGS_bs;


        std::vector<std::string> inputFiles;
        std::vector<uint32_t> numBytesThisUtterance;
        uint32_t numUtterances(0);

        if (!FLAGS_i.empty()) {
            std::istringstream stream(FLAGS_i);
            ArkFile::SetNumBytesForCurrentUtterance(stream, inputFiles, numBytesThisUtterance, numUtterances);
        }
        size_t numInputFiles(inputFiles.size());
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load inference engine -------------------------------------
        slog::info << "Loading Inference Engine" << slog::endl;
        Core ie;

        /** Printing device version **/
        slog::info << "Device info: " << slog::endl;
        std::cout << ie.GetVersions(deviceStr) << std::endl;
        // -----------------------------------------------------------------------------------------------------

        // 2. Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
        slog::info << "Loading network files" << slog::endl;

        CNNNetwork network;
        if (!FLAGS_m.empty()) {
            /** Read network model **/
            network = ie.ReadNetwork(FLAGS_m);
            CheckNumberOfInputs(network.getInputsInfo().size(), numInputFiles);
            // -------------------------------------------------------------------------------------------------

            // --------------------------- 3. Set batch size ---------------------------------------------------
            /** Set batch size.  Unlike in imaging, batching in time (rather than space) is done for speech recognition. **/
            network.setBatchSize(batchSize);
            slog::info << "Batch size is " << std::to_string(network.getBatchSize())
                       << slog::endl;
        }

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Set parameters and scale factors -------------------------------------
        /** Setting parameter for per layer metrics **/
        std::map<std::string, std::string> gnaPluginConfig;
        std::map<std::string, std::string> genericPluginConfig;
        if (useGna) {
            std::string gnaDevice =
                    useHetero ? FLAGS_d.substr(FLAGS_d.find("GNA"), FLAGS_d.find(",") - FLAGS_d.find("GNA")) : FLAGS_d;
            gnaPluginConfig[GNAConfigParams::KEY_GNA_DEVICE_MODE] =
                    gnaDevice.find("_") == std::string::npos ? "GNA_AUTO" : gnaDevice;
        }

        if (FLAGS_pc) {
            genericPluginConfig[PluginConfigParams::KEY_PERF_COUNT] = PluginConfigParams::YES;
        }

        if (FLAGS_q.compare("user") == 0) {
            if (!FLAGS_rg.empty()) {
                slog::warn << "Custom scale factor will be ignored - using scale factor from provided imported gna model: "
                           << FLAGS_rg << slog::endl;
            } else {
                auto scaleFactorInput = ParseScaleFactors(FLAGS_sf);
                if (numInputFiles != scaleFactorInput.size()) {
                    std::string errMessage("Incorrect command line for multiple inputs: "
                        + std::to_string(scaleFactorInput.size()) + " scale factors provided for "
                        + std::to_string(numInputFiles) + " input files.");
                    throw std::logic_error(errMessage);
                }

                for (size_t i = 0; i < scaleFactorInput.size(); ++i) {
                    slog::info << "For input " << i << " using scale factor of " << scaleFactorInput[i] << slog::endl;
                    std::string scaleFactorConfigKey = GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_") + std::to_string(i);
                    gnaPluginConfig[scaleFactorConfigKey] = scaleFactorInput[i];
                }
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
                    ArkFile::GetKaldiArkInfo(inputFileName, 0, &numArrays, &numBytes);
                    ptrFeatures.resize(numBytes);
                    ArkFile::LoadKaldiArkArray(inputFileName,
                        0,
                        name,
                        ptrFeatures,
                        &numFrames,
                        &numFrameElements,
                        &numBytesPerElement);
                    auto floatScaleFactor =
                        ScaleFactorForQuantization(ptrFeatures.data(), MAX_VAL_2B_FEAT, numFrames * numFrameElements);
                    slog::info << "Using scale factor of " << floatScaleFactor << " calculated from first utterance."
                        << slog::endl;
                    std::string scaleFactorConfigKey = GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_") + std::to_string(i);
                    gnaPluginConfig[scaleFactorConfigKey] = std::to_string(floatScaleFactor);
                }
            }
        }

        if (FLAGS_qb == 8) {
            gnaPluginConfig[GNAConfigParams::KEY_GNA_PRECISION] = "I8";
        } else {
            gnaPluginConfig[GNAConfigParams::KEY_GNA_PRECISION] = "I16";
        }

        gnaPluginConfig[GNAConfigParams::KEY_GNA_LIB_N_THREADS] = std::to_string((FLAGS_cw_r > 0 || FLAGS_cw_l > 0) ? 1 : FLAGS_nthreads);
        gnaPluginConfig[GNA_CONFIG_KEY(COMPACT_MODE)] = CONFIG_VALUE(NO);
        gnaPluginConfig[GNA_CONFIG_KEY(PWL_MAX_ERROR_PERCENT)] = std::to_string(FLAGS_pwl_me);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Write model to file --------------------------------------------------
        // Embedded GNA model dumping (for Intel(R) Speech Enabling Developer Kit)
        if (!FLAGS_we.empty()) {
            gnaPluginConfig[GNAConfigParams::KEY_GNA_FIRMWARE_MODEL_IMAGE] = FLAGS_we;
            gnaPluginConfig[GNAConfigParams::KEY_GNA_FIRMWARE_MODEL_IMAGE_GENERATION] = FLAGS_we_gen;
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Loading model to the device ------------------------------------------

        if (useGna) {
            genericPluginConfig.insert(std::begin(gnaPluginConfig), std::end(gnaPluginConfig));
        }
        auto t0 = Time::now();
        std::vector<std::string> outputs;
        ExecutableNetwork executableNet;

        if (!FLAGS_oname.empty()) {
            std::vector<std::string> output_names = ParseBlobName(FLAGS_oname);
            std::vector<size_t> ports;
            for (const auto& outBlobName : output_names) {
                int pos_layer = outBlobName.rfind(":");
                if (pos_layer == -1) {
                    throw std::logic_error(std::string("Output ") + std::string(outBlobName)
                    + std::string(" doesn't have a port"));
                }
                outputs.push_back(outBlobName.substr(0, pos_layer));
                try {
                    ports.push_back(std::stoi(outBlobName.substr(pos_layer + 1)));
                } catch (const std::exception &) {
                    throw std::logic_error("Ports should have integer type");
                }
            }

            for (size_t i = 0; i < outputs.size(); i++) {
                network.addOutput(outputs[i], ports[i]);
            }
        }
        if (!FLAGS_m.empty()) {
            slog::info << "Loading model to the device" << slog::endl;
            executableNet = ie.LoadNetwork(network, deviceStr, genericPluginConfig);
        } else {
            slog::info << "Importing model to the device" << slog::endl;
            executableNet = ie.ImportNetwork(FLAGS_rg.c_str(), deviceStr, genericPluginConfig);
        }
        ms loadTime = std::chrono::duration_cast<ms>(Time::now() - t0);
        slog::info << "Model loading time " << loadTime.count() << " ms" << slog::endl;

        // --------------------------- 7. Exporting gna model using InferenceEngine AOT API---------------------
        if (!FLAGS_wg.empty()) {
            slog::info << "Writing GNA Model to file " << FLAGS_wg << slog::endl;
            t0 = Time::now();
            executableNet.Export(FLAGS_wg);
            ms exportTime = std::chrono::duration_cast<ms>(Time::now() - t0);
            slog::info << "Exporting time " << exportTime.count() << " ms" << slog::endl;
            return 0;
        }

        if (!FLAGS_we.empty()) {
            slog::info << "Exported GNA embedded model to file " << FLAGS_we << slog::endl;
            if (!FLAGS_we_gen.empty()) {
                slog::info << "GNA embedded model export done for GNA generation: " << FLAGS_we_gen << slog::endl;
            }
            return 0;
        }

        std::vector<InferRequestStruct> inferRequests((FLAGS_cw_r > 0 || FLAGS_cw_l > 0) ? 1 : FLAGS_nthreads);
        for (auto& inferRequest : inferRequests) {
            inferRequest = {executableNet.CreateInferRequest(), -1, batchSize};
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 8. Prepare input blobs --------------------------------------------------
        /** Taking information about all topology inputs **/
        ConstInputsDataMap cInputInfo = executableNet.GetInputsInfo();
        CheckNumberOfInputs(cInputInfo.size(), numInputFiles);

        /** Stores all input blobs data **/
        std::vector<Blob::Ptr> ptrInputBlobs;
        if (!FLAGS_iname.empty()) {
            std::vector<std::string> inputNameBlobs = ParseBlobName(FLAGS_iname);
            if (inputNameBlobs.size() != cInputInfo.size()) {
                std::string errMessage(std::string("Number of network inputs ( ") + std::to_string(cInputInfo.size()) +
                                       " ) is not equal to the number of inputs entered in the -iname argument ( " +
                                       std::to_string(inputNameBlobs.size()) + " ).");
                throw std::logic_error(errMessage);
            }
            for (const auto& input : inputNameBlobs) {
                Blob::Ptr blob = inferRequests.begin()->inferRequest.GetBlob(input);
                if (!blob) {
                    std::string errMessage("No blob with name : " + input);
                    throw std::logic_error(errMessage);
                }
                ptrInputBlobs.push_back(blob);
            }
        } else {
            for (const auto& input : cInputInfo) {
                ptrInputBlobs.push_back(inferRequests.begin()->inferRequest.GetBlob(input.first));
            }
        }
        InputsDataMap inputInfo;
        if (!FLAGS_m.empty()) {
            inputInfo = network.getInputsInfo();
        }
        /** Configure input precision if model is loaded from IR **/
        for (auto &item : inputInfo) {
            Precision inputPrecision = Precision::FP32;  // specify Precision::I16 to provide quantized inputs
            item.second->setPrecision(inputPrecision);
        }

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 9. Prepare output blobs -------------------------------------------------
        ConstOutputsDataMap cOutputInfo(executableNet.GetOutputsInfo());
        OutputsDataMap outputInfo;
        if (!FLAGS_m.empty()) {
            outputInfo = network.getOutputsInfo();
        }
        std::vector<Blob::Ptr> ptrOutputBlob;
        if (!outputs.empty()) {
            for (const auto& output : outputs) {
                Blob::Ptr blob = inferRequests.begin()->inferRequest.GetBlob(output);
                if (!blob) {
                    std::string errMessage("No blob with name : " + output);
                    throw std::logic_error(errMessage);
                }
                ptrOutputBlob.push_back(blob);
            }
        } else {
            for (auto& output : cOutputInfo) {
                ptrOutputBlob.push_back(inferRequests.begin()->inferRequest.GetBlob(output.first));
            }
        }

        for (auto &item : outputInfo) {
            DataPtr outData = item.second;
            if (!outData) {
                throw std::logic_error("output data pointer is not valid");
            }

            Precision outputPrecision = Precision::FP32;  // specify Precision::I32 to retrieve quantized outputs
            outData->setPrecision(outputPrecision);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 10. Do inference --------------------------------------------------------
        std::vector<std::string> output_name_files;
        std::vector<std::string> reference_name_files;
        size_t count_file = 1;
        if (!FLAGS_o.empty()) {
            output_name_files = ParseBlobName(FLAGS_o);
            if (output_name_files.size() != outputs.size() && !outputs.empty()) {
                throw std::logic_error("The number of output files is not equal to the number of network outputs.");
            }
            count_file = output_name_files.empty() ? 1 : output_name_files.size();
        }
        if (!FLAGS_r.empty()) {
            reference_name_files = ParseBlobName(FLAGS_r);
            if (reference_name_files.size() != outputs.size() && !outputs.empty()) {
                throw std::logic_error("The number of reference files is not equal to the number of network outputs.");
            }
            count_file = reference_name_files.empty() ? 1 : reference_name_files.size();
        }
        for (size_t next_output = 0; next_output < count_file; next_output++) {
            std::vector<std::vector<uint8_t>> ptrUtterances;
            std::vector<uint8_t> ptrScores;
            std::vector<uint8_t> ptrReferenceScores;
            score_error_t frameError, totalError;

            ptrUtterances.resize(inputFiles.size());

            // initialize memory state before starting
            for (auto &&state : inferRequests.begin()->inferRequest.QueryState()) {
                state.Reset();
            }

            for (uint32_t utteranceIndex = 0; utteranceIndex < numUtterances; ++utteranceIndex) {
                std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> utterancePerfMap;
                std::string uttName;
                uint32_t numFrames(0), n(0);
                std::vector<uint32_t> numFrameElementsInput;

                uint32_t numFramesReference(0), numFrameElementsReference(0), numBytesPerElementReference(0),
                        numBytesReferenceScoreThisUtterance(0);
                auto dims = outputs.empty() ? cOutputInfo.rbegin()->second->getDims() : cOutputInfo[outputs[next_output]]->getDims();
                const auto numScoresPerFrame = std::accumulate(std::begin(dims), std::end(dims), size_t{1}, std::multiplies<size_t>());

                slog::info << "Number scores per frame : " << numScoresPerFrame << slog::endl;

                numFrameElementsInput.resize(numInputFiles);
                for (size_t i = 0; i < inputFiles.size(); i++) {
                    std::vector<uint8_t> ptrUtterance;
                    auto inputFilename = inputFiles[i].c_str();
                    uint32_t currentNumFrames(0), currentNumFrameElementsInput(0), currentNumBytesPerElementInput(0);
                    ArkFile::GetKaldiArkInfo(inputFilename, utteranceIndex, &n, &numBytesThisUtterance[i]);
                    ptrUtterance.resize(numBytesThisUtterance[i]);
                    ArkFile::LoadKaldiArkArray(inputFilename,
                                      utteranceIndex,
                                      uttName,
                                      ptrUtterance,
                                      &currentNumFrames,
                                      &currentNumFrameElementsInput,
                                      &currentNumBytesPerElementInput);
                    if (numFrames == 0) {
                        numFrames = currentNumFrames;
                    } else if (numFrames != currentNumFrames) {
                        std::string errMessage(
                                "Number of frames in ark files is different: " + std::to_string(numFrames) +
                                " and " + std::to_string(currentNumFrames));
                        throw std::logic_error(errMessage);
                    }

                    ptrUtterances[i] = ptrUtterance;
                    numFrameElementsInput[i] = currentNumFrameElementsInput;
                }

                int i = 0;
                for (auto &ptrInputBlob : ptrInputBlobs) {
                    if (ptrInputBlob->size() != numFrameElementsInput[i++] * batchSize) {
                        throw std::logic_error("network input size(" + std::to_string(ptrInputBlob->size()) +
                                               ") mismatch to ark file size (" +
                                               std::to_string(numFrameElementsInput[i - 1] * batchSize) + ")");
                    }
                }

                ptrScores.resize(numFrames * numScoresPerFrame * sizeof(float));
                if (!FLAGS_r.empty()) {
                    std::string refUtteranceName;
                    ArkFile::GetKaldiArkInfo(reference_name_files[next_output].c_str(), utteranceIndex, &n, &numBytesReferenceScoreThisUtterance);
                    ptrReferenceScores.resize(numBytesReferenceScoreThisUtterance);
                    ArkFile::LoadKaldiArkArray(reference_name_files[next_output].c_str(),
                                      utteranceIndex,
                                      refUtteranceName,
                                      ptrReferenceScores,
                                      &numFramesReference,
                                      &numFrameElementsReference,
                                      &numBytesPerElementReference);
                }

                double totalTime = 0.0;

                std::cout << "Utterance " << utteranceIndex << ": " << std::endl;

                Score::ClearScoreError(&totalError);
                totalError.threshold = frameError.threshold = MAX_SCORE_DIFFERENCE;
                auto outputFrame = &ptrScores.front();
                std::vector<uint8_t *> inputFrame;
                for (auto &ut : ptrUtterances) {
                    inputFrame.push_back(&ut.front());
                }

                std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> callPerfMap;

                size_t frameIndex = 0;
                uint32_t numFramesFile = numFrames;
                numFrames += FLAGS_cw_l + FLAGS_cw_r;
                uint32_t numFramesThisBatch{batchSize};

                auto t0 = Time::now();
                auto t1 = t0;

                while (frameIndex <= numFrames) {
                    if (frameIndex == numFrames) {
                        if (std::find_if(inferRequests.begin(),
                                         inferRequests.end(),
                                         [&](InferRequestStruct x) { return (x.frameIndex != -1); }) ==
                            inferRequests.end()) {
                            break;
                        }
                    }

                    bool inferRequestFetched = false;
                    for (auto &inferRequest : inferRequests) {
                        if (frameIndex == numFrames) {
                            numFramesThisBatch = 1;
                        } else {
                            numFramesThisBatch = (numFrames - frameIndex < batchSize) ? (numFrames - frameIndex)
                                                                                      : batchSize;
                        }

                        if (inferRequest.frameIndex != -1) {
                            StatusCode code = inferRequest.inferRequest.Wait(
                                    InferenceEngine::IInferRequest::WaitMode::RESULT_READY);

                            if (code != StatusCode::OK) {
                                if (!useHetero) continue;
                                if (code != StatusCode::INFER_NOT_STARTED) continue;
                            }
                            ConstOutputsDataMap newOutputInfo;
                            if (inferRequest.frameIndex >= 0) {
                                if (!FLAGS_o.empty()) {
                                    outputFrame =
                                            &ptrScores.front() +
                                            numScoresPerFrame * sizeof(float) * (inferRequest.frameIndex);
                                    if (!outputs.empty()) {
                                        newOutputInfo[outputs[next_output]] = cOutputInfo[outputs[next_output]];
                                    } else {
                                        newOutputInfo = cOutputInfo;
                                    }
                                    Blob::Ptr outputBlob = inferRequest.inferRequest.GetBlob(newOutputInfo.rbegin()->first);
                                    MemoryBlob::CPtr moutput = as<MemoryBlob>(outputBlob);

                                    if (!moutput) {
                                        throw std::logic_error("We expect output to be inherited from MemoryBlob, "
                                                               "but in fact we were not able to cast output to MemoryBlob");
                                    }
                                    // locked memory holder should be alive all time while access to its buffer happens
                                    auto moutputHolder = moutput->rmap();
                                    auto byteSize =
                                            numScoresPerFrame * sizeof(float);
                                    std::memcpy(outputFrame,
                                                moutputHolder.as<const void *>(),
                                                byteSize);
                                }
                                if (!FLAGS_r.empty()) {
                                    if (!outputs.empty()) {
                                        newOutputInfo[outputs[next_output]] = cOutputInfo[outputs[next_output]];
                                    } else {
                                        newOutputInfo = cOutputInfo;
                                    }
                                    Blob::Ptr outputBlob = inferRequest.inferRequest.GetBlob(newOutputInfo.rbegin()->first);
                                    MemoryBlob::CPtr moutput = as<MemoryBlob>(outputBlob);
                                    if (!moutput) {
                                        throw std::logic_error("We expect output to be inherited from MemoryBlob, "
                                                               "but in fact we were not able to cast output to MemoryBlob");
                                    }
                                    // locked memory holder should be alive all time while access to its buffer happens
                                    auto moutputHolder = moutput->rmap();
                                    Score::CompareScores(moutputHolder.as<float *>(),
                                                  &ptrReferenceScores[inferRequest.frameIndex *
                                                                      numFrameElementsReference *
                                                                      numBytesPerElementReference],
                                                  &frameError,
                                                  inferRequest.numFramesThisBatch,
                                                  numFrameElementsReference);
                                    Score::UpdateScoreError(&frameError, &totalError);
                                }
                                if (FLAGS_pc) {
                                    // retrieve new counters
                                    PerformanceCounters::GetPerformanceCounters(inferRequest.inferRequest, callPerfMap);
                                    // summarize retrieved counters with all previous
                                    PerformanceCounters::SumPerformanceCounters(callPerfMap, utterancePerfMap);
                                }
                            }
                        }

                        if (frameIndex == numFrames) {
                            inferRequest.frameIndex = -1;
                            continue;
                        }

                        if (FLAGS_iname.empty()) {
                            size_t num_files = FLAGS_iname.empty() ? numInputFiles : ptrInputBlobs.size();
                            for (size_t i = 0; i < num_files; ++i) {
                                MemoryBlob::Ptr minput = as<MemoryBlob>(ptrInputBlobs[i]);
                                if (!minput) {
                                    slog::err << "We expect ptrInputBlobs[" << i
                                              << "] to be inherited from MemoryBlob, " <<
                                              "but in fact we were not able to cast input blob to MemoryBlob"
                                              << slog::endl;
                                    return 1;
                                }
                                // locked memory holder should be alive all time while access to its buffer happens
                                auto minputHolder = minput->wmap();

                                std::memcpy(minputHolder.as<void *>(),
                                            inputFrame[i],
                                            minput->byteSize());
                            }
                        }

                        int index = static_cast<int>(frameIndex) - (FLAGS_cw_l + FLAGS_cw_r);
                        inferRequest.inferRequest.StartAsync();
                        inferRequest.frameIndex = index < 0 ? -2 : index;
                        inferRequest.numFramesThisBatch = numFramesThisBatch;

                        frameIndex += numFramesThisBatch;
                        for (size_t j = 0; j < inputFiles.size(); j++) {
                            if (FLAGS_cw_l > 0 || FLAGS_cw_r > 0) {
                                int idx = frameIndex - FLAGS_cw_l;
                                if (idx > 0 && idx < static_cast<int>(numFramesFile)) {
                                    inputFrame[j] += sizeof(float) * numFrameElementsInput[j] * numFramesThisBatch;
                                } else if (idx >= static_cast<int>(numFramesFile)) {
                                    inputFrame[j] = &ptrUtterances[j].front() +
                                                    (numFramesFile - 1) * sizeof(float) * numFrameElementsInput[j] *
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
                for (auto &&state : inferRequests.begin()->inferRequest.QueryState()) {
                    state.Reset();
                }

                if (!FLAGS_o.empty()) {
                    bool shouldAppend = (utteranceIndex == 0) ? false : true;
                    ArkFile::SaveKaldiArkArray(output_name_files[next_output].c_str(), shouldAppend, uttName, &ptrScores.front(),
                                      numFramesFile, numScoresPerFrame);
                }

                /** Show performance results **/
                std::cout << "Total time in Infer (HW and SW):\t" << totalTime << " ms"
                          << std::endl;
                std::cout << "Frames in utterance:\t\t\t" << numFrames << " frames"
                          << std::endl;
                std::cout << "Average Infer time per frame:\t\t" << totalTime / static_cast<double>(numFrames) << " ms"
                          << std::endl;
                if (FLAGS_pc) {
                    // print
                    PerformanceCounters::PrintPerformanceCounters(utterancePerfMap, frameIndex, std::cout, getFullDeviceName(ie, FLAGS_d));
                }
                if (!FLAGS_r.empty()) {
                    Printresults::PrintReferenceCompareResults(totalError, numFrames, std::cout);
                }
                std::cout << "End of Utterance " << utteranceIndex << std::endl << std::endl;
            }
        }
        // -----------------------------------------------------------------------------------------------------
    }
    catch (const std::exception &error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened" << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}
