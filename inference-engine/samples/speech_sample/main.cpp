// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "speech_sample.hpp"

#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <memory>
#include <map>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <utility>
#include <time.h>
#include <thread>
#include <chrono>
#include <limits>
#include <iomanip>
#include <inference_engine.hpp>
#include <gna/gna_config.hpp>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>

#ifndef ALIGN
#define ALIGN(memSize, pad)   ((static_cast<int>((memSize) + pad - 1) / pad) * pad)
#endif
#define MAX_SCORE_DIFFERENCE 0.0001f
#define MAX_VAL_2B_FEAT 16384

using namespace InferenceEngine;

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
typedef std::chrono::duration<float> fsec;
typedef struct {
    uint32_t numScores;
    uint32_t numErrors;
    float threshold;
    float maxError;
    float rmsError;
    float sumError;
    float sumRmsError;
    float sumSquaredError;
    float maxRelError;
    float sumRelError;
    float sumSquaredRelError;
} score_error_t;

void GetKaldiArkInfo(const char *fileName,
                     uint32_t numArrayToFindSize,
                     uint32_t *ptrNumArrays,
                     uint32_t *ptrNumMemoryBytes) {
    uint32_t numArrays = 0;
    uint32_t numMemoryBytes = 0;

    std::ifstream in_file(fileName, std::ios::binary);
    if (in_file.good()) {
        while (!in_file.eof()) {
            std::string line;
            uint32_t numRows = 0u, numCols = 0u, num_bytes = 0u;
            std::getline(in_file, line, '\0');  // read variable length name followed by space and NUL
            std::getline(in_file, line, '\4');  // read "BFM" followed by space and control-D
            if (line.compare("BFM ") != 0) {
                break;
            }
            in_file.read(reinterpret_cast<char *>(&numRows), sizeof(uint32_t));  // read number of rows
            std::getline(in_file, line, '\4');                                   // read control-D
            in_file.read(reinterpret_cast<char *>(&numCols), sizeof(uint32_t));  // read number of columns
            num_bytes = numRows * numCols * sizeof(float);
            in_file.seekg(num_bytes, in_file.cur);                               // read data

            if (numArrays == numArrayToFindSize) {
                numMemoryBytes += num_bytes;
            }
            numArrays++;
        }
        in_file.close();
    } else {
        fprintf(stderr, "Failed to open %s for reading in GetKaldiArkInfo()!\n", fileName);
        exit(-1);
    }

    if (ptrNumArrays != NULL) *ptrNumArrays = numArrays;
    if (ptrNumMemoryBytes != NULL) *ptrNumMemoryBytes = numMemoryBytes;
}

void LoadKaldiArkArray(const char *fileName, uint32_t arrayIndex, std::string &ptrName, std::vector<uint8_t> &memory,
                       uint32_t *ptrNumRows, uint32_t *ptrNumColumns, uint32_t *ptrNumBytesPerElement) {
    std::ifstream in_file(fileName, std::ios::binary);
    if (in_file.good()) {
        uint32_t i = 0;
        while (i < arrayIndex) {
            std::string line;
            uint32_t numRows = 0u, numCols = 0u;
            std::getline(in_file, line, '\0');  // read variable length name followed by space and NUL
            std::getline(in_file, line, '\4');  // read "BFM" followed by space and control-D
            if (line.compare("BFM ") != 0) {
                break;
            }
            in_file.read(reinterpret_cast<char *>(&numRows), sizeof(uint32_t));     // read number of rows
            std::getline(in_file, line, '\4');                                     // read control-D
            in_file.read(reinterpret_cast<char *>(&numCols), sizeof(uint32_t));     // read number of columns
            in_file.seekg(numRows * numCols * sizeof(float), in_file.cur);         // read data
            i++;
        }
        if (!in_file.eof()) {
            std::string line;
            std::getline(in_file, ptrName, '\0');     // read variable length name followed by space and NUL
            std::getline(in_file, line, '\4');       // read "BFM" followed by space and control-D
            if (line.compare("BFM ") != 0) {
                fprintf(stderr, "Cannot find array specifier in file %s in LoadKaldiArkArray()!\n", fileName);
                exit(-1);
            }
            in_file.read(reinterpret_cast<char *>(ptrNumRows), sizeof(uint32_t));        // read number of rows
            std::getline(in_file, line, '\4');                                            // read control-D
            in_file.read(reinterpret_cast<char *>(ptrNumColumns), sizeof(uint32_t));    // read number of columns
            size_t willWrite = *ptrNumRows * *ptrNumColumns * sizeof(float);
            in_file.read(reinterpret_cast<char *>(&memory.front()),
                         *ptrNumRows * *ptrNumColumns * sizeof(float));  // read array data
        }
        in_file.close();
    } else {
        fprintf(stderr, "Failed to open %s for reading in GetKaldiArkInfo()!\n", fileName);
        exit(-1);
    }

    *ptrNumBytesPerElement = sizeof(float);
}

void SaveKaldiArkArray(const char *fileName,
                       bool shouldAppend,
                       std::string name,
                       void *ptrMemory,
                       uint32_t numRows,
                       uint32_t numColumns) {
    std::ios_base::openmode mode = std::ios::binary;
    if (shouldAppend) {
        mode |= std::ios::app;
    }
    std::ofstream out_file(fileName, mode);
    if (out_file.good()) {
        out_file.write(name.c_str(), name.length());  // write name
        out_file.write("\0", 1);
        out_file.write("BFM ", 4);
        out_file.write("\4", 1);
        out_file.write(reinterpret_cast<char *>(&numRows), sizeof(uint32_t));
        out_file.write("\4", 1);
        out_file.write(reinterpret_cast<char *>(&numColumns), sizeof(uint32_t));
        out_file.write(reinterpret_cast<char *>(ptrMemory), numRows * numColumns * sizeof(float));
        out_file.close();
    } else {
        throw std::runtime_error(std::string("Failed to open %s for writing in SaveKaldiArkArray()!\n") + fileName);
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

void ClearScoreError(score_error_t *error) {
    error->numScores = 0;
    error->numErrors = 0;
    error->maxError = 0.0;
    error->rmsError = 0.0;
    error->sumError = 0.0;
    error->sumRmsError = 0.0;
    error->sumSquaredError = 0.0;
    error->maxRelError = 0.0;
    error->sumRelError = 0.0;
    error->sumSquaredRelError = 0.0;
}

void UpdateScoreError(score_error_t *error, score_error_t *totalError) {
    totalError->numErrors += error->numErrors;
    totalError->numScores += error->numScores;
    totalError->sumRmsError += error->rmsError;
    totalError->sumError += error->sumError;
    totalError->sumSquaredError += error->sumSquaredError;
    if (error->maxError > totalError->maxError) {
        totalError->maxError = error->maxError;
    }
    totalError->sumRelError += error->sumRelError;
    totalError->sumSquaredRelError += error->sumSquaredRelError;
    if (error->maxRelError > totalError->maxRelError) {
        totalError->maxRelError = error->maxRelError;
    }
}

uint32_t CompareScores(float *ptrScoreArray,
                       void *ptrRefScoreArray,
                       score_error_t *scoreError,
                       uint32_t numRows,
                       uint32_t numColumns) {
    uint32_t numErrors = 0;

    ClearScoreError(scoreError);

    float *A = ptrScoreArray;
    float *B = reinterpret_cast<float *>(ptrRefScoreArray);
    for (uint32_t i = 0; i < numRows; i++) {
        for (uint32_t j = 0; j < numColumns; j++) {
            float score = A[i * numColumns + j];
            float refscore = B[i * numColumns + j];
            float error = fabs(refscore - score);
            float rel_error = error / (static_cast<float>(fabs(refscore)) + 1e-20f);
            float squared_error = error * error;
            float squared_rel_error = rel_error * rel_error;
            scoreError->numScores++;
            scoreError->sumError += error;
            scoreError->sumSquaredError += squared_error;
            if (error > scoreError->maxError) {
                scoreError->maxError = error;
            }
            scoreError->sumRelError += rel_error;
            scoreError->sumSquaredRelError += squared_rel_error;
            if (rel_error > scoreError->maxRelError) {
                scoreError->maxRelError = rel_error;
            }
            if (error > scoreError->threshold) {
                numErrors++;
            }
        }
    }
    scoreError->rmsError = sqrt(scoreError->sumSquaredError / (numRows * numColumns));
    scoreError->sumRmsError += scoreError->rmsError;
    scoreError->numErrors = numErrors;

    return (numErrors);
}

float StdDevError(score_error_t error) {
    return (sqrt(error.sumSquaredError / error.numScores
                 - (error.sumError / error.numScores) * (error.sumError / error.numScores)));
}

float StdDevRelError(score_error_t error) {
    return (sqrt(error.sumSquaredRelError / error.numScores
                 - (error.sumRelError / error.numScores) * (error.sumRelError / error.numScores)));
}

#if !defined(__arm__) && !defined(_M_ARM) && !defined(__aarch64__) && !defined(_M_ARM64)
#if defined(_WIN32) || defined(WIN32)
#include <intrin.h>
#include <windows.h>
#else

#include <cpuid.h>

#endif

inline void native_cpuid(unsigned int *eax, unsigned int *ebx,
                         unsigned int *ecx, unsigned int *edx) {
    size_t level = *eax;
#if defined(_WIN32) || defined(WIN32)
    int regs[4] = {static_cast<int>(*eax), static_cast<int>(*ebx), static_cast<int>(*ecx), static_cast<int>(*edx)};
    __cpuid(regs, level);
    *eax = static_cast<uint32_t>(regs[0]);
    *ebx = static_cast<uint32_t>(regs[1]);
    *ecx = static_cast<uint32_t>(regs[2]);
    *edx = static_cast<uint32_t>(regs[3]);
#else
    __get_cpuid(level, eax, ebx, ecx, edx);
#endif
}

// return GNA module frequency in MHz
float getGnaFrequencyMHz() {
    uint32_t level = 0;
    uint32_t eax = 1;
    uint32_t ebx = 0;
    uint32_t ecx = 0;
    uint32_t edx = 0;
    uint32_t family = 0;
    uint32_t model = 0;
    const uint8_t sixth_family = 6;
    const uint8_t cannon_lake_model = 102;
    const uint8_t gemini_lake_model = 122;

    native_cpuid(&eax, &ebx, &ecx, &edx);
    family = (eax >> 8) & 0xF;

    // model is the concatenation of two fields
    // | extended model | model |
    // copy extended model data
    model = (eax >> 16) & 0xF;
    // shift
    model <<= 4;
    // copy model data
    model += (eax >> 4) & 0xF;

    if (family == sixth_family && model == cannon_lake_model) {
        return 400;
    } else if (family == sixth_family &&
               model == gemini_lake_model) {
        return 200;
    } else {
        // counters not supported and we retrns just default value
        return 1;
    }
}

#endif  // !defined(__arm__) && !defined(_M_ARM)

void printReferenceCompareResults(score_error_t const &totalError,
                                  size_t framesNum,
                                  std::ostream &stream) {
    stream << "         max error: " <<
           totalError.maxError << std::endl;
    stream << "         avg error: " <<
           totalError.sumError / totalError.numScores << std::endl;
    stream << "     avg rms error: " <<
           totalError.sumRmsError / framesNum << std::endl;
    stream << "       stdev error: " <<
           StdDevError(totalError) << std::endl << std::endl;
    stream << std::endl;
}

void printPerformanceCounters(std::map<std::string,
        InferenceEngine::InferenceEngineProfileInfo> const &utterancePerfMap,
                              size_t callsNum,
                              std::ostream &stream) {
#if !defined(__arm__) && !defined(_M_ARM) && !defined(__aarch64__) && !defined(_M_ARM64)
    stream << std::endl << "Performance counts:" << std::endl;
    stream << std::setw(10) << std::right << "" << "Counter descriptions";
    stream << std::setw(22) << "Utt scoring time";
    stream << std::setw(18) << "Avg infer time";
    stream << std::endl;

    stream << std::setw(46) << "(ms)";
    stream << std::setw(24) << "(us per call)";
    stream << std::endl;

    for (const auto &it : utterancePerfMap) {
        std::string const &counter_name = it.first;
        float current_units = it.second.realTime_uSec;
        float call_units = current_units / callsNum;
        float freq = 1.0;
        // if GNA HW counters
        // get frequency of GNA module
        freq = getGnaFrequencyMHz();
        current_units /= freq * 1000;
        call_units /= freq;
        stream << std::setw(30) << std::left << counter_name.substr(4, counter_name.size() - 1);
        stream << std::setw(16) << std::right << current_units;
        stream << std::setw(21) << std::right << call_units;
        stream << std::endl;
    }
    stream << std::endl;
#endif
}

void getPerformanceCounters(InferenceEngine::InferRequest &request,
                            std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfCounters) {
    auto retPerfCounters = request.GetPerformanceCounts();

    for (const auto &pair : retPerfCounters) {
        perfCounters[pair.first] = pair.second;
    }
}

void sumPerformanceCounters(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> const &perfCounters,
                            std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &totalPerfCounters) {
    for (const auto &pair : perfCounters) {
        totalPerfCounters[pair.first].realTime_uSec += pair.second.realTime_uSec;
    }
}

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    slog::info << "Parsing input parameters" << slog::endl;

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
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

    if ((FLAGS_d.compare("GPU") != 0) && (FLAGS_d.compare("CPU") != 0) && (FLAGS_d.compare("GNA_AUTO") != 0) &&
        (FLAGS_d.compare("GNA_HW") != 0)
        && (FLAGS_d.compare("GNA_SW") != 0) && (FLAGS_d.compare("GNA_SW_EXACT") != 0)) {
        throw std::logic_error("Specified device is not supported.");
    }

    float scaleFactorInput = static_cast<float>(FLAGS_sf);
    if (scaleFactorInput <= 0.0f) {
        throw std::logic_error("Scale factor out of range (must be non-negative).");
    }

    uint32_t batchSize = (uint32_t) FLAGS_bs;
    if ((batchSize < 1) || (batchSize > 8)) {
        throw std::logic_error("Batch size out of range (1..8).");
    }

    /** default is a static quantisation **/
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
        throw std::logic_error("Not valid value for 'nthreads' argument. It should be > 0 ");
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

        if (FLAGS_l.empty()) {
            slog::info << "No extensions provided" << slog::endl;
        }

        bool useGna = (FLAGS_d.find("GNA") != std::string::npos);
        auto deviceStr = FLAGS_d.substr(0, (FLAGS_d.find("_")));
        float scaleFactorInput = static_cast<float>(FLAGS_sf);
        uint32_t batchSize = (uint32_t) FLAGS_bs;
        /** Extract input ark file name **/
        std::string inputArkName = fileNameNoExt(FLAGS_i) + ".ark";

        uint32_t numUtterances(0), numBytesThisUtterance(0);
        if (!FLAGS_i.empty()) {
            GetKaldiArkInfo(inputArkName.c_str(), 0, &numUtterances, &numBytesThisUtterance);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load Plugin for inference engine -------------------------------------
        slog::info << "Loading plugin" << slog::endl;
        /** Loading plugin for device **/
        InferencePlugin plugin = PluginDispatcher({FLAGS_pp, "../../../lib/intel64", ""}).getPluginByDevice(deviceStr);

        /** Printing plugin version **/
        std::cout << plugin.GetVersion() << std::endl << std::endl;
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
        slog::info << "Loading network files" << slog::endl;

        CNNNetReader netBuilder;
        if (!FLAGS_m.empty()) {
            /** Read network model **/
            netBuilder.ReadNetwork(FLAGS_m);

            /** Extract model name and load weights **/
            std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
            netBuilder.ReadWeights(binFileName);

            // -------------------------------------------------------------------------------------------------

            // --------------------------- 3. Set batch size ---------------------------------------------------
            /** Set batch size.  Unlike in imaging, batching in time (rather than space) is done for speech recognition. **/
            netBuilder.getNetwork().setBatchSize(batchSize);
            slog::info << "Batch size is " << std::to_string(netBuilder.getNetwork().getBatchSize())
                       << slog::endl;
        }

        /** Setting plugin parameter for per layer metrics **/
        std::map<std::string, std::string> gnaPluginConfig;
        std::map<std::string, std::string> genericPluginConfig;
        if (FLAGS_d.compare("CPU") != 0) {
            gnaPluginConfig[GNAConfigParams::KEY_GNA_DEVICE_MODE] = FLAGS_d;
        }
        if (FLAGS_pc) {
            genericPluginConfig[PluginConfigParams::KEY_PERF_COUNT] = PluginConfigParams::YES;
        }

        if (FLAGS_q.compare("user") == 0) {
            std::cout << "[ INFO ] Using scale factor of " << FLAGS_sf << std::endl;
            gnaPluginConfig[GNA_CONFIG_KEY(SCALE_FACTOR)] = std::to_string(FLAGS_sf);
        } else {  // "static" quantization with calculated scale factor
            std::string name;
            std::vector<uint8_t> ptrFeatures;
            uint32_t numArrays(0), numBytes(0), numFrames(0), numFrameElements(0), numBytesPerElement(0);
            GetKaldiArkInfo(inputArkName.c_str(), 0, &numArrays, &numBytes);
            ptrFeatures.resize(numBytes);
            LoadKaldiArkArray(inputArkName.c_str(),
                              0,
                              name,
                              ptrFeatures,
                              &numFrames,
                              &numFrameElements,
                              &numBytesPerElement);
            scaleFactorInput =
                    ScaleFactorForQuantization(ptrFeatures.data(), MAX_VAL_2B_FEAT, numFrames * numFrameElements);
            slog::info << "Using scale factor of " << scaleFactorInput << " calculated from first utterance."
                       << slog::endl;
            gnaPluginConfig[GNA_CONFIG_KEY(SCALE_FACTOR)] = std::to_string(scaleFactorInput);
        }

        if (FLAGS_qb == 8) {
            gnaPluginConfig[GNAConfigParams::KEY_GNA_PRECISION] = "I8";
        } else {
            gnaPluginConfig[GNAConfigParams::KEY_GNA_PRECISION] = "I16";
        }

        gnaPluginConfig[GNAConfigParams::KEY_GNA_LIB_N_THREADS] = std::to_string(FLAGS_nthreads);
        gnaPluginConfig[GNA_CONFIG_KEY(COMPACT_MODE)] = CONFIG_VALUE(NO);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Write model to file --------------------------------------------------
        // Embedded GNA model dumping (for Intel(R) Speech Enabling Developer Kit)
        if (!FLAGS_we.empty()) {
            gnaPluginConfig[GNAConfigParams::KEY_GNA_FIRMWARE_MODEL_IMAGE] = FLAGS_we;
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Loading model to the plugin ------------------------------------------

        if (useGna) {
            genericPluginConfig.insert(std::begin(gnaPluginConfig), std::end(gnaPluginConfig));
        }
        auto t0 = Time::now();
        ExecutableNetwork executableNet;
        if (!FLAGS_m.empty()) {
            slog::info << "Loading model to the plugin" << slog::endl;
            executableNet = plugin.LoadNetwork(netBuilder.getNetwork(), genericPluginConfig);
        } else {
            slog::info << "Importing model to the plugin" << slog::endl;
            executableNet = plugin.ImportNetwork(FLAGS_rg.c_str(), genericPluginConfig);
        }


        ms loadTime = std::chrono::duration_cast<ms>(Time::now() - t0);
        slog::info << "Model loading time " << loadTime.count() << " ms" << slog::endl;

        // --------------------------- 6. Exporting gna model using InferenceEngine AOT API---------------------
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
            return 0;
        }

        std::vector<std::pair<InferRequest, size_t>> inferRequests(FLAGS_nthreads);
        for (auto& inferRequest : inferRequests) {
            inferRequest = {executableNet.CreateInferRequest(), -1};
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 7. Prepare input blobs --------------------------------------------------
        /** Taking information about all topology inputs **/
        ConstInputsDataMap cInputInfo = executableNet.GetInputsInfo();
        InputsDataMap inputInfo;
        if (!FLAGS_m.empty()) {
            inputInfo = netBuilder.getNetwork().getInputsInfo();
        }

        /** Stores all input blobs data **/
        if (cInputInfo.size() != 1) {
            throw std::logic_error("Sample supports only topologies with  1 input");
        }

        Blob::Ptr ptrInputBlob = inferRequests[0].first.GetBlob(cInputInfo.begin()->first);

        /** configure input precision if model loaded from IR **/
        for (auto &item : inputInfo) {
            Precision inputPrecision = Precision::FP32;  // specify Precision::I16 to provide quantized inputs
            item.second->setPrecision(inputPrecision);
            item.second->getInputData()->layout = NC;  // row major layout
        }

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 8. Prepare output blobs -------------------------------------------------
        ConstOutputsDataMap cOutputInfo(executableNet.GetOutputsInfo());
        OutputsDataMap outputInfo;
        if (!FLAGS_m.empty()) {
            outputInfo = netBuilder.getNetwork().getOutputsInfo();
        }

        Blob::Ptr ptrOutputBlob = inferRequests[0].first.GetBlob(cOutputInfo.begin()->first);

        for (auto &item : outputInfo) {
            DataPtr outData = item.second;
            if (!outData) {
                throw std::logic_error("output data pointer is not valid");
            }

            Precision outputPrecision = Precision::FP32;  // specify Precision::I32 to retrieve quantized outputs
            outData->setPrecision(outputPrecision);
            outData->layout = NC;  // row major layout
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 9. Do inference ---------------------------------------------------------
        std::vector<uint8_t> ptrUtterance;
        std::vector<uint8_t> ptrScores;
        std::vector<uint8_t> ptrReferenceScores;
        score_error_t frameError, totalError;

        for (uint32_t utteranceIndex = 0; utteranceIndex < numUtterances; ++utteranceIndex) {
            std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> utterancePerfMap;
            std::string uttName;
            uint32_t numFrames(0), numFrameElementsInput(0), numBytesPerElementInput(0), n(0);
            uint32_t numFramesReference(0), numFrameElementsReference(0), numBytesPerElementReference(0),
                    numBytesReferenceScoreThisUtterance(0);
            const uint32_t numScoresPerFrame = ptrOutputBlob->size() / batchSize;
            GetKaldiArkInfo(inputArkName.c_str(), utteranceIndex, &n, &numBytesThisUtterance);
            ptrUtterance.resize(numBytesThisUtterance);
            LoadKaldiArkArray(inputArkName.c_str(),
                              utteranceIndex,
                              uttName,
                              ptrUtterance,
                              &numFrames,
                              &numFrameElementsInput,
                              &numBytesPerElementInput);

            uint32_t numFrameElementsInputPadded = numFrameElementsInput;

            if (ptrInputBlob->size() != numFrameElementsInputPadded * batchSize) {
                throw std::logic_error("network input size(" + std::to_string(ptrInputBlob->size()) +
                                       ") mismatch to ark file size (" +
                                       std::to_string(numFrameElementsInputPadded * batchSize) + ")");
            }
            ptrScores.resize(numFrames * numScoresPerFrame * sizeof(float));
            if (!FLAGS_r.empty()) {
                std::string refUtteranceName;
                GetKaldiArkInfo(FLAGS_r.c_str(), utteranceIndex, &n, &numBytesReferenceScoreThisUtterance);
                ptrReferenceScores.resize(numBytesReferenceScoreThisUtterance);
                LoadKaldiArkArray(FLAGS_r.c_str(),
                                  utteranceIndex,
                                  refUtteranceName,
                                  ptrReferenceScores,
                                  &numFramesReference,
                                  &numFrameElementsReference,
                                  &numBytesPerElementReference);
            }

            double totalTime = 0.0;

            std::cout << "Utterance " << utteranceIndex << ": " << std::endl;

            ClearScoreError(&totalError);
            totalError.threshold = frameError.threshold = MAX_SCORE_DIFFERENCE;
            auto inputFrame = &ptrUtterance.front();
            auto outputFrame = &ptrScores.front();

            size_t frameIndex{0};
            uint32_t numFramesThisBatch{batchSize};

            auto t0 = Time::now();
            auto t1 = t0;

            // Doing inference
            while (frameIndex <= numFrames) {
                if (frameIndex == numFrames) {
                    bool hasRequests = false;
                    for (auto &inferRequest : inferRequests) {
                        if (inferRequest.second != -1) {
                            hasRequests = true;
                        }
                    }
                    if (!hasRequests) {
                        break;
                    }
                }

                bool inferRequestFetched = false;
                for (auto &inferRequest : inferRequests) {
                    if (frameIndex == numFrames) {
                        numFramesThisBatch = 1;
                    } else {
                        numFramesThisBatch = (numFrames - frameIndex < batchSize) ? (numFrames - frameIndex) : batchSize;
                    }

                    if (inferRequest.second != -1) {
                        StatusCode code = inferRequest.first.Wait(
                                InferenceEngine::IInferRequest::WaitMode::RESULT_READY);

                        if (code != StatusCode::OK) {
                            continue;
                        }

                        if (!FLAGS_o.empty()) {
                            Blob::Ptr outputBlob = inferRequest.first.GetBlob(cOutputInfo.begin()->first);
                            std::memcpy(outputFrame,
                                        outputBlob->buffer(),
                                        outputBlob->byteSize());
                            outputFrame += numScoresPerFrame * sizeof(float);
                        }

                        if (!FLAGS_r.empty()) {
                            Blob::Ptr outputBlob = inferRequest.first.GetBlob(cOutputInfo.begin()->first);
                            CompareScores(outputBlob->buffer().as<float *>(),
                                          &ptrReferenceScores[inferRequest.second *
                                                              numFrameElementsReference *
                                                              numBytesPerElementReference],
                                          &frameError,
                                          numFramesThisBatch,
                                          numFrameElementsReference);
                            UpdateScoreError(&frameError, &totalError);
                        }
                    }

                    inferRequest.second = -1;

                    if (frameIndex == numFrames) {
                        continue;
                    }

                    Blob::Ptr inputBlob = inferRequest.first.GetBlob(cInputInfo.begin()->first);
                    std::memcpy(inputBlob->buffer(),
                                inputFrame,
                                inputBlob->byteSize());

                    inferRequest.first.StartAsync();

                    inferRequest.second = frameIndex;
                    frameIndex += numFramesThisBatch;
                    inputFrame += sizeof(float) * numFrameElementsInput * numFramesThisBatch;
                    inferRequestFetched |= true;
                }

                if (!inferRequestFetched) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }

                if (FLAGS_pc) {
                    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> callPerfMap;
                    // retrive new counters
                    for (auto inferRequest : inferRequests) {
                        getPerformanceCounters(inferRequest.first, callPerfMap);
                        // summarize retrived counters with all previous
                        sumPerformanceCounters(callPerfMap, utterancePerfMap);
                    }
                }
            }
            t1 = Time::now();

            fsec fs = t1 - t0;
            ms d = std::chrono::duration_cast<ms>(fs);
            totalTime += d.count();

            // resetting state between utterances
            for (auto &&state : executableNet.QueryState()) {
                state.Reset();
            }

            if (!FLAGS_o.empty()) {
                bool shouldAppend = (utteranceIndex == 0) ? false : true;
                SaveKaldiArkArray(FLAGS_o.c_str(), shouldAppend, uttName, &ptrScores.front(),
                                  numFrames, numScoresPerFrame);
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
                printPerformanceCounters(utterancePerfMap, frameIndex, std::cout);
            }
            if (!FLAGS_r.empty()) {
                printReferenceCompareResults(totalError, numFrames, std::cout);
            }
            std::cout << "End of Utterance " << utteranceIndex << std::endl << std::endl;
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
