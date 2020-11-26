// Copyright (C) 2018-2020 Intel Corporation
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

struct InferRequestStruct {
    InferRequest inferRequest;
    int frameIndex;
    uint32_t numFramesThisBatch;
};

void CheckNumberOfInputs(size_t numInputs, size_t numInputArkFiles) {
    if (numInputs != numInputArkFiles) {
        throw std::logic_error("Number of network inputs (" + std::to_string(numInputs) + ")"
                               " is not equal to number of ark files (" + std::to_string(numInputArkFiles) + ")");
    }
}

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
    uint32_t eax = 1;
    uint32_t ebx = 0;
    uint32_t ecx = 0;
    uint32_t edx = 0;
    uint32_t family = 0;
    uint32_t model = 0;
    const uint8_t sixth_family = 6;
    const uint8_t cannon_lake_model = 102;
    const uint8_t gemini_lake_model = 122;
    const uint8_t ice_lake_model = 126;
    const uint8_t next_model = 140;

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

    if (family == sixth_family) {
        switch (model) {
            case cannon_lake_model:
            case ice_lake_model:
            case next_model:
                return 400;
            case gemini_lake_model:
                return 200;
            default:
                return 1;
        }
    } else {
        // counters not supported and we returns just default value
        return 1;
    }
}

#endif  // if not ARM

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
                              std::ostream &stream, std::string fullDeviceName) {
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
        float current_units = static_cast<float>(it.second.realTime_uSec);
        float call_units = current_units / callsNum;
        // if GNA HW counters
        // get frequency of GNA module
        float freq = getGnaFrequencyMHz();
        current_units /= freq * 1000;
        call_units /= freq;
        stream << std::setw(30) << std::left << counter_name.substr(4, counter_name.size() - 1);
        stream << std::setw(16) << std::right << current_units;
        stream << std::setw(21) << std::right << call_units;
        stream << std::endl;
    }
    stream << std::endl;
    std::cout << std::endl;
    std::cout << "Full device name: " << fullDeviceName << std::endl;
    std::cout << std::endl;
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
            blobName.push_back(str.substr(pos_last, pos_next));
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

        auto isFeature = [&](const std::string xFeature) { return FLAGS_d.find(xFeature) != std::string::npos; };

        bool useGna = isFeature("GNA");
        bool useHetero = isFeature("HETERO");
        std::string deviceStr =
            useHetero && useGna ? "HETERO:GNA,CPU" : FLAGS_d.substr(0, (FLAGS_d.find("_")));
        uint32_t batchSize = (FLAGS_cw_r > 0 || FLAGS_cw_l > 0) ? 1 : (uint32_t)FLAGS_bs;

        std::vector<std::string> inputArkFiles;
        std::vector<uint32_t> numBytesThisUtterance;
        uint32_t numUtterances(0);
        // --------------------------- 1. Load inference engine -------------------------------------
        slog::info << "Loading Inference Engine" << slog::endl;
        Core ie;

        // 2. Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
        slog::info << "Loading network files" << slog::endl;

        CNNNetwork network;
        network = ie.ReadNetwork(FLAGS_m);
        network.setBatchSize(1);

        std::vector<std::string> outputs;
        ExecutableNetwork executableNet = ie.LoadNetwork(network, deviceStr);
        InferRequest inferRequest = executableNet.CreateInferRequest();

        ConstInputsDataMap cInputInfo = executableNet.GetInputsInfo();

        /** Stores all input blobs data **/
        std::vector<Blob::Ptr> ptrInputBlobs;
        for (const auto& input : cInputInfo) {
            ptrInputBlobs.push_back(inferRequest.GetBlob(input.first));
        }
        InputsDataMap inputInfo;
        inputInfo = network.getInputsInfo();
        /** Configure input precision if model is loaded from IR **/
        for (auto &item : inputInfo) {
            Precision inputPrecision = Precision::FP32;  // specify Precision::I16 to provide quantized inputs
            item.second->setPrecision(inputPrecision);
        }

        std::vector<Blob::Ptr> ptrOutputBlobs;
        ConstOutputsDataMap cOutputInfo = executableNet.GetOutputsInfo();
        for (const auto& output : cOutputInfo) {
            ptrOutputBlobs.push_back(inferRequest.GetBlob(output.first));
        }

        //! [part1]

        // initialize memory state before starting
        for (auto &&state : executableNet.QueryState()) {
            state.Reset();
        }

        std::vector<float> data = { 1,2,3,4,5,6 };
        for (size_t next_input = 0; next_input < data.size()/2; next_input++) {
            MemoryBlob::Ptr minput = as<MemoryBlob>(ptrInputBlobs[0]);
            // locked memory holder should be alive all time while access to its buffer happens
            auto minputHolder = minput->wmap();

            std::memcpy(minputHolder.as<void *>(),
                &data[next_input],
                sizeof(float));
            inferRequest.StartAsync();
            StatusCode code = inferRequest.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);

            while(code != StatusCode::OK){
                code = inferRequest.Wait(
                                    InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
            }
            IE_SUPPRESS_DEPRECATED_START
            //auto states = executableNet.QueryState();
            IE_SUPPRESS_DEPRECATED_END
           // auto mstate = as<MemoryBlob>(states[0].GetState());
            //auto state_buf = mstate->rmap();
            //std::cout << *(state_buf.as<float*>())<<"\n";
            MemoryBlob::Ptr moutput = as<MemoryBlob>(ptrOutputBlobs[0]);
            std::cout<<moutput->getTensorDesc().getDims()[1]<<"\n";
            // locked memory holder should be alive all time while access to its buffer happens
            auto moutputHolder = moutput->rmap();
            float *output = moutputHolder.as<float*>();
            std::cout << output[0] << " "<<output[1] << " "<<output[2] << " "<<output[3] << "\n";

        }

        // resetting state between utterances
        for (auto &&state : executableNet.QueryState()) {
            state.Reset();
        }

        for (size_t next_input = data.size()/2 +1; next_input < data.size(); next_input++) {
            std::vector<std::vector<uint8_t>> ptrUtterances;
            std::vector<uint8_t> ptrScores;

            std::cout << "Utterance " << next_input << ": " << std::endl;

            MemoryBlob::Ptr minput = as<MemoryBlob>(ptrInputBlobs[0]);
            // locked memory holder should be alive all time while access to its buffer happens
            auto minputHolder = minput->wmap();

            std::memcpy(minputHolder.as<void *>(),
                &data[next_input],
                sizeof(int));
            inferRequest.Infer();
            auto states = executableNet.QueryState();
            //auto mstate = as<MemoryBlob>(states[0].GetState());
            //auto state_buf = mstate->rmap();
            //std::cout << *(state_buf.as<float*>()) << "\n";
        }
        //! [part1]

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
