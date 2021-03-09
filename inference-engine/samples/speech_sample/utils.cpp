// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <iostream>

#include <samples/common.hpp>
#include <samples/slog.hpp>

#include "utils.hpp"

void ArkFile::GetKaldiArkInfo(const char* fileName,
    uint32_t numArrayToFindSize,
    uint32_t* ptrNumArrays,
    uint32_t* ptrNumMemoryBytes) {
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
            in_file.read(reinterpret_cast<char*>(&numRows), sizeof(uint32_t));  // read number of rows
            std::getline(in_file, line, '\4');                                   // read control-D
            in_file.read(reinterpret_cast<char*>(&numCols), sizeof(uint32_t));  // read number of columns
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

void ArkFile::LoadKaldiArkArray(const char* fileName, uint32_t arrayIndex, std::string& ptrName, std::vector<uint8_t>& memory,
    uint32_t* ptrNumRows, uint32_t* ptrNumColumns, uint32_t* ptrNumBytesPerElement) {
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
            in_file.read(reinterpret_cast<char*>(&numRows), sizeof(uint32_t));     // read number of rows
            std::getline(in_file, line, '\4');                                     // read control-D
            in_file.read(reinterpret_cast<char*>(&numCols), sizeof(uint32_t));     // read number of columns
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
            in_file.read(reinterpret_cast<char*>(ptrNumRows), sizeof(uint32_t));        // read number of rows
            std::getline(in_file, line, '\4');                                            // read control-D
            in_file.read(reinterpret_cast<char*>(ptrNumColumns), sizeof(uint32_t));    // read number of columns
            in_file.read(reinterpret_cast<char*>(&memory.front()),
                *ptrNumRows * *ptrNumColumns * sizeof(float));  // read array data
        }
        in_file.close();
    } else {
        fprintf(stderr, "Failed to open %s for reading in GetKaldiArkInfo()!\n", fileName);
        exit(-1);
    }

    *ptrNumBytesPerElement = sizeof(float);
}

void ArkFile::SaveKaldiArkArray(const char* fileName,
    bool shouldAppend,
    std::string name,
    void* ptrMemory,
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
        out_file.write(reinterpret_cast<char*>(&numRows), sizeof(uint32_t));
        out_file.write("\4", 1);
        out_file.write(reinterpret_cast<char*>(&numColumns), sizeof(uint32_t));
        out_file.write(reinterpret_cast<char*>(ptrMemory), numRows * numColumns * sizeof(float));
        out_file.close();
    } else {
        throw std::runtime_error(std::string("Failed to open %s for writing in SaveKaldiArkArray()!\n") + fileName);
    }
}

void ArkFile::SetNumBytesForCurrentUtterance(std::istringstream& stream,
                                             std::vector<std::string>& inputArkFiles,
                                             std::vector<uint32_t>& numBytesThisUtterance,
                                             uint32_t& numUtterances) {
    uint32_t currentNumUtterances(0), currentNumBytesThisUtterance(0);
    std::string outStr;
    while (getline(stream, outStr, ',')) {
        std::string filename(fileNameNoExt(outStr) + ".ark");
        inputArkFiles.push_back(filename);

        ArkFile::GetKaldiArkInfo(filename.c_str(), 0, &currentNumUtterances, &currentNumBytesThisUtterance);
        if (numUtterances == 0) {
            numUtterances = currentNumUtterances;
        } else if (currentNumUtterances != numUtterances) {
            throw std::logic_error("Incorrect input files. Number of utterance must be the same for all ark files");
        }
        numBytesThisUtterance.push_back(currentNumBytesThisUtterance);
    }
}

void PerformanceCounters::GetPerformanceCounters(InferenceEngine::InferRequest& request,
                                    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& perfCounters) {
    auto retPerfCounters = request.GetPerformanceCounts();

    for (const auto& pair : retPerfCounters) {
        perfCounters[pair.first] = pair.second;
    }
}

void PerformanceCounters::SumPerformanceCounters(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> const& perfCounters,
                                                 std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& totalPerfCounters) {
    for (const auto& pair : perfCounters) {
        totalPerfCounters[pair.first].realTime_uSec += pair.second.realTime_uSec;
    }
}

void PerformanceCounters::PrintPerformanceCounters(std::map<std::string,
                                                   InferenceEngine::InferenceEngineProfileInfo> const& utterancePerfMap,
                                                   size_t callsNum,
                                                   std::ostream& stream,
                                                   std::string fullDeviceName) {
#if !defined(__arm__) && !defined(_M_ARM) && !defined(__aarch64__) && !defined(_M_ARM64)
    stream << std::endl << "Performance counts:" << std::endl;
    stream << std::setw(10) << std::right << "" << "Counter descriptions";
    stream << std::setw(22) << "Utt scoring time";
    stream << std::setw(18) << "Avg infer time";
    stream << std::endl;

    stream << std::setw(46) << "(ms)";
    stream << std::setw(24) << "(us per call)";
    stream << std::endl;

    for (const auto& it : utterancePerfMap) {
        std::string const& counter_name = it.first;
        float current_units = static_cast<float>(it.second.realTime_uSec);
        float call_units = current_units / callsNum;
        // if GNA HW counters
        // get frequency of GNA module
        float freq = GetGnaFrequencyMHz();
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


#if !defined(__arm__) && !defined(_M_ARM) && !defined(__aarch64__) && !defined(_M_ARM64)
#ifdef _WIN32
#include <intrin.h>
#include <windows.h>
#else

#include <cpuid.h>

#endif

inline void native_cpuid(unsigned int* eax, unsigned int* ebx, unsigned int* ecx, unsigned int* edx) {
    size_t level = *eax;
#ifdef _WIN32
    int regs[4] = { static_cast<int>(*eax), static_cast<int>(*ebx), static_cast<int>(*ecx), static_cast<int>(*edx) };
    __cpuid(regs, level);
    *eax = static_cast<uint32_t>(regs[0]);
    *ebx = static_cast<uint32_t>(regs[1]);
    *ecx = static_cast<uint32_t>(regs[2]);
    *edx = static_cast<uint32_t>(regs[3]);
#else
    __get_cpuid(level, eax, ebx, ecx, edx);
#endif
}

float PerformanceCounters::GetGnaFrequencyMHz() {
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
#endif // if not ARM

void Score::ClearScoreError(score_error_t* error) {
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

void Score::UpdateScoreError(score_error_t* error, score_error_t* totalError) {
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

uint32_t Score::CompareScores(float* ptrScoreArray,
                              void* ptrRefScoreArray,
                              score_error_t* scoreError,
                              uint32_t numRows,
                              uint32_t numColumns) {
    uint32_t numErrors = 0;

    ClearScoreError(scoreError);

    float* A = ptrScoreArray;
    float* B = reinterpret_cast<float*>(ptrRefScoreArray);
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

float Printresults::StdDevError(score_error_t error) {
    return (sqrt(error.sumSquaredError / error.numScores
        - (error.sumError / error.numScores) * (error.sumError / error.numScores)));
}

void Printresults::PrintReferenceCompareResults(score_error_t const& totalError,
                                           size_t framesNum,
                                           std::ostream& stream) {
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
