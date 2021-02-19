// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <map>

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


class ArkFile{

public:
    static void GetKaldiArkInfo(const char* fileName,
                                uint32_t numArrayToFindSize,
                                uint32_t * ptrNumArrays,
                                uint32_t * ptrNumMemoryBytes);

    static void LoadKaldiArkArray(const char* fileName,
                                  uint32_t arrayIndex,
                                  std::string& ptrName,
                                  std::vector<uint8_t>& memory,
                                  uint32_t* ptrNumRows,
                                  uint32_t* ptrNumColumns,
                                  uint32_t* ptrNumBytesPerElement);

    static void SaveKaldiArkArray(const char* fileName,
                                  bool shouldAppend,
                                  std::string name,
                                  void* ptrMemory,
                                  uint32_t numRows,
                                  uint32_t numColumns);

    static void SetNumBytesForCurrentUtterance(std::istringstream& stream,
                                               std::vector<std::string>& inputArkFiles,
                                               std::vector<uint32_t>& numBytesThisUtterance,
                                               uint32_t& numUtterances);

};

class PerformanceCounters {

public:
    static void GetPerformanceCounters(InferenceEngine::InferRequest& request,
                                       std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& perfCounters);

    static void SumPerformanceCounters(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> const& perfCounters,
                                       std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& totalPerfCounters);

    static void PrintPerformanceCounters(std::map<std::string,
                                         InferenceEngine::InferenceEngineProfileInfo> const& utterancePerfMap,
                                         size_t callsNum,
                                         std::ostream& stream,
                                         std::string fullDeviceName);

    // return GNA module frequency in MHz
    static float GetGnaFrequencyMHz();

};

class Score {

public:
    static void ClearScoreError(score_error_t* error);

    static void UpdateScoreError(score_error_t* error, score_error_t* totalError);

    static uint32_t CompareScores(float* ptrScoreArray,
                                  void* ptrRefScoreArray,
                                  score_error_t* scoreError,
                                  uint32_t numRows,
                                  uint32_t numColumns);
};
