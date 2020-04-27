// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstring>
#include <thread>
#include <chrono>

#include <blob_factory.hpp>
#include <ie_memcpy.h>
#include <format_reader_ptr.h>

#include <myriad_layers_tests.hpp>
#include <myriad_layers_reference_functions.hpp>

using namespace InferenceEngine;

PRETTY_PARAM(kernel, param_size)
PRETTY_PARAM(stride, param_size)
PRETTY_PARAM(pad, param_size)
PRETTY_PARAM(out_channels, int)
PRETTY_PARAM(group, int)
PRETTY_PARAM(dilation_factor, param_size)
PRETTY_PARAM(tfPad, paddings4)

struct RunInfo {
    bool hwMode = true;
};

class MyriadX_HW_Tests_nightly : public myriadLayersTests_nightly {
public:
    void CheckHWRun() {
        StatusCode st;

        std::map<std::string, InferenceEngineProfileInfo> perfMap;
        ASSERT_NO_THROW(st = _inferRequest->GetPerformanceCounts(perfMap, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        std::vector<std::pair<std::string, InferenceEngineProfileInfo>> perfVec(perfMap.begin(), perfMap.end());
        std::sort(perfVec.begin(), perfVec.end(),
            [=](const std::pair<std::string, InferenceEngineProfileInfo> &pair1,
                const std::pair<std::string, InferenceEngineProfileInfo> &pair2) {
                return pair1.second.execution_index < pair2.second.execution_index;
            });

        size_t maxLayerName = 0u, maxExecType = 0u;
        for (auto it = perfVec.begin(); it != perfVec.end(); ++it) {
            maxLayerName = std::max(maxLayerName, it->first.length());
            maxExecType = std::max(maxExecType, std::strlen(it->second.exec_type));
        }

        size_t indexWidth = 7, nameWidth = maxLayerName + 5, typeWidth = maxExecType + 5, timeWidth = 10;
        size_t totalWidth = indexWidth + nameWidth + typeWidth + timeWidth;

        std::cout << std::endl;
        std::cout << "Detailed Per Stage Profile" << std::endl;

        for (size_t i = 0; i < totalWidth; i++) {
            std::cout << "=";
        }

        std::cout << std::endl;
        std::cout << std::setw(indexWidth) << std::left << "Index"
                  << std::setw(nameWidth) << std::left << "Name"
                  << std::setw(typeWidth) << std::left << "Type"
                  << std::setw(timeWidth) << std::right << "Time (ms)"
                  << std::endl;

        for (size_t i = 0; i < totalWidth; i++) {
            std::cout << "-";
        }
        std::cout << std::endl;

        bool hasHWStage = false;
        long long totalTime = 0;

        for (const auto& p : perfVec) {
            const auto& stageName = p.first;
            const auto& info = p.second;

            if (info.status == InferenceEngineProfileInfo::EXECUTED) {
                std::string stageType(info.exec_type);
                if (stageType.find("MyriadXHw") != std::string::npos) {
                    hasHWStage = true;
                }

                std::cout << std::setw(indexWidth) << std::left << info.execution_index
                          << std::setw(nameWidth) << std::left << stageName
                          << std::setw(typeWidth) << std::left << info.exec_type
                          << std::setw(timeWidth) << std::right << info.realTime_uSec / 1000.0
                          << std::endl;

                totalTime += info.realTime_uSec;
            }
        }

        for (int i = 0; i < totalWidth; i++) {
            std::cout << "-";
        }
        std::cout << std::endl;

        std::cout << std::setw(totalWidth / 2) << std::right << "Total inference time:"
                  << std::setw(totalWidth / 2 + 1) << std::right << totalTime / 1000.0
                  << std::endl;

        for (int i = 0; i < totalWidth; i++) {
            std::cout << "-";
        }
        std::cout << std::endl;

        EXPECT_TRUE(hasHWStage);
    }

    void RunNetwork(const CNNNetwork& network,
                    const Blob::Ptr& input,
                    Blob::Ptr& output,
                    const char* inputName,
                    const char* outputName,
                    const RunInfo& runInfo,
                    const std::string& logLevel = CONFIG_VALUE(LOG_NONE)) {
        _inferRequest.reset();
        _exeNetwork.reset();

        StatusCode st;

        std::map<std::string, std::string> config = {
            { VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), runInfo.hwMode ? CONFIG_VALUE(YES) : CONFIG_VALUE(NO) },

            { CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES) },
            { VPU_CONFIG_KEY(PERF_REPORT_MODE), VPU_CONFIG_VALUE(PER_STAGE) },

            { CONFIG_KEY(LOG_LEVEL), logLevel }
        };

        ASSERT_NO_THROW(st = _vpuPluginPtr->LoadNetwork(_exeNetwork, network, config, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(_inferRequest, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        ASSERT_NO_THROW(st = _inferRequest->SetBlob(inputName, input, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        ASSERT_NO_THROW(st = _inferRequest->GetBlob(outputName, output, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    }

    void CompareWithSW(float errorThreshold, vpu::LayoutPreference layoutPreference = vpu::LayoutPreference::ChannelMajor) {
        Blob::Ptr swOutput;
        {
            SCOPED_TRACE("SW");

            ResetGeneratedNet();
            ASSERT_TRUE(generateNetAndInfer(NetworkInitParams()
                                            .useHWOpt(false)
                                            .runRefGraph(false)
                                            .layoutPreference(layoutPreference)));

            auto outBlob = _outputMap.begin()->second;
            swOutput = make_shared_blob<ie_fp16>(outBlob->getTensorDesc());
            swOutput->allocate();
            std::copy_n(outBlob->cbuffer().as<const uint8_t*>(), outBlob->byteSize(), swOutput->buffer().as<uint8_t*>());
        }

        {
            SCOPED_TRACE("HW");            

            ResetGeneratedNet();
            ASSERT_TRUE(generateNetAndInfer(NetworkInitParams()
                                            .useHWOpt(true)
                                            .runRefGraph(false)
                                            .layoutPreference(layoutPreference)));
            ASSERT_NO_FATAL_FAILURE(CheckHWRun());

            auto outBlob = _outputMap.begin()->second;
            CompareCommonAbsolute(outBlob, swOutput, errorThreshold);
        }
    }

    void CompareWithItself(int numIters) {
        ResetGeneratedNet();
        ASSERT_TRUE(generateNetAndInfer(NetworkInitParams()
                                        .useHWOpt(true)
                                        .runRefGraph(false)));

        auto outBlob = _outputMap.begin()->second;

        auto firstOutput = make_shared_blob<ie_fp16>(outBlob->getTensorDesc());
        firstOutput->allocate();
        std::copy_n(outBlob->cbuffer().as<const ie_fp16*>(), outBlob->size(), firstOutput->buffer().as<ie_fp16*>());

        for (int i = 0; i < numIters; ++i) {
            ASSERT_TRUE(Infer());
            ASSERT_NO_FATAL_FAILURE(CompareCommonAbsolute(outBlob, firstOutput, 0.0f)) << i;
        }
    }
};
