// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <gtest/gtest.h>
#include <ie_plugin_config.hpp>
#include "custom_matcher.hpp"

using namespace InferenceEngine;

InferenceEngine::ExecutableNetwork Regression::Matchers::CustomMatcher::createExecutableNetworkFromAOT() {
    ExecutableNetwork executableApi;
    try {
        ctx.setFileNames(config._paths_to_images);
        ctx.setModelPrecision(config.modelPrecision);

        executableApi = config.ie_core->ImportNetwork(config._path_to_aot_model, config._device_name, config.plugin_config);
    }
    catch (std::exception &e) {
        GTEST_MESSAGE_(e.what(), ::testing::TestPartResult::kFatalFailure);
    }

    return executableApi;

}

InferenceEngine::ExecutableNetwork Regression::Matchers::CustomMatcher::createExecutableNetworkFromIR(){
    ExecutableNetwork executableApi;
    try {
        ctx.setFileNames(config._paths_to_images);
        ctx.setModelPrecision(config.modelPrecision);

        if (config.make_model) {
            ctx.setModelPath(config._path_to_models);
            config.make_model(ctx);
        }

        std::string binFileName = testing::FileUtils::fileNameNoExt(config._path_to_models) + ".bin";
        network = config.ie_core->ReadNetwork(config._path_to_models, binFileName);

        // Change batch size if it is not equal 1
        auto inputs = network.getInputsInfo();

        if (config._inputPrecision) {
            for (auto && input : inputs) {
                input.second->setPrecision(config._inputPrecision);
                // NC is a proper layout for 2d blob if different is not specified, like CN
                auto layout = input.second->getTensorDesc().getDims().size() == 4 ? NCHW : NC;
                input.second->getInputData()->setLayout(layout);
            }
        }

        //TODO: why this need
        if (inputs.begin()->second->getTensorDesc().getDims().at(0) != 1) {
            std::cerr << "[WARNING]: Batch size will be equal to 1." << std::endl;
            network.setBatchSize(1);
        }

        if (config.batchSize != 1) {
            network.setBatchSize(config.batchSize);
        }

        if (!config.outputLayer.empty()) {
            network.addOutput(config.outputLayer);
        }

        if (config.useDynamicBatching) {
            config.plugin_config[PluginConfigParams::KEY_DYN_BATCH_ENABLED] = PluginConfigParams::YES;
        }

        auto outInfo = network.getOutputsInfo();

        auto loadedExecutableNetwork = config.ie_core->LoadNetwork(network, config._device_name, config.plugin_config);
        if (config.useExportImport) {
            std::stringstream stream;
            loadedExecutableNetwork.Export(stream);
            executableApi = config.ie_core->ImportNetwork(stream);
        } else {
            executableApi = loadedExecutableNetwork;
        }

    }
    catch (std::exception &e) {
        GTEST_MESSAGE_(e.what(), ::testing::TestPartResult::kFatalFailure);
    }

    return executableApi;
}

void Regression::Matchers::CustomMatcher::matchCustom() {
    try {
        ExecutableNetwork executableApi;
        std::vector<InferRequest> inferRequests;
        ConstInputsDataMap constInputs;
        ConstOutputsDataMap constOutInfo;
        ResponseDesc dsc;
        StatusCode sts = OK;

        if (!config._path_to_aot_model.empty()) {
            ASSERT_NO_FATAL_FAILURE(executableApi = createExecutableNetworkFromAOT());
        } else {
            ASSERT_NO_FATAL_FAILURE(executableApi = createExecutableNetworkFromIR());
        }

        if (executableApi.operator IExecutableNetwork::Ptr &() != nullptr) {
            for (int i=0; i != config._nrequests; i++ ) {
                inferRequests.push_back(executableApi.CreateInferRequest());
            }
        }

        if (config.useDynamicBatching) {
            for (auto && req : inferRequests) {
                req.SetBatch(config.dynBatch);
            }
        }

        auto make_unified_endpoints = [&] () {
            if (executableApi.operator IExecutableNetwork::Ptr &() != nullptr) {
                return std::make_pair(executableApi.GetInputsInfo(), executableApi.GetOutputsInfo());
            }
            auto inputs2 = network.getInputsInfo();
            ConstInputsDataMap constInputs2;
            for (const auto & input : inputs2) {
                constInputs2[input.first] = input.second;
            }
            auto output2 = network.getOutputsInfo();
            ConstOutputsDataMap constOutInfo2;
            for (const auto & output : output2) {
                constOutInfo2[output.first] = output.second;
            }
            return std::make_pair(constInputs2, constOutInfo2);
        };

        auto endpoints = make_unified_endpoints();

        for (auto && fetch_input : config.fetch_input) {
            // each fetcher can be used multiple times
            for (;;) {
                // load new input - reset if necessary
                decltype(fetch_input(ctx)) fetchResult;

                int requestProcessed = 0;
                for (int i = 0; i != config._nrequests; i++) {
                    int inputId = 0;
                    for (auto input : endpoints.first) {
                        InferenceEngine::Blob::Ptr inputBlb;
                        inputBlb = inferRequests[i].GetBlob(input.first);
                        ctx.setInput(input.second->name(), inputBlb);
                        ctx.setInputIdx(inputId);
                        decltype(fetch_input(ctx)) fetchResultForInput;
                        ASSERT_NO_FATAL_FAILURE(fetchResultForInput = fetch_input(ctx));
                        if (inputId != 0) {
                            ASSERT_EQ(fetchResult.fetched, fetchResultForInput.fetched);
                            ASSERT_EQ(fetchResult.fetchMore, fetchResultForInput.fetchMore);
                            ASSERT_EQ(fetchResult.frameNumber, fetchResultForInput.frameNumber);
                            ASSERT_EQ(fetchResult.reset, fetchResultForInput.reset);
                        } else {
                            fetchResult = fetchResultForInput;
                        }
                        inputId++;
                    }

                    if (fetchResult.fetched) {
                        // number of requests to infer in parallel
                        requestProcessed++;
                        // increasing frame number this however might be done in input fetcher if CTX passed by non const reference
                        // value used in read_next_.. fetchers family
                        ctx.setFrameNumber(ctx.getFrameNumber() + 1);
                    }
                    // cannot spawn more requests due to reset
                    if (fetchResult.reset) {
                        break;
                    }
                    // end of stream
                    if (!fetchResult.fetchMore) {
                        break;
                    }
                }

                if (fetchResult.fetched) {
                    // Infer model
                    if (requestProcessed == 1) {
                        inferRequests.front().Infer();
                        sts = OK;
                    } else {
                        for (int i = 0; i != requestProcessed; i++) {
                            inferRequests[i].StartAsync();
                        }
                        for (int i = 0; i != requestProcessed; i++) {
                            inferRequests[i].Wait(IInferRequest::RESULT_READY);
                        }
                        sts = OK;
                    }

                    if (!fetchResult.hasResult) {
                        continue;
                    }

                        // for infer request case will copy resulted blob
                    for (int i = 0; i != requestProcessed;i++) {
                        auto &outputs = ctx.newOutputs();
                        for (auto output : endpoints.second) {
                            auto tblob = dynamic_pointer_cast<TBlob<float>>(inferRequests[i].GetBlob(output.second->getName()));
                            outputs[output.second->getName()] = make_shared_blob(*tblob);
                        }
                    }
                }

                if (fetchResult.reset) {
                    auto states = executableApi.QueryState();
                    ASSERT_FALSE(states.empty());
                    for (auto& state : states) {
                        state.Reset();
                    }
                    // also store reset indicator for comparison routine
                    auto &outputs = ctx.newOutputs();
                    outputs["reset"] = nullptr;
                    //continue;
                }

                //FAIL()<<"stop after one frame";

                // Check errors
                if (sts == GENERAL_ERROR) {
                    THROW_IE_EXCEPTION << "Scoring failed! Critical error: " << dsc.msg;
                } else if (sts == NOT_IMPLEMENTED) {
                    THROW_IE_EXCEPTION << "Scoring failed! Input data is incorrect and not supported!";
                } else if (sts == NETWORK_NOT_LOADED) {
                    THROW_IE_EXCEPTION << "Scoring failed! " << dsc.msg;
                }
                if (!fetchResult.fetchMore) break;
            }
        }
    }
    catch (std::exception &e) {
        FAIL() << e.what();
    }
}

void Regression::Matchers::CustomMatcher::checkResult() {
    bool cmpNear = !isApproximatelyEqual(config.nearValue, 0.0);
    bool cmpNearAvg = !isApproximatelyEqual(config.nearAvgValue, 0.0);
    bool isSaveOutput = !!config.outputBlob;

    /**
     * In case where external comparison is used
     */
    if (isSaveOutput) {
        if (!config.fetch_result) {

            decltype(ctx.allOutputs().begin()) output;

            // calculating all outputs size
            SizeVector dimsMerged;
            for(auto && output :  ctx.allOutputs()) {
                auto outputBlobIt = config.outputLayer.empty() ? output.begin() : output.find(config.outputLayer);
                auto outBlob  = outputBlobIt->second;

                if (dimsMerged.empty()) {
                    dimsMerged = outBlob->getTensorDesc().getDims();
                } else {
                    ASSERT_EQ(dimsMerged.size(), outBlob->getTensorDesc().getDims().size());
                    int added = 0;
                    std::transform(begin(dimsMerged),
                                   end(dimsMerged),
                                   begin(dimsMerged = outBlob->getTensorDesc().getDims()),
                                   begin(dimsMerged),
                                   [&added](size_t l, size_t r) {
                                       added += l != r;
                                       return added ? l + r : l;
                                   });
                    ASSERT_LE(added,1);

                    if (added == 0 && !dimsMerged.empty()) {
                        dimsMerged.back() += outBlob->getTensorDesc().getDims().back();
                    }
                }
            }

            config.outputBlob->deallocate();
            config.outputBlob->getTensorDesc() = TensorDesc(config.outputBlob->getTensorDesc().getPrecision(),
                                                            dimsMerged,
                                                            TensorDesc::getLayoutByDims(dimsMerged));
            config.outputBlob->allocate();
            float *buff = config.outputBlob->buffer();

            // copying all output frames into allocated blob
            for(auto && output :  ctx.allOutputs()) {

                auto outputBlobIt = config.outputLayer.empty() ? output.begin() : output.find(config.outputLayer);
                auto outBlob = dynamic_pointer_cast<TBlob<float>>(outputBlobIt->second);

                for (auto value : *outBlob) {
                    *(buff++) = value;
                }
            }

        } else {
            auto outBlob = dynamic_pointer_cast<TBlob<float>>(config.fetch_result(ctx));

            config.outputBlob->deallocate();
            config.outputBlob->getTensorDesc() = TensorDesc(outBlob->getTensorDesc().getPrecision(),
                                                            outBlob->getTensorDesc().getDims(),
                                                            TensorDesc::getLayoutByDims(outBlob->getTensorDesc().getDims()));
            config.outputBlob->allocate();
            float *buff = config.outputBlob->buffer();

            int i = 0;
            for (auto value : *outBlob) {
                buff[i++] = value;
            }
        }
        return;
    }

    if (cmpNear || cmpNearAvg) {
        int idx = 0;
        float avgDiff = 0.0;
        float sz = 0.0;
        float maxDiff = 0.0;
        float maxAverageDiff = 0.0;
        float rms = 0.0;
        int nFrame = -1;
        float avgFrames = 0.0;

        if (!config.fetch_result) {
            decltype(ctx.allOutputs().begin()) output;
            for(;;) {
                avgFrames++;
                if (nFrame == -1) {
                    output = ctx.allOutputs().begin();
                    nFrame = 0;
                } else {
                    nFrame++;
                    ++output;
                }
                if (output == ctx.allOutputs().end()) {
                    break;
                }
                auto outputBlobIt = config.outputLayer.empty() ? output->begin() : output->find(config.outputLayer);
                auto outBlob  = dynamic_pointer_cast<TBlob<float>>(outputBlobIt->second);

                // fo reset case we are storing fake blob pointer
                if (outBlob == nullptr) {
                    avgDiff = 0.0;
                    rms = 0.0;
                    nFrame--;
                    avgFrames = 0.0;
                    continue;
                }
                float rmsp = 0.0;
                float avgDiffp = 0.0;
                ASSERT_LE(outBlob->size(), config.referenceOutput.size());
                for (auto value : *outBlob) {
                    if (cmpNear) {
                       ASSERT_NEAR(value, config.referenceOutput[idx], config.nearValue) << " at " << idx;
                    }
                    auto diff = abs(value - config.referenceOutput[idx]);
                    avgDiffp += diff;
                    rmsp     += diff*diff;
                    maxDiff   = std::max(maxDiff, diff);
                    idx++;
                }

                rmsp = sqrt(rmsp / outBlob->size());
                rms += rmsp;
                avgDiffp /= outBlob->size();
                avgDiff += avgDiffp;
                maxAverageDiff = std::max(maxAverageDiff, avgDiff / avgFrames);

                //TODO: add test_log parse from command line
// #define TEST_LOG
#ifdef TEST_LOG
                auto threshold_similarity_max = config.nearValue - maxDiff;
                auto threshold_similarity_avg = config.nearAvgValue - avgDiff / avgFrames;

                cout << "Frame #  " << nFrame << "\n";
                cout << "MaxDiff   : " << maxDiff << " ("
                    << std::fixed << std::setprecision(5) << threshold_similarity_max <<")" << "\n";
                cout << "RMSE      : " << rmsp << "\n";
                cout << "AvgDiff/f : " << avgDiffp << "\n";
                cout << "MaxAvgDiff: " << maxAverageDiff
                    << std::fixed << std::setprecision(5) << " (" << threshold_similarity_avg <<")" << std::endl;
#endif

                if (cmpNearAvg) {
                    ASSERT_NEAR(avgDiff / avgFrames, 0, config.nearAvgValue);
                }
            }
        } else {
            auto ptr = dynamic_pointer_cast<TBlob<float>>(config.fetch_result(ctx));

            for (auto value : *ptr) {
                if (cmpNear) {
                    ASSERT_NEAR(value, config.referenceOutput[idx], config.nearValue) << " at " << idx;
                }
                if (cmpNearAvg) {
                    avgDiff += abs(value - config.referenceOutput[idx]);
                }
                idx++;
            }
            if (cmpNearAvg) {
                avgDiff /= ptr->size();
            }
        }
    } else {
        // for small expectations lets use string as a compare buddy
        stringstream ss, ssr;

        if (!config.fetch_result) {
            for (auto output : ctx.outputs()) {
                auto outBlob = dynamic_pointer_cast<TBlob<float>>(output.second);
                for (auto value : *outBlob) {
                    ss << setprecision(precision) << fixed << (float)value << ".";
                }
            }
        } else {
            auto ptr = dynamic_pointer_cast<TBlob<float>>(config.fetch_result(ctx));

            for (auto value : *ptr) {
                ss << setprecision(precision) << fixed << (float)value << ".";
            }
        }

        for (auto value : config.referenceOutput) {
            ssr << setprecision(precision) << fixed << (float)value << ".";
        }

        ASSERT_STREQ(ssr.str().c_str(), ss.str().c_str());
    }
}
