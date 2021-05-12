// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"
#include "ie_parallel.hpp"
#include <ngraph/op/ctc_loss.hpp>
#include <nodes/common/tensor_desc_creator.h>

#include <cmath>


namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

using MKLDNNPlugin::TensorDescCreatorTypes;

class CTCLossImpl : public ExtLayerBase {
public:
    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
        try {
            auto ctcLossOp = ngraph::as_type_ptr<const ngraph::op::v4::CTCLoss>(op);
            if (!ctcLossOp) {
                errorMessage = "Node is not an instance of the CTCLoss operation from operation set v4.";
                return false;
            }
        } catch (...) {
            return false;
        }

        return true;
    }

    explicit CTCLossImpl(const std::shared_ptr<ngraph::Node>& op) {
        try {
            std::string errorMessage;
            if (!isSupportedOperation(op, errorMessage)) {
                IE_THROW(NotImplemented) << errorMessage;
            }

            _logPrefix = std::string("CTCLoss layer with name '") + op->get_friendly_name() + "'";

            if (op->get_input_size() != 4 && op->get_input_size() != 5)
                IE_THROW() << _logPrefix << " has invalid inputs number.";

            auto ctcLossOp = ngraph::as_type_ptr<const ngraph::op::v4::CTCLoss>(op);
            _ctcMergeRepeated = ctcLossOp->get_ctc_merge_repeated();
            _preprocessCollapseRepeated = ctcLossOp->get_preprocess_collapse_repeated();
            _unique = ctcLossOp->get_unique();

            std::vector<DataConfigurator> inDataConfigurators;
            inDataConfigurators.push_back({TensorDescCreatorTypes::ncsp, Precision::FP32});
            for (int i = 1; i < op->get_input_size(); i++) {
                inDataConfigurators.push_back({TensorDescCreatorTypes::ncsp, Precision::I32});
            }
            addConfig(op, inDataConfigurators,
                          {{TensorDescCreatorTypes::ncsp, Precision::FP32}});
        } catch (InferenceEngine::Exception &ex) {
            errorMsg = ex.what();
            throw;
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs,
                       std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        StatusCode returnCode = OK;

        const float* logits = inputs[0]->cbuffer().as<const float*>() +
            inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const int* logitsLength = inputs[1]->cbuffer().as<const int*>() +
            inputs[1]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const int* labels = inputs[2]->cbuffer().as<const int*>() +
            inputs[2]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const int* labelsLength = inputs[3]->cbuffer().as<const int*>() +
            inputs[3]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        float* dstData = outputs[0]->buffer().as<float*>() +
            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        const auto& logitsShape = inputs[0]->getTensorDesc().getDims();
        const size_t batchNum = logitsShape[0];
        const size_t maxTime = logitsShape[1];
        const size_t classesNum = logitsShape[2];

        int blankIndex = classesNum - 1;
        if (inputs.size() > 4) {
            blankIndex = inputs[4]->cbuffer().as<const int*>()[0];
        }

        std::vector<int> decodedTargetLenB(batchNum, 0);
        std::vector<std::vector<int>> targetDB(batchNum);
        std::vector<std::vector<std::vector<float>>> logProbabilitiesB(batchNum);
        std::vector<std::string> errorMsgB(parallel_get_max_threads());

        auto threadBody_1 = [&](const int ithr, const int nthr) {
            size_t start(0lu), end(0lu);
            splitter(batchNum, nthr, ithr, start, end);
            if (start >= end)
                return;

            for (size_t b = start; b < end; b++) {
                if (logitsLength[b] < 0 || labelsLength[b] < 0 || logitsLength[b] > maxTime || labelsLength[b] > logitsLength[b]) {
                    errorMsgB[ithr] = _logPrefix + ". Logit length cannot be greater than max sequence length. "
                        + "Label length cannot be greater than a logit length"
                        + " and both cannot be negative.\nMaxSeqLen: "
                        + std::to_string(maxTime) + "; Logit len: " + std::to_string(logitsLength[b])
                        + "; Label len: " + std::to_string(labelsLength[b]);
                    returnCode = GENERAL_ERROR;
                    return;
                }
                const size_t actualLogitLen = logitsLength[b];
                const size_t actualTargetLen = labelsLength[b];
                size_t decodedTargetLen = 0lu;

                // Decoding target: merge repeated characters if preprocess_collapse_repeated == True,
                // find unique elemnts if unique == True.
                // Inserts blanks before each index and a blank at the end.
                const int* target = &labels[b * maxTime];
                targetDB[b].resize(actualTargetLen * 2 + 1);
                auto& targetD = targetDB[b];
                if (_unique) {
                    std::unordered_set<int> uniqVals;
                    for (size_t t = 0lu; t < actualTargetLen; t++) {
                        if (uniqVals.find(target[t]) != uniqVals.end()) {
                            continue;
                        }
                        uniqVals.insert(target[t]);
                        targetD[decodedTargetLen++] = blankIndex;
                        targetD[decodedTargetLen++] = target[t];
                    }
                    targetD[decodedTargetLen++] = blankIndex;
                } else if (_preprocessCollapseRepeated) {
                    auto prevValue = target[0];
                    targetD[decodedTargetLen++] = blankIndex;
                    targetD[decodedTargetLen++] = target[0];
                    for (size_t t = 1lu; t < actualTargetLen; t++) {
                        if (target[t] == prevValue) {
                            continue;
                        }
                        targetD[decodedTargetLen++] = blankIndex;
                        targetD[decodedTargetLen++] = prevValue = target[t];
                    }
                    targetD[decodedTargetLen++] = blankIndex;
                } else {
                    for (size_t t = 0lu; t < actualTargetLen; t++) {
                        targetD[decodedTargetLen++] = blankIndex;
                        targetD[decodedTargetLen++] = target[t];
                    }
                    targetD[decodedTargetLen++] = blankIndex;
                }
                decodedTargetLenB[b] = decodedTargetLen;

                auto& logProbabilities = logProbabilitiesB[b];
                logProbabilities.resize(actualLogitLen);
                for (size_t ll = 0; ll < actualLogitLen; ll++) {
                    logProbabilities[ll].resize(decodedTargetLen);
                }
            } // for batch
        }; // threadBody_1

        parallel_nt(0, threadBody_1);
        if (returnCode != OK) {
            std::string resErr("");
            for (auto& err : errorMsgB) {
                if (!err.empty())
                    resErr += err + "\n";
                resErr.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return returnCode;
        }

        const size_t TC = maxTime * classesNum;

        size_t workAmount2 = 0lu;
        for (size_t b = 0; b < batchNum; b++) {
            workAmount2 += logitsLength[b];
        }

        auto threadBody_2 = [&](const int ithr, const int nthr) {
            size_t start(0lu), end(0lu);
            size_t sB(0lu), sT(0lu);
            splitter(workAmount2, nthr, ithr, start, end);
            if (start >= end)
                return;
            int64_t cw = 0, st = start;
            for (; sB < batchNum; sB++) {
                cw += logitsLength[sB];
                if (cw >= st) {
                    sT = logitsLength[sB] + st - cw;
                    break;
                }
            }
            size_t workCounter = start;

            for (size_t b = sB; b < batchNum; b++) {
                const size_t actualLogitLen = logitsLength[b];
                const size_t decodedTargetLen = decodedTargetLenB[b];
                auto& logProbabilities = logProbabilitiesB[b];
                auto& targetD = targetDB[b];

                double expSum = 0.0;
                size_t btcT = b * TC + sT * classesNum;
                // logProbabilities = logSoftmax = logits[b][t][c] - ln(sum_c(exp(logits[b][t])))
                for (size_t t = sT; t < actualLogitLen; t++) {
                    expSum = 0.0;
                    for (size_t c = 0lu; c < classesNum; c++) {
                        expSum += std::exp(logits[btcT + c]);
                    }
                    for (size_t s = 0lu; s < decodedTargetLen; s++) {
                        logProbabilities[t][s] = logits[btcT + targetD[s]] - std::log(expSum);
                    }
                    btcT += classesNum;
                    if (++workCounter >= end) {
                        return;
                    }
                }
                sT = 0lu;
            }  // for batch
        }; // threadBody_2

        parallel_nt(0, threadBody_2);

        const auto float_inf = std::numeric_limits<float>::infinity();

        auto sumLogs = [&float_inf](float log1, float log2) {
            if (log1 == -float_inf) {
                return log2;
            } else if (log2 == -float_inf) {
                return log1;
            } else {
                if (log1 > log2)
                    return log1 + std::log1pf(std::exp(log2 - log1));
                else
                    return log2 + std::log1pf(std::exp(log1 - log2));
            }
        };

        auto threadBody_3 = [&](const int ithr, const int nthr) {
            size_t start(0lu), end(0lu);
            splitter(batchNum, nthr, ithr, start, end);
            if (start >= end)
                return;

            // As per Connectionist Temporal Classification - Labeling Unsegmented Sequence Data with Recurrent Neural Networks:
            // Graves et al., 2016, paragraph 4.1 (10)
            for (size_t b = start; b < end; b++) {
                auto& targetD = targetDB[b];
                auto& logProbabilities = logProbabilitiesB[b];
                const int actualLogitLen = logitsLength[b];
                const int decodedTargetLen = decodedTargetLenB[b];
                std::vector<std::vector<float>> logBwd(decodedTargetLen, std::vector<float>(actualLogitLen, -float_inf));
                for (int s = decodedTargetLen - 2; s < decodedTargetLen; s++)
                    logBwd[s][actualLogitLen - 1] = 0.f;

                for (int t = actualLogitLen - 2; t >= 0; t--) {
                    const int t_1 = t + 1;
                    for (int s = std::max(0, decodedTargetLen - (2 * (actualLogitLen - t)));
                            s < std::min(decodedTargetLen, 2 * (t_1)); s++) {
                        if (_ctcMergeRepeated || targetD[s] == blankIndex) {
                            logBwd[s][t] = sumLogs(logBwd[s][t],
                                logBwd[s][t_1] + logProbabilities[t_1][s]);
                        }

                        if (s + 1 < decodedTargetLen) {
                            logBwd[s][t] = sumLogs(logBwd[s][t],
                                logBwd[s + 1][t_1] + logProbabilities[t_1][s + 1]);
                        }

                        if (s + 2 < decodedTargetLen) {
                            if (targetD[s] != blankIndex && (!_ctcMergeRepeated || (targetD[s] != targetD[s + 2]))) {
                                logBwd[s][t] = sumLogs(logBwd[s][t],
                                    logBwd[s + 2][t_1] + logProbabilities[t_1][s + 2]);
                            }
                        }
                    }
                }

                logBwd[0][0] += logProbabilities[0][0];
                logBwd[1][0] += logProbabilities[0][(decodedTargetLen > 1) ? 1 : 0];

                dstData[b] = -sumLogs(logBwd[0][0], logBwd[1][0]);
            } // for batch
        }; // threadBody_3

        parallel_nt(0, threadBody_3);

        return returnCode;
    } // execute

protected:
    bool _ctcMergeRepeated;
    bool _preprocessCollapseRepeated;
    bool _unique;

    std::string _logPrefix;
};

REG_FACTORY_FOR(CTCLossImpl, CTCLoss);
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
