// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <limits>
#include <unordered_set>

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
template <typename T, typename U>
void CTCLoss(const T* logits,
             const Shape& logitsShape,
             const U* logitsLength,
             const U* labels,
             const U* labelsLength,
             const U* blankIndexP,
             const bool preprocessCollapseRepeated,
             const bool ctcMergeRepeated,
             const bool unique,
             T* output) {
    const size_t batchNum = logitsShape[0];
    const size_t maxTime = logitsShape[1];
    const size_t classesNum = logitsShape[2];
    U blankIndex = static_cast<U>(classesNum - 1);
    if (blankIndexP != nullptr) {
        blankIndex = blankIndexP[0];
    }

    std::vector<U> targetD(maxTime);
    std::vector<U> pathS(maxTime);

    const size_t TC = maxTime * classesNum;

    for (size_t b = 0; b < batchNum; b++) {
        U actualLogitLen = logitsLength[b];
        U actualTargetLen = labelsLength[b];
        if (static_cast<size_t>(actualLogitLen) > maxTime || static_cast<size_t>(actualTargetLen) > maxTime ||
            actualTargetLen > actualLogitLen) {
            OPENVINO_THROW(std::string("Logit or label length cannot greater than max sequence"
                                       "length. Also a label length cannot be greater than a"
                                       "logit length.\nMaxSeqLen: ") +
                           std::to_string(maxTime) + "; Logit len: " + std::to_string(actualLogitLen) +
                           "; Label len: " + std::to_string(actualTargetLen));
        }

        const U* target = &labels[b * maxTime];
        // Decoding target:
        // merge repeated characters if preprocess_collapse_repeated == True,
        // find unique elemnts if unique == True
        size_t decodedTargetLen = 0lu;
        if (unique) {
            std::unordered_set<U> uniqVals;
            for (size_t t = 0lu; t < static_cast<size_t>(actualTargetLen); t++) {
                if (uniqVals.find(target[t]) != uniqVals.end()) {
                    continue;
                }
                uniqVals.insert(target[t]);
                targetD[decodedTargetLen++] = target[t];
            }
        } else if (preprocessCollapseRepeated) {
            U prevValue = target[0];
            targetD[decodedTargetLen++] = target[0];
            for (size_t t = 1lu; t < static_cast<size_t>(actualTargetLen); t++) {
                if (target[t] == prevValue) {
                    continue;
                }
                targetD[decodedTargetLen++] = target[t];
                prevValue = target[t];
            }
        } else {
            std::copy(target, target + actualTargetLen, targetD.data());
            decodedTargetLen = actualTargetLen;
        }

        const size_t BTC = b * TC;

        std::vector<std::unordered_map<size_t, T>> logProbabilities(actualLogitLen);
        T logProb = 0.f, kExp = 0.f;
        for (size_t t = 0; t < static_cast<size_t>(actualLogitLen); t++) {
            kExp = 0.f;
            const size_t btcT = BTC + classesNum * t;
            for (size_t c = 0; c < classesNum; c++) {
                kExp += std::exp(logits[btcT + c]);
            }
            for (size_t s = 0; s < decodedTargetLen; s++) {
                logProb = logits[btcT + targetD[s]] - std::log(kExp);
                logProbabilities[t].insert({targetD[s], logProb});
            }
            logProb = logits[btcT + blankIndex] - std::log(kExp);
            logProbabilities[t].insert({blankIndex, logProb});
        }

        const auto type_inf = std::numeric_limits<T>::infinity();
        T res = -type_inf;

        // Looking for aligned paths
        std::function<void(size_t, size_t, size_t, T)> findPaths =
            [&](size_t targetIdx, size_t start, size_t end, T prevLogProb) {
                if (end > static_cast<size_t>(actualLogitLen)) {
                    if (res == -type_inf) {
                        res = prevLogProb;
                    } else if (prevLogProb != -type_inf) {
                        if (res > prevLogProb)
                            res = res + static_cast<T>(std::log1pf(static_cast<float>(std::exp(prevLogProb - res))));
                        else
                            res = prevLogProb +
                                  static_cast<T>(std::log1pf(static_cast<float>(std::exp(res - prevLogProb))));
                    }
                    return;
                }

                size_t nextIdx = targetIdx + 1;
                int64_t st64 = start;
                T newLogProb = prevLogProb;
                if (!ctcMergeRepeated) {
                    for (size_t pos = start; pos < end; pos++) {
                        newLogProb = prevLogProb;
                        for (size_t bl = start; bl < pos; bl++) {
                            newLogProb += logProbabilities[bl][blankIndex];
                        }
                        newLogProb += logProbabilities[pos][targetD[targetIdx]];
                        if (end == static_cast<size_t>(actualLogitLen)) {
                            for (size_t ble = pos + 1; ble < static_cast<size_t>(actualLogitLen); ble++) {
                                newLogProb += logProbabilities[ble][blankIndex];
                            }
                        }
                        findPaths(nextIdx, pos + 1, end + 1, newLogProb);
                    }
                } else {
                    for (size_t pos = start; pos < end; pos++) {
                        newLogProb = prevLogProb;
                        size_t next_start = pos + 1;
                        for (size_t bl = start; bl < pos; bl++) {
                            newLogProb += logProbabilities[bl][blankIndex];
                        }
                        if (end == static_cast<size_t>(actualLogitLen)) {
                            for (int64_t ble = pos + 1; ble < actualLogitLen; ble++) {
                                newLogProb += logProbabilities[ble][blankIndex];
                            }
                        }
                        if (targetIdx < decodedTargetLen - 1 && targetD[targetIdx] == targetD[targetIdx + 1]) {
                            newLogProb += logProbabilities[next_start++][blankIndex];
                        }
                        for (int64_t bl = pos; bl >= st64; bl--) {
                            newLogProb += logProbabilities[bl][targetD[targetIdx]];
                            findPaths(nextIdx, next_start, end + 1, newLogProb);
                            if (bl > 0)
                                newLogProb -= logProbabilities[bl - 1][blankIndex];
                        }
                    }
                }
            };  // findPaths

        findPaths(0lu, 0lu, actualLogitLen - decodedTargetLen + 1lu, 0.f);

        output[b] = -res;

    }  // for (size_t b = 0; b < batchNum; b++)
}  // CTCLoss
}  // namespace reference
}  // namespace ov
