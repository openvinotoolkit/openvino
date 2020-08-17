// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <math.h>
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
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
                         T* output)
            {
                const size_t batchNum = logitsShape[0];
                const size_t maxTime = logitsShape[1];
                const size_t classesNum = logitsShape[2];
                U blankIndex = classesNum - 1;
                if (blankIndexP != nullptr)
                {
                    blankIndex = blankIndexP[0];
                }

                std::vector<U> targetD(maxTime);
                std::vector<U> pathS(maxTime);

                const size_t TC = maxTime * classesNum;

                for (size_t b = 0; b < batchNum; b++)
                {
                    U actualLogitLen = logitsLength[b];
                    U actualTargetLen = labelsLength[b];
                    if (actualLogitLen >= maxTime || actualTargetLen >= maxTime ||
                        actualTargetLen > actualLogitLen)
                    {
                        throw ngraph_error(
                            std::string("Logit or label length cannot be more than max sequence"
                                        "length. Also a label length cannot be greater than a"
                                        "logit length.\nMaxSeqLen: ") +
                            std::to_string(maxTime) + "; Logit len: " +
                            std::to_string(actualLogitLen) + "; Label len: " +
                            std::to_string(actualTargetLen));
                    }

                    const U* target = &labels[b * maxTime];
                    // Decoding target:
                    // merge repeated characters if preprocess_collapse_repeated == True,
                    // find unique elemnts if unique == True
                    size_t decodedTargetLen = 0lu;
                    if (unique)
                    {
                        std::unordered_set<U> uniqVals;
                        for (size_t t = 0lu; t < actualTargetLen; t++)
                        {
                            if (uniqVals.find(target[t]) != uniqVals.end())
                            {
                                continue;
                            }
                            uniqVals.insert(target[t]);
                            targetD[decodedTargetLen++] = target[t];
                        }
                    }
                    else if (preprocessCollapseRepeated)
                    {
                        U prevValue = target[0];
                        targetD[decodedTargetLen++] = target[0];
                        for (size_t t = 1lu; t < actualTargetLen; t++)
                        {
                            if (target[t] == prevValue)
                            {
                                continue;
                            }
                            targetD[decodedTargetLen++] = target[t];
                            prevValue = target[t];
                        }
                    }
                    else
                    {
                        std::copy(target, target + actualTargetLen, targetD.data());
                        decodedTargetLen = actualTargetLen;
                    }

                    const size_t BTC = b * TC;

                    std::vector<T> kExp(actualLogitLen, 0);
                    for (size_t t = 0; t < actualLogitLen; t++)
                    {
                        size_t btcT = BTC + classesNum * t;
                        for (size_t c = 0; c < classesNum; c++)
                        {
                            kExp[t] += std::exp(logits[btcT + c]);
                        }
                    }

                    T res = -std::numeric_limits<T>::infinity();

                    // Looking for aligned paths
                    std::function<void(size_t targetIdx, size_t start, size_t end)> findPaths = [&](
                        size_t targetIdx, size_t start, size_t end) {
                        if (end > actualLogitLen)
                        {
                            T prod = 0;
                            for (size_t t = 0; t < actualLogitLen; t++)
                            {
                                prod += std::log(std::exp(logits[BTC + classesNum * t + pathS[t]]) /
                                                 kExp[t]);
                            }
                            if (res == -std::numeric_limits<T>::infinity())
                                res = prod;
                            else if (prod != -std::numeric_limits<T>::infinity())
                                res = res + std::log1pf(std::exp(prod - res));

                            return;
                        }

                        size_t nextIdx = targetIdx + 1;
                        int64_t st64 = start;
                        if (!ctcMergeRepeated)
                        {
                            for (size_t pos = start; pos < end; pos++)
                            {
                                for (size_t bl = start; bl < pos; bl++)
                                {
                                    pathS[bl] = blankIndex;
                                }
                                pathS[pos] = targetD[targetIdx];
                                findPaths(nextIdx, pos + 1, end + 1);
                            }
                        }
                        else
                        {
                            for (size_t pos = start; pos < end; pos++)
                            {
                                for (size_t bl = start; bl < pos; bl++)
                                {
                                    pathS[bl] = blankIndex;
                                }
                                for (int64_t bl = pos; bl >= st64; bl--)
                                {
                                    pathS[bl] = targetD[targetIdx];
                                    if (end == actualLogitLen)
                                    {
                                        for (int64_t ble = pos + 1; ble < actualLogitLen; ble++)
                                        {
                                            pathS[ble] = blankIndex;
                                        }
                                    }
                                    size_t next_start = pos + 1;
                                    if (targetIdx < decodedTargetLen - 1 &&
                                        targetD[targetIdx] == targetD[targetIdx + 1])
                                    {
                                        pathS[next_start++] = blankIndex;
                                    }
                                    findPaths(nextIdx, next_start, end + 1);
                                }
                            }
                        }
                    }; // findPaths

                    findPaths(0lu, 0lu, actualLogitLen - decodedTargetLen + 1lu);

                    output[b] = -res;

                } // for (size_t b = 0; b < batchNum; b++)
            }     // CTCLoss
        }         // reference
    }             // runtime
} // ngraph
