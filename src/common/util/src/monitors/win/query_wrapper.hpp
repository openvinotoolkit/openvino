// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#define NOMINMAX
#include <pdh.h>
#include <PdhMsg.h>
class QueryWrapper {
public:
    QueryWrapper();
    ~QueryWrapper();
    QueryWrapper(const QueryWrapper&) = delete;
    QueryWrapper& operator=(const QueryWrapper&) = delete;
    bool pdhAddCounterW(LPCWSTR szFullCounterPath, DWORD_PTR dwUserData, PDH_HCOUNTER* phCounter);
    bool pdhExpandWildCardPathW(LPCWSTR szDataSource,
                                LPCWSTR szWildCardPath,
                                PZZWSTR mszExpandedPathList,
                                LPDWORD pcchPathListLength,
                                DWORD dwFlags);
    PDH_STATUS pdhGetFormattedCounterValue(PDH_HCOUNTER hCounter,
                                           DWORD dwFormat,
                                           LPDWORD lpdwType,
                                           PPDH_FMT_COUNTERVALUE pValue);
    bool pdhCollectQueryData();
    bool pdhSetCounterScaleFactor(PDH_HCOUNTER hCounter, LONG lFactor);

private:
    PDH_HQUERY query;
    HMODULE hPdh;
};
