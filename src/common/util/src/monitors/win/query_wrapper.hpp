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
    PDH_STATUS pdhAddCounterW(LPCWSTR szFullCounterPath, DWORD_PTR dwUserData, PDH_HCOUNTER* phCounter);
    PDH_STATUS pdhExpandWildCardPathW(LPCWSTR szDataSource,
                                      LPCWSTR szWildCardPath,
                                      PZZWSTR mszExpandedPathList,
                                      LPDWORD pcchPathListLength,
                                      DWORD dwFlags);
    PDH_STATUS pdhGetFormattedCounterValue(PDH_HCOUNTER hCounter,
                                           DWORD dwFormat,
                                           LPDWORD lpdwType,
                                           PPDH_FMT_COUNTERVALUE pValue);
    PDH_STATUS pdhCollectQueryData();
    PDH_STATUS pdhSetCounterScaleFactor(PDH_HCOUNTER hCounter, LONG lFactor);

private:
    PDH_HQUERY query;
    HMODULE hPdh;
};
