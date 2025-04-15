// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "query_wrapper.hpp"

#define NOMINMAX
#include <windows.h>

#include <string>
#include <system_error>
QueryWrapper::QueryWrapper() {
    hPdh = LoadLibraryA("pdh.dll");
    if (hPdh) {
        using PdhOpenQuery_fn = PDH_STATUS (*)(LPCWSTR, DWORD_PTR, PDH_HQUERY*);
        auto pPdhOpenQuery = reinterpret_cast<PdhOpenQuery_fn>(GetProcAddress(hPdh, "PdhOpenQueryW"));
        if (pPdhOpenQuery) {
            PDH_STATUS status = pPdhOpenQuery(NULL, NULL, &query);
            if (ERROR_SUCCESS != status) {
                throw std::runtime_error("PdhOpenQuery() failed. Error status: " + std::to_string(status));
            }
        }
    }
}
QueryWrapper::~QueryWrapper() {
    if (hPdh) {
        using PdhCloseQuery_fn = void (*)(PDH_HQUERY);
        auto pPdhCloseQuery = reinterpret_cast<PdhCloseQuery_fn>(GetProcAddress(hPdh, "PdhCloseQuery"));
        if (pPdhCloseQuery) {
            pPdhCloseQuery(query);
        }
    }
}

PDH_STATUS QueryWrapper::pdhExpandWildCardPathW(LPCWSTR szDataSource,
                                                LPCWSTR szWildCardPath,
                                                PZZWSTR mszExpandedPathList,
                                                LPDWORD pcchPathListLength,
                                                DWORD dwFlags) {
    if (!hPdh) {
        return ERROR;
    }
    try {
        using PdhExpandWildCardPathW_fn = PDH_STATUS (*)(LPCWSTR, LPCWSTR, PZZWSTR, LPDWORD, DWORD);
        auto pPdhExpandWildCardPathW =
            reinterpret_cast<PdhExpandWildCardPathW_fn>(GetProcAddress(hPdh, "PdhExpandWildCardPathW"));
        if (pPdhExpandWildCardPathW) {
            return pPdhExpandWildCardPathW(szDataSource,
                                           szWildCardPath,
                                           mszExpandedPathList,
                                           pcchPathListLength,
                                           dwFlags);
            // return status == ERROR_SUCCESS || status == PDH_MORE_DATA;
        }
    } catch (...) {
        return ERROR;
    }
    return ERROR;
}

PDH_STATUS QueryWrapper::pdhAddCounterW(LPCWSTR szFullCounterPath, DWORD_PTR dwUserData, PDH_HCOUNTER* phCounter) {
    if (!hPdh) {
        return ERROR;
    }
    try {
        using PdhAddCounterW_fn = PDH_STATUS (*)(PDH_HQUERY, LPCWSTR, DWORD_PTR, PDH_HCOUNTER*);
        auto pPdhAddCounterW = reinterpret_cast<PdhAddCounterW_fn>(GetProcAddress(hPdh, "PdhAddCounterW"));
        if (pPdhAddCounterW) {
            return pPdhAddCounterW(query, szFullCounterPath, dwUserData, phCounter);
        }
    } catch (...) {
        return ERROR;
    }
    return ERROR;
}
PDH_STATUS QueryWrapper::pdhGetFormattedCounterValue(PDH_HCOUNTER hCounter,
                                                     DWORD dwFormat,
                                                     LPDWORD lpdwType,
                                                     PPDH_FMT_COUNTERVALUE pValue) {
    if (!hPdh) {
        return ERROR;
    }
    try {
        using PdhGetFormattedCounterValue_fn = PDH_STATUS (*)(PDH_HCOUNTER, DWORD, LPDWORD, PPDH_FMT_COUNTERVALUE);
        auto pPdhGetFormattedCounterValue =
            reinterpret_cast<PdhGetFormattedCounterValue_fn>(GetProcAddress(hPdh, "PdhGetFormattedCounterValue"));
        if (pPdhGetFormattedCounterValue) {
            return pPdhGetFormattedCounterValue(hCounter, dwFormat, lpdwType, pValue);
        }
    } catch (...) {
        return ERROR;
    }
    return ERROR;
}

PDH_STATUS QueryWrapper::pdhCollectQueryData() {
    if (!hPdh) {
        return ERROR;
    }
    try {
        using PdhCollectQueryData_fn = PDH_STATUS (*)(PDH_HQUERY);
        auto pPdhCollectQueryData =
            reinterpret_cast<PdhCollectQueryData_fn>(GetProcAddress(hPdh, "PdhCollectQueryData"));
        if (pPdhCollectQueryData) {
            return pPdhCollectQueryData(query);
        }
    } catch (...) {
        return ERROR;
    }
    return ERROR;
}

PDH_STATUS QueryWrapper::pdhSetCounterScaleFactor(PDH_HCOUNTER hCounter, LONG dwScaleFactor) {
    if (!hPdh) {
        return ERROR;
    }
    try {
        using PdhSetCounterScaleFactor_fn = PDH_STATUS (*)(PDH_HCOUNTER, LONG);
        auto pPdhSetCounterScaleFactor =
            reinterpret_cast<PdhSetCounterScaleFactor_fn>(GetProcAddress(hPdh, "PdhSetCounterScaleFactor"));
        if (pPdhSetCounterScaleFactor) {
            return pPdhSetCounterScaleFactor(hCounter, dwScaleFactor);
        }
    } catch (...) {
        return ERROR;
    }
    return ERROR;
}