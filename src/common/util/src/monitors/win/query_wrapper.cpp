// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "query_wrapper.hpp"

#define NOMINMAX
#include <windows.h>

#include <string>
#include <system_error>
QueryWrapper::QueryWrapper() {
    h_pdh = LoadLibraryA("pdh.dll");
    if (h_pdh) {
        using pdh_open_query_fn = PDH_STATUS (*)(LPCWSTR, DWORD_PTR, PDH_HQUERY*);
        auto p_pdh_open_query = reinterpret_cast<pdh_open_query_fn>(GetProcAddress(h_pdh, "PdhOpenQueryW"));
        if (p_pdh_open_query) {
            PDH_STATUS status = p_pdh_open_query(NULL, NULL, &query);
            if (ERROR_SUCCESS != status) {
                throw std::runtime_error("PdhOpenQuery() failed. Error status: " + std::to_string(status));
            }
        }
    }
}
QueryWrapper::~QueryWrapper() {
    if (h_pdh) {
        using pdh_close_query_fn = void (*)(PDH_HQUERY);
        auto p_pdh_close_query = reinterpret_cast<pdh_close_query_fn>(GetProcAddress(h_pdh, "PdhCloseQuery"));
        if (p_pdh_close_query) {
            p_pdh_close_query(query);
        }
    }
}

bool QueryWrapper::pdh_add_counterW(LPCWSTR sz_full_counter_path, DWORD_PTR dw_user_data, PDH_HCOUNTER* ph_counter) {
    if (!h_pdh) {
        return false;
    }
    using pdh_add_counter_w_fn = PDH_STATUS (*)(PDH_HQUERY, LPCWSTR, DWORD_PTR, PDH_HCOUNTER*);
    auto p_pdh_add_counter_w = reinterpret_cast<pdh_add_counter_w_fn>(GetProcAddress(h_pdh, "PdhAddCounterW"));
    if (p_pdh_add_counter_w) {
        auto status = p_pdh_add_counter_w(query, sz_full_counter_path, dw_user_data, ph_counter);
        if (status != ERROR_SUCCESS)
            throw std::runtime_error("pPdhAddCounterW() failed. Error status: " + std::to_string(status));
        return true;
    }
    return false;
}

bool QueryWrapper::pdh_expand_wild_card_pathW(LPCWSTR sz_data_source,
                                              LPCWSTR sz_wild_card_path,
                                              PZZWSTR msz_expanded_path_list,
                                              LPDWORD pcch_path_list_length,
                                              DWORD dw_flags) {
    if (!h_pdh) {
        return false;
    }
    using pdh_expand_wild_card_path_w_fn = PDH_STATUS (*)(LPCWSTR, LPCWSTR, PZZWSTR, LPDWORD, DWORD);
    auto p_pdh_expand_wild_card_pathw =
        reinterpret_cast<pdh_expand_wild_card_path_w_fn>(GetProcAddress(h_pdh, "PdhExpandWildCardPathW"));
    if (p_pdh_expand_wild_card_pathw) {
        auto status = p_pdh_expand_wild_card_pathw(sz_data_source,
                                                   sz_wild_card_path,
                                                   msz_expanded_path_list,
                                                   pcch_path_list_length,
                                                   dw_flags);
        if (status != ERROR_SUCCESS && status != PDH_MORE_DATA)
            throw std::runtime_error("PPdhExpandWildCardPathW() failed. Error status: " + std::to_string(status));
        return true;
    }
    return false;
}

PDH_STATUS QueryWrapper::pdh_get_formatted_counter_value(PDH_HCOUNTER h_counter,
                                                         DWORD dw_format,
                                                         LPDWORD lpdw_type,
                                                         PPDH_FMT_COUNTERVALUE p_value) {
    if (!h_pdh) {
        return ERROR_INVALID_HANDLE;
    }
    using pdh_get_formatted_counter_value_fn = PDH_STATUS (*)(PDH_HCOUNTER, DWORD, LPDWORD, PPDH_FMT_COUNTERVALUE);
    auto p_pdh_get_formatted_counter_value =
        reinterpret_cast<pdh_get_formatted_counter_value_fn>(GetProcAddress(h_pdh, "PdhGetFormattedCounterValue"));
    if (p_pdh_get_formatted_counter_value) {
        return p_pdh_get_formatted_counter_value(h_counter, dw_format, lpdw_type, p_value);
    }
    return ERROR_INVALID_FUNCTION;
}

bool QueryWrapper::pdh_collect_query_data() {
    if (!h_pdh) {
        return false;
    }
    using pdh_collect_query_data_fn = PDH_STATUS (*)(PDH_HQUERY);
    auto p_pdh_collect_query_data =
        reinterpret_cast<pdh_collect_query_data_fn>(GetProcAddress(h_pdh, "PdhCollectQueryData"));
    if (p_pdh_collect_query_data) {
        auto status = p_pdh_collect_query_data(query);
        if (status != ERROR_SUCCESS)
            throw std::runtime_error("PdhCollectQueryData() failed. Error status: " + std::to_string(status));
        return true;
    }
    return false;
}

bool QueryWrapper::pdh_set_counter_scale_factor(PDH_HCOUNTER h_counter, LONG l_factor) {
    if (!h_pdh) {
        return false;
    }
    using pdh_set_counter_scale_factor_fn = PDH_STATUS (*)(PDH_HCOUNTER, LONG);
    auto p_pdh_set_counter_scale_factor =
        reinterpret_cast<pdh_set_counter_scale_factor_fn>(GetProcAddress(h_pdh, "PdhSetCounterScaleFactor"));
    if (p_pdh_set_counter_scale_factor) {
        auto status = p_pdh_set_counter_scale_factor(h_counter, l_factor);
        if (status != ERROR_SUCCESS)
            throw std::runtime_error("PdhSetCounterScaleFactor() failed. Error status: " + std::to_string(status));
        return true;
    }
    return false;
}