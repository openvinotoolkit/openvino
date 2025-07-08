// Copyright (C) 2019-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#define NOMINMAX
#include <pdh.h>
#include <PdhMsg.h>

#include <filesystem>
class QueryWrapper {
public:
    QueryWrapper();
    ~QueryWrapper();
    QueryWrapper(const QueryWrapper&) = delete;
    QueryWrapper& operator=(const QueryWrapper&) = delete;
    bool pdh_add_counterW(const std::filesystem::path& sz_full_counter_path,
                          DWORD_PTR dw_user_data,
                          PDH_HCOUNTER* ph_counter);
    bool pdh_expand_wild_card_pathW(const std::filesystem::path& sz_data_source,
                                    const std::filesystem::path& sz_wild_card_path,
                                    PZZWSTR msz_expanded_path_list,
                                    LPDWORD pcch_path_list_length,
                                    DWORD dw_flags);
    PDH_STATUS pdh_get_formatted_counter_value(PDH_HCOUNTER h_counter,
                                               DWORD dw_format,
                                               LPDWORD lpdw_type,
                                               PPDH_FMT_COUNTERVALUE p_value);
    bool pdh_collect_query_data();
    bool pdh_set_counter_scale_factor(PDH_HCOUNTER h_counter, LONG l_factor);

private:
    PDH_HQUERY query;
    HMODULE h_pdh;
};
