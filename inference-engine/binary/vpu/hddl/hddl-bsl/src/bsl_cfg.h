// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#ifndef HDDL_BSL_BSL_CFG_H
#define HDDL_BSL_BSL_CFG_H

#include <stdbool.h>
#include <json-c/json.h>

typedef struct json_object* CFG_HANDLER;
#define CFG_HANDLER_DEFAULT_VALUE NULL

BSL_STATUS cfg_get_path(char* path, size_t buffer_size);

BSL_STATUS cfg_open(const char* file_path, CFG_HANDLER* handler);
BSL_STATUS cfg_close(CFG_HANDLER handler);

CFG_HANDLER cfg_get_field(CFG_HANDLER config, const char* field_name);
bool cfg_get_autoscan_switch(CFG_HANDLER handler);
bool cfg_type_is_enabled(CFG_HANDLER config);

void cfg_print_all(CFG_HANDLER config);

#endif  // HDDL_BSL_BSL_CFG_H
