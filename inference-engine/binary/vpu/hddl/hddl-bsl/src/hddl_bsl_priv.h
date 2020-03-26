// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#ifndef __HDDL_BSL_PRIV_H__
#define __HDDL_BSL_PRIV_H__

#include <assert.h>
#include <stdio.h>
#include <sys/stat.h>
#include <stdbool.h>

#include "hddl-bsl.h"
#include "osl/osl.h"
#include "bsl_cfg.h"

#ifdef DEBUG
#include <stdio.h>
#define LOG(...) printf(__VA_ARGS__)
#else
#define LOG(...) \
  do {           \
  } while (0)
#endif

typedef BSL_STATUS (*device_init_t)();
typedef BSL_STATUS (*device_reset_t)(int);
typedef BSL_STATUS (*device_reset_all_t)();
typedef BSL_STATUS (*device_config_t)(struct json_object* config);
typedef BSL_STATUS (*device_add_t)(int);
typedef BSL_STATUS (*device_scan_t)(int*);
typedef BSL_STATUS (*device_get_device_num)(int*);
typedef BSL_STATUS (*device_discard_t)(int);
typedef BSL_STATUS (*device_destroy_t)();

typedef struct {
  device_init_t device_init;
  device_reset_t device_reset;
  device_reset_all_t device_reset_all;
  device_config_t device_config;
  device_add_t device_add;
  device_scan_t device_scan;
  device_scan_t device_get_device_num;
  device_discard_t device_discard;
  device_destroy_t device_destroy;
} HddlController_t;

void mcu_init(HddlController_t* ctrl);
void ioexpander_init(HddlController_t* ctrl);
void hid_f75114_init(HddlController_t* ctrl);
// void c246_init(HddlController_t* ctrl);

typedef struct {
  const char* device_name;
  BslDeviceType device_type;
} BSL_DEVICE_NAME_MATCH;

BSL_DEVICE_NAME_MATCH* cfg_get_device_type_pair(int idx);
bool is_valid_device_type(BslDeviceType bsl_device);

#endif
