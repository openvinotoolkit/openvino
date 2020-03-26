// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "hddl_bsl_priv.h"

static BSL_DEVICE_NAME_MATCH s_cfg_device_type_str[BSL_DEVICE_TYPE_MAX] = {
    {"hid-f75114", HID_F75114},
    {"ioexpander", I2C_IOEXPANDER},
    {"mcu", I2C_MCU},
    //{"pch-c246", PCH_C246},
};

BSL_DEVICE_NAME_MATCH* cfg_get_device_type_pair(int idx) {
  return &s_cfg_device_type_str[idx];
}