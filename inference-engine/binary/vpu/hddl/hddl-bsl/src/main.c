// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <hddl-bsl.h>
#include <stdio.h>
#include <stdlib.h>

#include "hddl-bsl.h"
#include "hddl_bsl_priv.h"

HddlController_t m_hddl_controller[BSL_DEVICE_TYPE_MAX];
static BslDeviceType m_bsl_device = DEVICE_INVALID_TYPE;

static void bsl_fill_all_callback_functions();

static BSL_STATUS hddl_bsl_init_with_config(char* cfg_path);
static BSL_STATUS bsl_set_and_config_device(BslDeviceType dev_type, CFG_HANDLER config);
static BSL_STATUS bsl_init_by_auto_scan();

static BSL_STATUS hddl_bsl_init();
static BSL_STATUS hddl_bsl_destroy();

#ifndef WIN32
__attribute__((constructor))
#endif
void libbsl_init()
{
  BSL_STATUS status = hddl_bsl_init();
  if (status != BSL_SUCCESS) {
    printf("bsl init failed for:\t");
    hddl_get_error_string(status);
  }
}

#ifdef WIN32
typedef void(__cdecl* PF)();
#pragma section(".CRT$XCG", read)
__declspec(allocate(".CRT$XCG")) PF lib_construct[1] = {libbsl_init};
#endif

#ifndef WIN32
__attribute__((destructor))
#endif
void libbsl_destroy()
{
  hddl_bsl_destroy();
}

static BSL_STATUS hddl_bsl_init() {
  bsl_fill_all_callback_functions();

  char path[MAX_PATH_LENGTH];
  BSL_STATUS status = cfg_get_path(path, sizeof(path));

  if (status == BSL_SUCCESS) {
    printf("Config file detected at %s\n", path);
    return hddl_bsl_init_with_config(path);
  }

  if (status == BSL_ERROR_HDDL_INSTALL_DIR_TOO_LONG) {
    printf("${HDDL_INSTALL_DIR} is too long\n");
    printf("Please reduce the length of this value then retry\n");
    return status;
  }

  if (status == BSL_ERROR_HDDL_INSTALL_DIR_NOT_DIR) {
    printf("${HDDL_INSTALL_DIR}=%s is not a valid dir\n", path);
    printf("Please check the correctness of that path\n");
    return status;
  }

  if (status == BSL_ERROR_HDDL_INSTALL_DIR_NOT_PROVIDED) {
    printf("${HDDL_INSTALL_DIR} not provided\n");
  }

  if (status == BSL_ERROR_CFG_OPEN_FAILED) {
    printf("Config file decided based on ${HDDL_INSTALL_DIR} is %s\n", path);
    printf("Config file open failed due to non-existing or permissions\n");
  }

  printf("Trying to init with auto-scan\n");
  return bsl_init_by_auto_scan();
}

static BSL_STATUS hddl_bsl_init_with_config(char* cfg_path) {
  CFG_HANDLER config = CFG_HANDLER_DEFAULT_VALUE;
  BSL_STATUS status;
  status = cfg_open(cfg_path, &config);
  for (int i = 0; i < BSL_DEVICE_TYPE_MAX; i++) {
    BSL_DEVICE_NAME_MATCH* name_pair = cfg_get_device_type_pair(i);
    CFG_HANDLER device_cfg = cfg_get_field(config, name_pair->device_name);
    if (device_cfg == NULL) {
      continue;
    }

    bool enabled = cfg_type_is_enabled(device_cfg);
    if (!enabled) {
      printf("%s is disabled by config, skipping\n", name_pair->device_name);
      continue;
    }

    status = bsl_set_and_config_device(name_pair->device_type, device_cfg);
    if (status == BSL_SUCCESS) {
      cfg_close(config);
      return status;
    }
    printf("%s init returned status ", name_pair->device_name);
    hddl_get_error_string(status);
  }

  if (cfg_get_autoscan_switch(config)) {
    status = bsl_init_by_auto_scan();
  } else {
    printf("Auto-scan is disabled by config, aborting\n");
  }

  cfg_close(config);
  return status;
}

void bsl_fill_all_callback_functions() {
  mcu_init(&m_hddl_controller[I2C_MCU]);
  ioexpander_init(&m_hddl_controller[I2C_IOEXPANDER]);
  hid_f75114_init(&m_hddl_controller[HID_F75114]);
  // c246_init(&m_hddl_controller[PCH_C246]);
}

BSL_STATUS bsl_set_and_config_device(BslDeviceType dev_type, CFG_HANDLER config) {
  BSL_STATUS status;

  device_config_t device_config_callback = m_hddl_controller[dev_type].device_config;
  if (device_config_callback == NULL) {
    return BSL_ERROR_CALLBACK_NOT_FOUND;
  }
  status = device_config_callback(config);
  if (BSL_SUCCESS != status) {
    return status;
  }
  status = hddl_set_device(dev_type);
  if (BSL_SUCCESS != status) {
    return status;
  }
  device_init_t device_init_callback = m_hddl_controller[dev_type].device_init;
  if (device_init_callback == NULL) {
    return BSL_ERROR_CALLBACK_NOT_FOUND;
  }
  return device_init_callback();
}

BSL_STATUS bsl_init_by_auto_scan() {
  BSL_STATUS status;
  int device_count = 0;

  printf("Performing auto-scan\n");
  for (int i = 0; i < BSL_DEVICE_TYPE_MAX; i++) {
    BslDeviceType device_type = cfg_get_device_type_pair(i)->device_type;
    if (!is_valid_device_type(device_type)) {
      continue;
    }
    status = m_hddl_controller[device_type].device_scan(&device_count);
    if (status != BSL_SUCCESS) {
      LOG("[%s %d] scan device type %d failed with %d\n", __func__, __LINE__, i, status);
      // hddl_get_error_string(status);
      continue;
    }

    if (device_count > 0) {
      LOG("Found %d devices\n", device_count);
      BslDeviceType dev_type = device_type;
      hddl_set_device(dev_type);
      status = m_hddl_controller[dev_type].device_init();
      return BSL_SUCCESS;
    }
  }
  LOG("No device found\n");

  return BSL_ERROR_NO_DEVICE_FOUND;
}

static BSL_STATUS hddl_bsl_destroy() {
  BslDeviceType dev_type = hddl_get_device();
  BSL_STATUS status = BSL_SUCCESS;
  if (is_valid_device_type(dev_type)) {
    status = m_hddl_controller[dev_type].device_destroy();
  }
  return status;
}

BSL_STATUS hddl_reset(int device_addr) {
  printf("Reset device address: %d with device type %d\n", device_addr, m_bsl_device);
  if (!is_valid_device_type(m_bsl_device)) {
    return BSL_ERROR_INVALID_DEVICE_TYPE;
  }

  device_reset_t reset_single_device_callback = m_hddl_controller[m_bsl_device].device_reset;
  if (reset_single_device_callback) {
    return reset_single_device_callback(device_addr);
  }
  return BSL_ERROR_CALLBACK_NOT_FOUND;
}

BSL_STATUS hddl_reset_all() {
  printf("Reset all devices with device type %d\n", m_bsl_device);

  if (!is_valid_device_type(m_bsl_device)) {
    return BSL_ERROR_INVALID_DEVICE_TYPE;
  }

  device_reset_all_t reset_all_callback = m_hddl_controller[m_bsl_device].device_reset_all;
  if (reset_all_callback) {
    return reset_all_callback();
  }
  return BSL_ERROR_CALLBACK_NOT_FOUND;
}

BSL_STATUS hddl_discard(int device_addr) {
  if (!is_valid_device_type(m_bsl_device)) {
    return BSL_ERROR_INVALID_DEVICE_TYPE;
  }

  printf("Discard device: %d\n", device_addr);
  device_discard_t device_discard_callback = m_hddl_controller[m_bsl_device].device_discard;
  if (device_discard_callback) {
    return device_discard_callback(device_addr);
  }
  return BSL_ERROR_CALLBACK_NOT_FOUND;
}

BslDeviceType hddl_get_device() {
  return m_bsl_device;
}

BSL_STATUS hddl_set_device(BslDeviceType bsl_device) {
  LOG("hddl_set_device bsl_device=%d\n", bsl_device);
  if (!is_valid_device_type(bsl_device)) {
    return BSL_ERROR_INVALID_DEVICE_TYPE;
  }
  int device_count = 0;
  BSL_STATUS status = m_hddl_controller[bsl_device].device_get_device_num(&device_count);
  if (0 == device_count || status != BSL_SUCCESS) {
    // if it is not 0, it should be scan already in hddl_bsl_init
    // if it is 0, it may not do a scan in hddl_bsl_init
    status = m_hddl_controller[bsl_device].device_scan(&device_count);
    if (status != BSL_SUCCESS) {
      return status;
    }
    device_init_t device_init_callback = m_hddl_controller[bsl_device].device_init;
    if (device_init_callback == NULL) {
      return BSL_ERROR_CALLBACK_NOT_FOUND;
    }
    status = device_init_callback();
  }
  if (status != BSL_SUCCESS) {
    return status;
  }
  if (0 == device_count) {
    return BSL_ERROR_NO_DEVICE_FOUND;
  }

  m_bsl_device = bsl_device;
  return BSL_SUCCESS;
}

bool is_valid_device_type(BslDeviceType bsl_device) {
  return (bsl_device >= BSL_DEVICE_TYPE_START) && (bsl_device < BSL_DEVICE_TYPE_MAX);
}
