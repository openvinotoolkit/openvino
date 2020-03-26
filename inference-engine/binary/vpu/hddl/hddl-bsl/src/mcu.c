// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <assert.h>
#include <stdio.h>

#include "hddl-bsl.h"
#include "hddl_bsl_priv.h"
#include "i2c-dev.h"

// mcu only support one pcie card, only smbus, the name prefix is fix
#define MCU_DEFAULT_I2C_PORT_NAME HDDL_SMBUS_NAME
// if port id is valid, use port id, if it is invalid, use the default smbus name
static char m_i2c_port_or_name[I2C_PORT_MAX_NAME_LEN];

static int m_mcu_address = I2C_INVALID_I2C_ADDR;  // Only support one
static int m_mcu_start_addr = 0x18;
static int m_mcu_end_addr = 0x1F;
static int m_mcu_dev_nums = 0;

static BSL_STATUS m_mcu_init() {
  return BSL_SUCCESS;
}
static BSL_STATUS m_mcu_destroy() {
  return BSL_SUCCESS;
}

static void m_mcu_parse_device_id(int device_id, int* real_device_id) {
  char data = *((char*)&device_id);
  *real_device_id = data & 0x07;
}

// only support one mcu
static BSL_STATUS m_mcu_reset(int device_id) {
  int real_device_id = 0;
  m_mcu_parse_device_id(device_id, &real_device_id);
  int reset_value = 1 << real_device_id;
  LOG("_mcu_reset reset_value value is %d\n", reset_value);
  if (m_mcu_address == I2C_INVALID_I2C_ADDR)
    return BSL_ERROR_I2C_INVALID_ADDRESS;
  return i2c_write_byte(m_i2c_port_or_name, m_mcu_address, 0x01, reset_value);
}

static BSL_STATUS m_mcu_discard(int device_id) {
  UNUSED(device_id);
  printf("Device Discard for mcu is not supported\n");
  return BSL_ERROR_UNSUPPORTED_FUNCTION;
}

static BSL_STATUS m_mcu_reset_all() {
  return i2c_write_byte(m_i2c_port_or_name, m_mcu_address, 0x01, 0xFF);
}

static int m_mcu_address_valid(int address) {
  if (address > m_mcu_end_addr || address < m_mcu_start_addr) {
    return 0;
  }
  return 1;
}

static BSL_STATUS m_mcu_set_address(int slave_addr) {
  if (m_mcu_address_valid(slave_addr)) {
    m_mcu_address = slave_addr;
    return 0;
  }
  return -1;
}

static BSL_STATUS m_mcu_config(struct json_object* config) {
  BSL_STATUS status = BSL_SUCCESS;
  struct json_object* jport = json_object_object_get(config, "i2c_port");

  if (jport) {
    const char* jport_string = json_object_get_string(jport);
    if (jport_string) {
      bsl_strncpy(m_i2c_port_or_name, sizeof(m_i2c_port_or_name), jport_string, I2C_PORT_MAX_NAME_LEN - 1);
    }
  }
  if (strlen(m_i2c_port_or_name) == 0) {
    bsl_strncpy(m_i2c_port_or_name, sizeof(m_i2c_port_or_name), MCU_DEFAULT_I2C_PORT_NAME, I2C_PORT_MAX_NAME_LEN);
  }

  struct json_object* jaddress = json_object_object_get(config, "i2c_addr");
  if (!jaddress) {
    return BSL_ERROR_CFG_PARSING_GET_NULL_OBJECT;
  }

  size_t addr_len = json_object_array_length(jaddress);
  if (addr_len == 0) {
    return BSL_ERROR_INVALID_CFG_FILE;
  }

  if (addr_len > 1) {
    LOG("MCU only support 1 device");
    addr_len = 1;
  }

  unsigned int j = 0;
  for (j = 0; j < addr_len; j++) {
    struct json_object* addr_obj = json_object_array_get_idx(jaddress, j);
    int value = json_object_get_int(addr_obj);
    status = m_mcu_set_address(value);
    if (BSL_SUCCESS != status) {
      LOG("%d is invalid address in bsl.json file", value);
    }
  }

  return status;
}

static BSL_STATUS m_mcu_scan(int* p_device_count) {
  int address_list[MAX_I2C_DEVICE_NUM];
  *p_device_count = 0;
  printf("scan mcu device...\n");
  errno_t osl_status =
      bsl_strncpy(m_i2c_port_or_name, sizeof(m_i2c_port_or_name), MCU_DEFAULT_I2C_PORT_NAME, I2C_PORT_MAX_NAME_LEN);
  if (osl_status) {
    return BSL_ERROR_STRNCPY;
  }

  BSL_STATUS status =
      i2c_fetch_address_by_scan(m_i2c_port_or_name, m_mcu_start_addr, m_mcu_end_addr, address_list, p_device_count);
  if (status < 0) {
    printf("found 0 mcu device\n");
    return status;
  }

  m_mcu_dev_nums = *p_device_count;
  if (m_mcu_dev_nums > 0) {
    m_mcu_address = address_list[0];
    m_mcu_dev_nums = 1;
    *p_device_count = 1;
  }
  printf("found %d mcu device\n", m_mcu_dev_nums);
  return BSL_SUCCESS;
}

static BSL_STATUS m_mcu_get_device_num(int* p_device_count) {
  if (!p_device_count)
    return BSL_ERROR_MEMORY_ERROR;
  *p_device_count = m_mcu_dev_nums;
  return BSL_SUCCESS;
}

void mcu_init(HddlController_t* ctrl) {
  ctrl->device_init = m_mcu_init;
  ctrl->device_reset = m_mcu_reset;
  ctrl->device_reset_all = m_mcu_reset_all;
  ctrl->device_config = m_mcu_config;
  ctrl->device_add = m_mcu_set_address;
  ctrl->device_scan = m_mcu_scan;
  ctrl->device_get_device_num = m_mcu_get_device_num;
  ctrl->device_discard = m_mcu_discard;
  ctrl->device_destroy = m_mcu_destroy;
}
