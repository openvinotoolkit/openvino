// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <assert.h>
#include <stdio.h>

#include "hddl-bsl.h"
#include "hddl_bsl_priv.h"
#include "hddl_bsl_thread.h"
#include "i2c-dev.h"

/*
1. By default reset pin is pulled high by 1.8v, ioexpander and reset jumper
(2 pins on board) can pull low the reset pin.
2. To reset the Myriad X, ioexpander or reset jumper pull low the reset pin
And then release the pull low, 1.8v pull high the reset pin.
3. For ioexpander
set 0x02 Reg to 00, set ioexpander as a drive low source
Use 0x06 Reg to enable/disable ioexpander output drive low
*/

#define IOEXPANDER_DEFAULT_I2C_PORT_NAME HDDL_SMBUS_NAME
// if port id is valid, use port id, if it is invalid, use the default smbus name
static char m_i2c_port_or_name[I2C_PORT_MAX_NAME_LEN];

static int m_ioexpander_address_list[MAX_I2C_DEVICE_NUM];
static int m_ioexpander_start_addr = 0x20;
static int m_ioexpander_end_addr = 0x27;
static unsigned char m_ioexpander_count = 0;

static bsl_mutex_t m_mutex;

static int m_ioexpander_address_valid(int address) {
  if (address > m_ioexpander_end_addr || address < m_ioexpander_start_addr) {
    return 0;
  }
  return 1;
}

// config all pin as output
BSL_STATUS m_ioexpander_init() {
  int ret = 0;

  // version 1
  //# Configure PORT0 as output
  // sudo i2cset -y 5 0x20 0x06 0x00 b
  // registers 6 and 7) shown in Table 7 configure the directions
  // of the I/O pins. If a bit in
  // this register is set to 1, the corresponding port pin is
  // enabled as an input with a high-impedance output driver. If
  // a bit in this register is cleared to 0, the corresponding
  // port pin is enabled as an output.

  // version 2
  // configure as input(default is input, no need config), the 1.8v
  // always pull high all myx reset pin
  // config all reset pins as low, since it is input,it is ok
  static int initialized = 0;
  if (initialized == 0) {
    int i = 0;
    bsl_mutex_init(&m_mutex);

    for (i = 0; i < m_ioexpander_count; i++) {
      int i2c_addr = m_ioexpander_address_list[i];
      LOG("_ioexpander_pre i2c_addr is %x\n", i2c_addr);
      ret = i2c_write_byte(m_i2c_port_or_name, i2c_addr, 0x02, 0x00);
    }
    initialized = 1;
  }
  return ret;
}

static BSL_STATUS m_ioexpander_destroy() {
  bsl_mutex_destroy(&m_mutex);
  return BSL_SUCCESS;
}

// high 3 bits is baord id
// low 5 bits is real device id
static void m_ioexpander_parse_device_id(int device_id, int* i2c_addr, int* real_device_id) {
  char data = *((char*)&device_id);
  int board_id = (int)((data & 0xE0) >> 5);
  *real_device_id = (int)(data & 0x1F);
  *i2c_addr = 0x20 + board_id;
}

// only need config as output
static BSL_STATUS m_ioexpander_reset(int device_id) {
  int i2c_addr = I2C_INVALID_I2C_ADDR;
  int value = 0;
  int real_device_id = 0;

  m_ioexpander_parse_device_id(device_id, &i2c_addr, &real_device_id);
  bsl_mutex_lock(&m_mutex);
  BSL_STATUS status = i2c_read_byte(m_i2c_port_or_name, i2c_addr, 0x06, &value);
  if (status) {
    bsl_mutex_unlock(&m_mutex);
    return status;
  }

  LOG("_ioexpander_reset origin value is %d\n", value);

  int reset_value = 1 << real_device_id;

  int down_value = value & (~reset_value);
  LOG("_ioexpander_reset down_value is %d\n", down_value);
  // config as output , enable pull low by ioexpander,
  // the pin value is already 0
  status = i2c_write_byte(m_i2c_port_or_name, i2c_addr, 0x06, down_value);
  usleep(20000);
  int up_value = value | reset_value;
  LOG("_ioexpander_reset up_value is %d\n", up_value);
  // config as input,1.8v works
  status = i2c_write_byte(m_i2c_port_or_name, i2c_addr, 0x06, up_value);
  bsl_mutex_unlock(&m_mutex);
  return status;
}

static BSL_STATUS m_ioexpander_discard(int device_id) {
  int i2c_addr = I2C_INVALID_I2C_ADDR;
  int value = 0;
  int real_device_id = 0;
  m_ioexpander_parse_device_id(device_id, &i2c_addr, &real_device_id);
  bsl_mutex_lock(&m_mutex);
  BSL_STATUS status = i2c_read_byte(m_i2c_port_or_name, i2c_addr, 0x06, &value);
  if (status) {
    bsl_mutex_unlock(&m_mutex);
    return status;
  }
  LOG("_ioexpander_discard origin value is %d\n", value);

  int reset_value = 1 << real_device_id;

  int down_value = value & (~reset_value);
  LOG("_ioexpander_discard down_value is %d\n", down_value);
  // only config it as input, since the pin data is ready 0
  status = i2c_write_byte(m_i2c_port_or_name, i2c_addr, 0x06, down_value);
  usleep(20000);
  bsl_mutex_unlock(&m_mutex);
  return status;
}

static BSL_STATUS m_ioexpander_reset_all() {
  // flip-flop controlling the output selection
  BSL_STATUS status = BSL_SUCCESS;

  if (m_ioexpander_count == 0) {
    return BSL_ERROR_NO_IOEXPANDER_DEVICE_FOUND;
  }

  for (int i = 0; i < m_ioexpander_count; i++) {
    int i2c_addr = m_ioexpander_address_list[i];
    LOG("_ioexpander_reset_all i2c_addr is %x\n", i2c_addr);
    if (i2c_addr == I2C_INVALID_I2C_ADDR)
      break;
    bsl_mutex_lock(&m_mutex);
    for (int j = 0; j < 8; j++) {
      int value = (0x01 << j);
      value = ~value;
      value = value & 0x00FF;
      // int down_value = value & (~reset_value);
      LOG("_ioexpander_reset down_value is %d\n", j);
      usleep(20000);
      // config as input,the pin data is 0, so equal pull low it
      status = i2c_write_byte(m_i2c_port_or_name, i2c_addr, 0x06, (int)value);
      usleep(20000);
      // config as output, 1.8 works
      status = i2c_write_byte(m_i2c_port_or_name, i2c_addr, 0x06, 0xFF);
    }
    bsl_mutex_unlock(&m_mutex);
  }
  return status;
}

static BSL_STATUS m_ioexpander_add_slave_address(int slave_addr) {
  if (m_ioexpander_count >= MAX_I2C_DEVICE_NUM) {
    return -1;
  }
  if (!m_ioexpander_address_valid(slave_addr)) {
    return -1;
  }

  m_ioexpander_address_list[m_ioexpander_count] = slave_addr;
  m_ioexpander_count++;
  return 0;
}

static BSL_STATUS m_ioexpander_config(struct json_object* config) {
  BSL_STATUS status = BSL_SUCCESS;
  struct json_object* jport = json_object_object_get(config, "i2c_port");

  if (jport) {
    const char* jport_string = json_object_get_string(jport);
    if (jport_string) {
      bsl_strncpy(m_i2c_port_or_name, sizeof(m_i2c_port_or_name), jport_string, I2C_PORT_MAX_NAME_LEN - 1);
    }
  }
  if (strlen(m_i2c_port_or_name) == 0) {
    bsl_strncpy(m_i2c_port_or_name, sizeof(m_i2c_port_or_name), IOEXPANDER_DEFAULT_I2C_PORT_NAME,
                I2C_PORT_MAX_NAME_LEN);
  }
  struct json_object* jaddress = json_object_object_get(config, "i2c_addr");
  if (jaddress == NULL) {
    return BSL_ERROR_CFG_PARSING_GET_NULL_OBJECT;
  }
  size_t addr_len = json_object_array_length(jaddress);
  if (addr_len == 0) {
    return BSL_ERROR_INVALID_CFG_FILE;
  }
  unsigned int j = 0;
  if (addr_len > MAX_I2C_DEVICE_NUM) {
    addr_len = MAX_I2C_DEVICE_NUM;
  }

  for (j = 0; j < addr_len; j++) {
    struct json_object* addr_obj = json_object_array_get_idx(jaddress, j);
    int value = json_object_get_int(addr_obj);
    status = m_ioexpander_add_slave_address(value);
    if (BSL_SUCCESS != status) {
      LOG("%d is invalid address in bsl.json file", value);
    }
  }

  return status;
}

static BSL_STATUS m_ioexpander_scan(int* p_device_count) {
  int address_list[MAX_I2C_DEVICE_NUM];
  *p_device_count = 0;
  printf("scan ioexpander device...\n");

  errno_t osl_status = bsl_strncpy(m_i2c_port_or_name, sizeof(m_i2c_port_or_name), IOEXPANDER_DEFAULT_I2C_PORT_NAME,
                                   I2C_PORT_MAX_NAME_LEN);
  if (osl_status) {
    return BSL_ERROR_STRNCPY;
  }

  BSL_STATUS status = i2c_fetch_address_by_scan(m_i2c_port_or_name, m_ioexpander_start_addr, m_ioexpander_end_addr,
                                                address_list, p_device_count);
  if (status < 0) {
    printf("found 0 ioexpander device\n");
    return status;
  }

  m_ioexpander_count = *p_device_count;
  if (*p_device_count > 0) {
    memcpy((void*)m_ioexpander_address_list, (const void*)address_list, sizeof(int) * m_ioexpander_count);
  }
  printf("found %d ioexpander device\n", m_ioexpander_count);
  return BSL_SUCCESS;
}

static BSL_STATUS m_ioexpander_get_device_num(int* p_device_count) {
  if (!p_device_count)
    return BSL_ERROR_MEMORY_ERROR;
  *p_device_count = m_ioexpander_count;
  return BSL_SUCCESS;
}

void ioexpander_init(HddlController_t* ctrl) {
  ctrl->device_init = m_ioexpander_init;
  ctrl->device_reset = m_ioexpander_reset;
  ctrl->device_reset_all = m_ioexpander_reset_all;
  ctrl->device_config = m_ioexpander_config;
  ctrl->device_add = m_ioexpander_add_slave_address;
  ctrl->device_scan = m_ioexpander_scan;
  ctrl->device_get_device_num = m_ioexpander_get_device_num;
  ctrl->device_discard = m_ioexpander_discard;
  ctrl->device_destroy = m_ioexpander_destroy;
}
