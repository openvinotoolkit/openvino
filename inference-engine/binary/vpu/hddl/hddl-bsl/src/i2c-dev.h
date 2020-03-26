// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#ifndef __I2C_DEV_H__
#define __I2C_DEV_H__

#define HDDL_SMBUS_NAME ("SMBus I801 adapter")
#define MAX_I2C_DEVICE_NUM (4)
#define I2C_PORT_MAX_NAME_LEN (128)

#define I2C_INVALID_I2C_ADDR (0xFFFFFFFF)

BSL_STATUS i2c_write_byte(const char* port_name_or_id, int i2c_addr, int reg, int value);
BSL_STATUS i2c_read_byte(const char* port_name_or_id, int i2c_addr, int reg, int* value);
BSL_STATUS i2c_fetch_address_by_scan(const char* port_name_or_id,
                                     int start_addr,
                                     int end_addr,
                                     int* dev_addr,
                                     int* p_device_count);

#endif
