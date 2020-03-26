// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "hddl-bsl.h"
#include "hddl_bsl_priv.h"
#include "i2c-dev.h"
#include "i2c_linux.h"
#include "i2cbusses.h"

#define MODE_AUTO 0
#define MODE_QUICK 1
#define MODE_READ 2
#define MODE_FUNC 3

static int _check_funcs(int file, int size, int pec) {
  unsigned long funcs;

  if (ioctl(file, FUNCTIONS, &funcs) < 0) {
    fprintf(stderr,
            "Error: Could not get the adapter "
            "functionality matrix: %s\n",
            strerror(errno));
    return -1;
  }

  switch (size) {
    case I2C_SMBUS_BYTE_DATA:
      if (!(funcs & FUNCTION_SMBUS_WRITE_BYTE_DATA)) {
        fprintf(stderr, MISSING_FUNC_FMT, "SMBus write byte");
        return -1;
      }
      break;
  }

  if (pec && !(funcs & (FUNCTION_SMBUS_PEC | FUNCTION_I2C))) {
    fprintf(stderr,
            "Warning: Adapter does "
            "not seem to support PEC\n");
  }

  return 0;
}

static BSL_STATUS i2c_file_open(const char* port_name_or_id, int* p_file) {
  int i2c_bus;
  BSL_STATUS status = i2c_bus_lookup(port_name_or_id, &i2c_bus);
  if (status < 0) {
    return status;
  }

  *p_file = i2c_dev_open(i2c_bus, /*filename, sizeof(filename), */ 0);
  return BSL_SUCCESS;
}

static BSL_STATUS i2c_check_funcs(int file, int i2c_addr) {
  unsigned long funcs;

  if (_check_funcs(file, I2C_SMBUS_BYTE_DATA, 0) || slave_addr_set(file, i2c_addr, 1)) {
    return BSL_ERROR_I2C_CHECK_FUNC_FIRST_ERROR;
  }

  if (ioctl(file, FUNCTIONS, &funcs) < 0) {
    return BSL_ERROR_I2C_CHECK_FUNC_SECOND_ERROR;
  }

  return BSL_SUCCESS;
}

static int i2c_file_close(int file) {
  close(file);
  return 0;
}

BSL_STATUS i2c_fetch_address_by_scan(const char* port_name_or_id,
                                     int start_addr,
                                     int end_addr,
                                     int* dev_addr,
                                     int* p_device_count) {
  int found_device_count = 0;
  int mode = MODE_AUTO;

  int file;
  BSL_STATUS status = i2c_file_open(port_name_or_id, &file);
  if (status < 0) {
    return status;
  }

  for (int index = start_addr; index <= end_addr; index++) {
    // Set slave address
    if (ioctl(file, I2C_SLAVE, index) < 0) {
      if (errno == EBUSY) {
        continue;
      } else {
        status = BSL_ERROR_IOCTL_FAIL;
        goto i2c_fetch_address_by_scan_cleanup_and_exit;
      }
    }

    int res;
    // Probe this address
    switch (mode) {
      case MODE_QUICK:
        /* This is known to corrupt the Atmel AT24RF08
           EEPROM */
        res = __i2c_smbus_write_quick(file, I2C_SMBUS_WRITE);
        break;
      case MODE_READ:
        /* This is known to lock SMBus on various
           write-only chips (mainly clock chips) */
        res = __i2c_smbus_read_byte(file);
        break;
      default:
        if ((index >= 0x30 && index <= 0x37) || (index >= 0x50 && index <= 0x5F))
          res = __i2c_smbus_read_byte(file);
        else
          res = __i2c_smbus_write_quick(file, I2C_SMBUS_WRITE);
    }
    if (res < 0) {
      continue;
    }

    if (found_device_count < MAX_I2C_DEVICE_NUM) {
      dev_addr[found_device_count] = index;
      found_device_count++;
    }
    if (found_device_count >= MAX_I2C_DEVICE_NUM)
      break;
  }

i2c_fetch_address_by_scan_cleanup_and_exit:
  *p_device_count = found_device_count;
  i2c_file_close(file);
  return status;
}

int i2c_write_byte(const char* port_name_or_id, int i2c_addr, int reg, int value) {
  int file;
  BSL_STATUS status = i2c_file_open(port_name_or_id, &file);
  if (status < 0) {
    return status;
  }

  status = i2c_check_funcs(file, i2c_addr);
  if (status) {
    goto i2c_write_byte_exit;
  }

  int res = __i2c_smbus_write_byte_data(file, reg, value);
  status = res;

i2c_write_byte_exit:
  close(file);
  return status;
}

int i2c_read_byte(const char* port_name_or_id, int i2c_addr, int reg, int* value) {
  int file;
  BSL_STATUS status = i2c_file_open(port_name_or_id, &file);
  if (status < 0) {
    return status;
  }

  status = i2c_check_funcs(file, i2c_addr);
  if (status) {
    goto i2c_read_byte_exit;
  }

  int res = __i2c_smbus_read_byte_data(file, reg);
  if (res >= 0) {
    *value = res;
    res = 0;
  }
  status = res;

i2c_read_byte_exit:
  close(file);
  return status;
}
