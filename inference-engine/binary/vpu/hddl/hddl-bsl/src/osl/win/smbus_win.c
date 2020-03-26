// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Windows.h>

#include <WinIoCtl.h>

#include "hddl-bsl.h"
#include "i2c-dev.h"

#define FILE_DEVICE_HDDLI2C 0x0116
#define IOCTL_HDDLI2C_REGISTER_READ CTL_CODE(FILE_DEVICE_HDDLI2C, 0x700, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define IOCTL_HDDLI2C_REGISTER_WRITE CTL_CODE(FILE_DEVICE_HDDLI2C, 0x701, METHOD_BUFFERED, FILE_ANY_ACCESS)

#define SPBHDDL_I2C_NAME L"HDDLSMBUS"

#define SPBHDDL_I2C_SYMBOLIC_NAME L"\\DosDevices\\" SPBHDDL_I2C_NAME
#define SPBHDDL_I2C_USERMODE_PATH L"\\\\.\\" SPBHDDL_I2C_NAME
#define SPBHDDL_I2C_USERMODE_PATH_SIZE sizeof(SPBHDDL_I2C_USERMODE_PATH)

BSL_STATUS i2c_fetch_address_by_scan(const char* port_name_or_id,
                                     int start_addr,
                                     int end_addr,
                                     int* dev_addr,
                                     int* p_device_count) {
  int found_device_count = 0;
  DWORD dwOutput;
  UCHAR outBuff = 0;
  UCHAR data[3] = {0x1f, 0x00, 0xff};  // Address, Register, data

  HANDLE hDevice = CreateFileW(SPBHDDL_I2C_USERMODE_PATH, GENERIC_READ | GENERIC_WRITE,
                               FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL);
  if (hDevice == INVALID_HANDLE_VALUE) {
    return BSL_ERROR_I2C_BUS_CreateFileW;
  }

  for (int i = start_addr; i <= end_addr; i++) {
    /* Set slave address */
    data[0] = i;
    BOOL ioresult = DeviceIoControl(hDevice, IOCTL_HDDLI2C_REGISTER_READ, data, sizeof(data), &outBuff, sizeof(outBuff),
                                    &dwOutput, NULL);

    if (ioresult) {
      if (found_device_count < MAX_I2C_DEVICE_NUM) {
        dev_addr[found_device_count] = i;
        found_device_count++;
      }
      if (found_device_count >= MAX_I2C_DEVICE_NUM)
        break;
    } else {
      DWORD error_code = GetLastError();
      if (error_code == 55)
        continue;
      else {
        break;
      }
    }
  }
  CloseHandle(hDevice);
  *p_device_count = found_device_count;
  return BSL_SUCCESS;
}

BSL_STATUS i2c_write_byte(const char* port_name_or_id, int i2c_addr, int reg, int value) {
  DWORD dwOutput;
  UCHAR data[3] = {0x1f, 0x01, 0xff};  // Address, Register, data

  data[0] = (UCHAR)(i2c_addr);
  data[1] = (UCHAR)(reg);
  data[2] = (UCHAR)(value & 0xFF);

  HANDLE hDevice = CreateFileW(SPBHDDL_I2C_USERMODE_PATH, GENERIC_READ | GENERIC_WRITE,
                               FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL);
  if (hDevice == INVALID_HANDLE_VALUE) {
    return BSL_ERROR_I2C_BUS_CreateFileW;
  }
  BOOL ioresult;
  ioresult = DeviceIoControl(hDevice, IOCTL_HDDLI2C_REGISTER_WRITE, data, sizeof(data), NULL, 0, &dwOutput, NULL);
  CloseHandle(hDevice);
  if (ioresult)
    return BSL_SUCCESS;
  else
    return BSL_ERROR_I2C_BUS_WRITE_FAIL;
}

BSL_STATUS i2c_read_byte(const char* port_name_or_id, int i2c_addr, int reg, int* value) {
  DWORD dwOutput;
  UCHAR outBuff = 0;
  UCHAR data[3] = {0x1f, 0x01, 0xff};  // Address, Register, data

  data[0] = (UCHAR)(i2c_addr);
  data[1] = (UCHAR)(reg);

  HANDLE hDevice = CreateFileW(SPBHDDL_I2C_USERMODE_PATH, GENERIC_READ | GENERIC_WRITE,
                               FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL);
  if (hDevice == INVALID_HANDLE_VALUE) {
    return BSL_ERROR_I2C_BUS_CreateFileW;
  }
  BOOL ioresult;
  ioresult = DeviceIoControl(hDevice, IOCTL_HDDLI2C_REGISTER_READ, data, sizeof(data), &outBuff, sizeof(outBuff),
                             &dwOutput, NULL);
  CloseHandle(hDevice);
  if (ioresult) {
    *value = outBuff;
    return BSL_SUCCESS;
  } else
    return BSL_ERROR_I2C_BUS_READ_FAIL;
}
