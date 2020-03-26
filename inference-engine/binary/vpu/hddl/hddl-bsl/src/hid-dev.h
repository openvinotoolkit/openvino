// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#ifndef __HDDL_BSL_HID_DEV_H__
#define __HDDL_BSL_HID_DEV_H__

#include <wchar.h>

#define MAX_HID_DEVICE_NUM (16)
#define MAX_HID_PARENT_NODE_NUM (16)
#define MAX_HID_USB_NODE_STRING_LEN (15)

struct hidapi_device_info_s {
  char* path;
  unsigned short vendor_id;
  unsigned short product_id;
  wchar_t* serial_number;
  unsigned short release_number;
  wchar_t* manufacturer_string;
  wchar_t* product_string;
  unsigned short usage_page;
  unsigned short usage;
  int interface_number;
  struct hidapi_device_info_s* next;
};

struct hidapi_device_internal;
typedef struct hidapi_device_internal hidapi_device_s;

struct hidapi_device_info_s* hidapi_enumerate(unsigned short vendor_id,
                                              unsigned short product_id,
                                              const char* path_array[],
                                              size_t path_array_len);

void hidapi_free_enumeration(struct hidapi_device_info_s* devs);

hidapi_device_s* hidapi_open(unsigned short vendor_id, unsigned short product_id, const wchar_t* serial_number);

void hidapi_close(hidapi_device_s* device);

hidapi_device_s* hidapi_open_path(const char* path);

int hidapi_get_fd(hidapi_device_s* device);

int hidapi_write(hidapi_device_s* device, const unsigned char* data, size_t length);

int hidapi_read_timeout(hidapi_device_s* dev, unsigned char* data, size_t length, int milliseconds);

int hidapi_read(hidapi_device_s* device, unsigned char* data, size_t length);

int hidapi_set_nonblocking(hidapi_device_s* device, int nonblock);

#endif
