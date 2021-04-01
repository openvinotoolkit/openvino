// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Common USB API

#ifndef _USB_COMMON_H
#define _USB_COMMON_H

//#include <stdlib.h>
//#include <stdint.h>
//#include <stdio.h>

#ifndef _USB_SOURCE
typedef void *usb_dev;
typedef void *usb_han;
#endif

enum
{
    LIBUSB_ERROR_NO_DEVICE = -4,
    LIBUSB_ERROR_TIMEOUT = -7
};


typedef void libusb_device;
typedef struct _usb_han *usb_hwnd;
typedef struct _usb_han libusb_device_handle;
extern int usb_init(void);
extern void usb_shutdown(void);

extern int usb_can_find_by_guid(void);
extern int usb_list_devices(uint16_t vid, uint16_t pid, uint8_t dev_des[][2 + 2 + 4 * 7 + 7]);
extern void *  enumerate_usb_device(uint16_t vid, uint16_t pid, const char *addr, int loud);
extern void *  usb_find_device_by_guid(int loud);
extern int usb_check_connected(usb_dev dev);
extern void * usb_open_device(usb_dev dev, uint8_t *ep, uint8_t intfaceno, char *err_string_buff, size_t err_max_len);
extern uint8_t usb_get_bulk_endpoint(usb_hwnd han, int dir);
extern size_t usb_get_endpoint_size(usb_hwnd han, uint8_t ep);
extern int usb_bulk_write(usb_hwnd han, uint8_t ep, const void *buffer, size_t sz, uint32_t *wrote_bytes, int timeout_ms);
extern int usb_bulk_read(usb_hwnd han, uint8_t ep, void *buffer, size_t sz, uint32_t *read_bytes, int timeout_ms);
extern void usb_free_device(usb_dev dev);
extern void usb_close_device(usb_hwnd han);

extern const char *usb_last_bulk_errmsg(void);
extern void usb_set_msgfile(FILE *file);
extern void usb_set_verbose(int value);
extern void usb_set_ignoreerrors(int value);

extern const char* libusb_strerror(int x);

#endif//_USB_COMMON_H
