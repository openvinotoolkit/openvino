// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef __cplusplus
extern "C" {
#endif

extern int usb_loglevel;

#define AUTO_VID                    0
#define AUTO_PID                    0
#define AUTO_UNBOOTED_PID           -1

#define DEFAULT_OPENVID             0x03E7
#ifdef ALTERNATE_PID
    #define DEFAULT_OPENPID             0xf63c      // Once opened in VSC mode, VID/PID change
#else
    #define DEFAULT_OPENPID             0xf63b     // Once opened in VSC mode, VID/PID change
#endif
#define DEFAULT_UNBOOTVID           0x03E7
#define DEFAULT_UNBOOTPID_2485      0x2485
#define DEFAULT_UNBOOTPID_2150      0x2150
#define DEFAULT_CHUNKSZ             1024*1024


typedef enum usbBootError {
    USB_BOOT_SUCCESS = 0,
    USB_BOOT_ERROR,
    USB_BOOT_DEVICE_NOT_FOUND,
    USB_BOOT_TIMEOUT
} usbBootError_t;

#if (!defined(_WIN32) && !defined(_WIN64))
usbBootError_t usb_find_device_with_bcd(unsigned idx, char *input_addr,
                                        unsigned addrsize, void **device, int vid, int pid,unsigned short* bcdusb);
#else
usbBootError_t usb_find_device(unsigned idx, char *addr, unsigned addrsize,
   void **device, int vid, int pid);
void initialize_usb_boot();
#endif
int usb_boot(const char *addr, const void *mvcmd, unsigned size);
int get_pid_by_name(const char* name);


#ifdef __cplusplus
}
#endif
