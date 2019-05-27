/*
* Copyright 2018-2019 Intel Corporation.
* The source code, information and material ("Material") contained herein is
* owned by Intel Corporation or its suppliers or licensors, and title to such
* Material remains with Intel Corporation or its suppliers or licensors.
* The Material contains proprietary information of Intel or its suppliers and
* licensors. The Material is protected by worldwide copyright laws and treaty
* provisions.
* No part of the Material may be used, copied, reproduced, modified, published,
* uploaded, posted, transmitted, distributed or disclosed in any way without
* Intel's prior express written permission. No license under any patent,
* copyright or other intellectual property rights in the Material is granted to
* or conferred upon you, either expressly, by implication, inducement, estoppel
* or otherwise.
* Any license under such intellectual property rights must be express and
* approved by Intel in writing.
*/
#ifdef __cplusplus
extern "C" {
#endif

extern int usb_loglevel;

#define AUTO_VID                    0
#define AUTO_PID                    0
#define AUTO_UNBOOTED_PID           -1

#define DEFAULT_OPENVID				0x03E7
#define DEFAULT_OPENPID				0xf63b		// Once opened in VSC mode, VID/PID change

#define DEFAULT_UNBOOTVID			0x03E7
#define DEFAULT_UNBOOTPID_2485		0x2485
#define DEFAULT_UNBOOTPID_2150		0x2150


typedef enum usbBootError {
    USB_BOOT_SUCCESS = 0,
    USB_BOOT_ERROR,
    USB_BOOT_DEVICE_NOT_FOUND,
    USB_BOOT_TIMEOUT
} usbBootError_t;

#if (!defined(_WIN32) && !defined(_WIN64))
usbBootError_t usb_find_device_with_bcd(unsigned idx, char *addr, unsigned addrsize, void **device, int vid, int pid,unsigned short* bcdusb);
#else
usbBootError_t usb_find_device(unsigned idx, char *addr, unsigned addrsize, void **device, int vid, int pid);
#endif
int usb_boot(const char *addr, const void *mvcmd, unsigned size);


#ifdef __cplusplus
}
#endif
