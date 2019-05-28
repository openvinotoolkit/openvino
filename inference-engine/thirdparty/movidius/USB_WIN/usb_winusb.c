// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma comment(lib, "winusb.lib")
#pragma comment(lib, "setupapi.lib")
//#define _CRT_SECURE_NO_WARNINGS

#define INITGUID
#include <Windows.h>
#include <winusb.h>
#include <Usbiodef.h>
#include <SetupAPI.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "usb_winusb.h"

#define USB_DIR_OUT		0
#define USB_DIR_IN		1

#define USB_DEV_NONE	NULL
#define USB_HAN_NONE	NULL

#define USB_ERR_NONE		0
#define USB_ERR_TIMEOUT		-1
#define USB_ERR_FAILED		-2
#define USB_ERR_INVALID		-3



///*
struct ep_info {
	uint8_t ep;
	size_t sz;
	ULONG last_timeout;
};
struct _usb_han {
	HANDLE devHan;
	WINUSB_INTERFACE_HANDLE winUsbHan;
	struct ep_info eps[2];
};

extern const char * usb_get_pid_name(int);

#if defined(_MSC_VER) && _MSC_VER < 1900
#define snprintf _snprintf
#endif

// Myriad 2: {19E08104-0543-40A5-B107-87EF463DCEF1}
DEFINE_GUID(GUID_DEVINTERFACE_Myriad2, 0x19e08104, 0x0543, 0x40a5,
	0xb1, 0x07, 0x87, 0xef, 0x46, 0x3d, 0xce, 0xf1);

// Myriad X: {504E1220-E189-413A-BDEC-ECFFAF3A3731}
DEFINE_GUID(GUID_DEVINTERFACE_MyriadX, 0x504e1220, 0xe189, 0x413a,
	0xbd, 0xec, 0xec, 0xff, 0xaf, 0x3a, 0x37, 0x31);

static FILE *msgfile = NULL;
static int verbose = 0, ignore_errors = 0;
static DWORD last_bulk_errcode = 0;
static char *errmsg_buff = NULL;
static size_t errmsg_buff_len = 0;

static const char *format_win32_msg(DWORD errId) {
	while(!FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
		NULL, errId, 0, errmsg_buff, (DWORD)errmsg_buff_len, NULL)) {
		if(GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
err_fail:
			snprintf(errmsg_buff, errmsg_buff_len, "Win32 Error 0x%08lx (Unable to retrieve error message)", errId);
			return errmsg_buff;
		}
		size_t nlen = errmsg_buff_len + (errmsg_buff_len / 2);
		if(nlen > 1024)
			goto err_fail;
		char *nbuff = realloc(errmsg_buff, nlen);
		if(nbuff == NULL)
			goto err_fail;
		errmsg_buff = nbuff;
		errmsg_buff_len = nlen;
	}
	return errmsg_buff;
}

static void wperror(const char *errmsg) {
	DWORD errId = GetLastError();
	fprintf(stderr, "%s: System err %d\n", errmsg, errId);
}

static void wstrerror(char *buff, const char *errmsg) {
	DWORD errId = GetLastError();
	snprintf(buff,strlen(buff), "%s: %s\n", errmsg, format_win32_msg(errId));
}
const char* libusb_strerror(int x)
{
	return format_win32_msg(x);
}
int usb_init(void) {
	msgfile = stdout;
	if(errmsg_buff == NULL) {
		errmsg_buff_len = 64;
		errmsg_buff = malloc(errmsg_buff_len);
		if(errmsg_buff == NULL) {
			perror("malloc");
			return -1;
		}
	}
	return 0;
}

void usb_shutdown(void) {
	if(errmsg_buff != NULL) {
		free(errmsg_buff);
		errmsg_buff = NULL;
	}
}

int usb_can_find_by_guid(void) {
	return 1;
}

static usb_dev retreive_dev_path(HDEVINFO devInfo, SP_DEVICE_INTERFACE_DATA *ifaceData) {
	usb_dev res;
	PSP_DEVICE_INTERFACE_DETAIL_DATA detData;
	ULONG len, reqLen;

	if(!SetupDiGetDeviceInterfaceDetail(devInfo, ifaceData, NULL, 0, &reqLen, NULL)) {
		if(GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
			wperror("SetupDiEnumDeviceInterfaces");
			SetupDiDestroyDeviceInfoList(devInfo);
			return USB_DEV_NONE;
		}
	}
	detData = (PSP_DEVICE_INTERFACE_DETAIL_DATA)_alloca(reqLen);
	detData->cbSize = sizeof(*detData);
	len = reqLen;
	if(!SetupDiGetDeviceInterfaceDetail(devInfo, ifaceData, detData, len, &reqLen, NULL)) {
		wperror("SetupDiGetDeviceInterfaceDetail");
		SetupDiDestroyDeviceInfoList(devInfo);
		return USB_DEV_NONE;
	}
	res = _strdup(detData->DevicePath);
	if(res == NULL) {
		perror("strdup");
	}
	SetupDiDestroyDeviceInfoList(devInfo);
	return res;
}

static const char *gen_addr(HDEVINFO devInfo, SP_DEVINFO_DATA *devInfoData, uint16_t pid) {
    static char buff[16];
    char li_buff[128];
    unsigned int port, hub;
    if (!SetupDiGetDeviceRegistryProperty(devInfo, devInfoData, SPDRP_LOCATION_INFORMATION, NULL, li_buff, sizeof(li_buff), NULL))
    {
        goto ret_err;
    }
	if(sscanf(li_buff, "Port_#%u.Hub_#%u", &port, &hub) != 2)
        goto ret_err;

	//matching it to libusboutput
	const char* dev_name = usb_get_pid_name(pid);
	if(dev_name == NULL)
		goto ret_err;

	snprintf(buff, sizeof(buff), "%u.%u-%s", hub, port, dev_name);
    buff[sizeof(buff) - 1] = '\0';
	return buff;
ret_err:
    return "<error>";
}
extern DEFAULT_OPENPID;

static int compareDeviceByHubAndPort(const void *l, const void *r) {
    int lHub = 0, lPort = 0;
    int rHub = 0, rPort = 0;

    if (sscanf(((const char *)l + 4), "%d.%d", &lHub, &lPort) == EOF) {
        perror("Can not parse hub and port of the devices");
    };
    if (sscanf(((const char *)r + 4), "%d.%d", &rHub, &rPort) == EOF) {
        perror("Can not parse hub and port of the devices");
    }

    if (lHub != rHub) {
        return rHub - lHub;
    }

    return rPort - lPort;
}

int usb_list_devices(uint16_t vid, uint16_t pid, uint8_t dev_des[][2 + 2 + 4 * 7 + 7]) {
	HDEVINFO devInfo;
	static int i;
	SP_DEVINFO_DATA devInfoData;
	char hwid_buff[128];

	devInfoData.cbSize = sizeof(devInfoData);

	devInfo = SetupDiGetClassDevs(&GUID_DEVINTERFACE_USB_DEVICE, NULL, NULL, DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
	if(devInfo == INVALID_HANDLE_VALUE) {
		wperror("SetupDiGetClassDevs");
		return -1;
	}

    for (i=0; SetupDiEnumDeviceInfo(devInfo, i, &devInfoData); i++) {
        if (!SetupDiGetDeviceRegistryProperty(devInfo, &devInfoData, SPDRP_HARDWAREID, NULL, hwid_buff, sizeof(hwid_buff), NULL)) {
            continue;
        }
        uint16_t fvid, fpid;
        if(sscanf(hwid_buff, "USB\\VID_%hx&PID_%hx", (int16_t *)&fvid, (int16_t *)&fpid) != 2) {
            continue;
        }

        dev_des[i][0] = ((fvid & 0xFF00)>>8);
        dev_des[i][1] = ((fvid & 0x00FF) >> 0);
        dev_des[i][2] = ((fpid & 0xFF00) >> 8);
        dev_des[i][3] = ((fpid & 0x00FF) >> 0);
        sprintf((char *)&dev_des[i][4], "%s", gen_addr(devInfo, &devInfoData, fpid));
    }
    SetupDiDestroyDeviceInfoList(devInfo);

    qsort(dev_des, i, sizeof(dev_des[0]), compareDeviceByHubAndPort);

    return i;
}

void * enumerate_usb_device(uint16_t vid, uint16_t pid, const char *addr, int loud) {
	HDEVINFO devInfo;
	SP_DEVICE_INTERFACE_DATA ifaceData;
	int i;
	SP_DEVINFO_DATA devInfoData;
	char hwid_buff[128];
	int found, found_ind = -1;
	const char *caddr;

	devInfoData.cbSize = sizeof(devInfoData);

	devInfo = SetupDiGetClassDevs(&GUID_DEVINTERFACE_USB_DEVICE, NULL, NULL, DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
	if(devInfo == INVALID_HANDLE_VALUE) {
		wperror("SetupDiGetClassDevs");
		return USB_DEV_NONE;
	}
	found = 0;
	for(i=0; SetupDiEnumDeviceInfo(devInfo, i, &devInfoData); i++) {
		if(!SetupDiGetDeviceRegistryProperty(devInfo, &devInfoData, SPDRP_HARDWAREID, NULL, hwid_buff, sizeof(hwid_buff), NULL))
			continue;
		uint16_t fvid, fpid;
        if(sscanf(hwid_buff, "USB\\VID_%hx&PID_%hx", (int16_t*)&fvid, (int16_t*)&fpid) != 2)
			continue;
		if(verbose && loud)
			fprintf(msgfile, "Vendor/Product ID: %04x:%04x\n", fvid, fpid);
		if((fvid == vid) && (fpid == pid)) {
			caddr = gen_addr(devInfo, &devInfoData, fpid);
			if((addr == NULL) || !strcmp(caddr, addr)) {
				if(verbose)
					fprintf(msgfile, "Found device with VID/PID %04x:%04x , address %s\n", vid, pid, caddr);
				if(!found) {
					found_ind = i;
					found = 1;
				}
				if(!(verbose && loud))
					break;
			}
		}
	}
	if(!found) {
		SetupDiDestroyDeviceInfoList(devInfo);
		return USB_DEV_NONE;
	}
	if(verbose && loud) {
		if(!SetupDiEnumDeviceInfo(devInfo, found_ind, &devInfoData)) {
			wperror("SetupDiEnumDeviceInfo");
			SetupDiDestroyDeviceInfoList(devInfo);
			return USB_DEV_NONE;
		}
	}
	ifaceData.cbSize = sizeof(ifaceData);
	if(!SetupDiEnumDeviceInterfaces(devInfo, &devInfoData, &GUID_DEVINTERFACE_USB_DEVICE, 0, &ifaceData)) {
		if(GetLastError() != ERROR_NO_MORE_ITEMS) {
			wperror("SetupDiEnumDeviceInterfaces");
		}
		SetupDiDestroyDeviceInfoList(devInfo);
		return USB_DEV_NONE;
	}
	return retreive_dev_path(devInfo, &ifaceData);
}

usb_dev findDeviceByGUID(GUID guid, int loud)
{
	HDEVINFO devInfo;
	SP_DEVICE_INTERFACE_DATA ifaceData;

	devInfo = SetupDiGetClassDevs(&guid, NULL, NULL, DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
	if (devInfo == INVALID_HANDLE_VALUE) {
		wperror("SetupDiGetClassDevs");
		return USB_DEV_NONE;
	}
	ifaceData.cbSize = sizeof(ifaceData);
	if (!SetupDiEnumDeviceInterfaces(devInfo, NULL, &guid, 0, &ifaceData)) {
		if (GetLastError() != ERROR_NO_MORE_ITEMS) {
			wperror("SetupDiEnumDeviceInterfaces");
		}
		SetupDiDestroyDeviceInfoList(devInfo);
		return USB_DEV_NONE;
	}
	return retreive_dev_path(devInfo, &ifaceData);
}

void * usb_find_device_by_guid(int loud) {
	void *dev = USB_DEV_NONE;
	//Try Myriad 2
	dev = findDeviceByGUID(GUID_DEVINTERFACE_Myriad2, loud);
	if (dev == USB_DEV_NONE)
	{
		//Try Myriad X
		dev = findDeviceByGUID(GUID_DEVINTERFACE_MyriadX, loud);
	}
	return dev;
}

int usb_check_connected(usb_dev dev) {
	HANDLE han;
	if(dev == USB_DEV_NONE)
		return 0;
	han = CreateFile(dev, 0, FILE_SHARE_WRITE | FILE_SHARE_READ,
		NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, NULL);
	if(han == INVALID_HANDLE_VALUE)
		return 0;
	CloseHandle(han);
	return 1;
}

void * usb_open_device(usb_dev dev, uint8_t *ep, uint8_t intfaceno, char *err_string_buff) {
	HANDLE devHan = INVALID_HANDLE_VALUE;
	WINUSB_INTERFACE_HANDLE winUsbHan = INVALID_HANDLE_VALUE;
	USB_INTERFACE_DESCRIPTOR ifaceDesc;
	WINUSB_PIPE_INFORMATION pipeInfo;
	usb_hwnd han = NULL;
	int i;

	if(dev == USB_DEV_NONE)
		return USB_HAN_NONE;

	devHan = CreateFile(dev, GENERIC_WRITE | GENERIC_READ, FILE_SHARE_WRITE | FILE_SHARE_READ,
		NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, NULL);
	if(devHan == INVALID_HANDLE_VALUE) {
		if(err_string_buff != NULL)
			wstrerror(err_string_buff, "CreateFile");
		goto exit_err;
	}

	if(!WinUsb_Initialize(devHan, &winUsbHan)) {
		if (err_string_buff != NULL)
			wstrerror(err_string_buff, "WinUsb_Initialize");
		goto exit_err;
	}

	if(!WinUsb_QueryInterfaceSettings(winUsbHan, 0, &ifaceDesc)) {
		if (err_string_buff != NULL)
			wstrerror(err_string_buff, "WinUsb_QueryInterfaceSettings");
		goto exit_err;
	}

	han = calloc(1, sizeof(*han));
	if(han == NULL) {
		strcpy(err_string_buff, _strerror("malloc"));
		goto exit_err;
	}
	han->devHan = devHan;
	han->winUsbHan = winUsbHan;

	for(i=0; i<ifaceDesc.bNumEndpoints; i++) {
		if(!WinUsb_QueryPipe(winUsbHan, 0, i, &pipeInfo)) {
			if (err_string_buff != NULL)
				wstrerror(err_string_buff, "WinUsb_QueryPipe");
			if(!ignore_errors)
				goto exit_err;
		}
		if(verbose) {
			fprintf(msgfile, "Found EP 0x%02x : max packet size is %u bytes\n",
				pipeInfo.PipeId, pipeInfo.MaximumPacketSize);
		}
		if(pipeInfo.PipeType != UsbdPipeTypeBulk)
			continue;
		int ind = USB_ENDPOINT_DIRECTION_IN(pipeInfo.PipeId) ? USB_DIR_IN : USB_DIR_OUT;
		han->eps[ind].ep = pipeInfo.PipeId;
		han->eps[ind].sz = pipeInfo.MaximumPacketSize;
		han->eps[ind].last_timeout = 0;
	}
	if(ep)
		*ep = han->eps[USB_DIR_OUT].ep;

	if(err_string_buff && (han->eps[USB_DIR_IN].ep == 0)) {
		sprintf(err_string_buff, "Unable to find BULK IN endpoint\n");
		goto exit_err;
	}
	if(err_string_buff && (han->eps[USB_DIR_OUT].ep == 0)) {
		sprintf(err_string_buff, "Unable to find BULK OUT endpoint\n");
		goto exit_err;
	}
	if(err_string_buff && (han->eps[USB_DIR_IN].sz == 0)) {
		sprintf(err_string_buff, "Unable to find BULK IN endpoint size\n");
		goto exit_err;
	}
	if(err_string_buff && (han->eps[USB_DIR_OUT].sz == 0)) {
		sprintf(err_string_buff, "Unable to find BULK OUT endpoint size\n");
		goto exit_err;
	}
	return han;
exit_err:
	usb_close_device(han);
	return USB_HAN_NONE;
}

uint8_t usb_get_bulk_endpoint(usb_hwnd han, int dir) {
	if((han == NULL) || ((dir != USB_DIR_OUT) && (dir != USB_DIR_IN)))
		return 0;
	return han->eps[dir].ep;
}

size_t usb_get_endpoint_size(usb_hwnd han, uint8_t ep) {
	if(han == NULL)
		return 0;
	if(han->eps[USB_DIR_OUT].ep == ep)
		return han->eps[USB_DIR_OUT].sz;
	if(han->eps[USB_DIR_IN].ep == ep)
		return han->eps[USB_DIR_IN].sz;
	return 0;
}

int usb_bulk_write(usb_hwnd han, uint8_t ep, const void *buffer, size_t sz, uint32_t *wrote_bytes, int timeout_ms) {
	ULONG wb = 0;
	if(wrote_bytes != NULL)
		*wrote_bytes = 0;
	if(han == NULL)
		return USB_ERR_INVALID;

	if(timeout_ms != han->eps[USB_DIR_OUT].last_timeout) {
		han->eps[USB_DIR_OUT].last_timeout = timeout_ms;
		if(!WinUsb_SetPipePolicy(han->winUsbHan, ep, PIPE_TRANSFER_TIMEOUT,
			sizeof(ULONG), &han->eps[USB_DIR_OUT].last_timeout)) {
			last_bulk_errcode = GetLastError();
			wperror("WinUsb_SetPipePolicy");
			return USB_ERR_FAILED;
		}
	}
	if(!WinUsb_WritePipe(han->winUsbHan, ep, (PUCHAR)buffer, (ULONG)sz, &wb, NULL)) {
		last_bulk_errcode = GetLastError();
		if(last_bulk_errcode == ERROR_SEM_TIMEOUT)
			return USB_ERR_TIMEOUT;
		wperror("WinUsb_WritePipe");
		printf("\nWinUsb_WritePipe failed with error:=%d\n", GetLastError());
		return USB_ERR_FAILED;
	}
	last_bulk_errcode = 0;
	if(wrote_bytes != NULL)
		*wrote_bytes = wb;
	return USB_ERR_NONE;
}

int usb_bulk_read(usb_hwnd han, uint8_t ep, void *buffer, size_t sz, uint32_t *read_bytes, int timeout_ms) {
	ULONG rb = 0;
	if(read_bytes != NULL)
		*read_bytes = 0;
	if(han == NULL)
		return USB_ERR_INVALID;

	if(timeout_ms != han->eps[USB_DIR_IN].last_timeout) {
		han->eps[USB_DIR_IN].last_timeout = timeout_ms;
		if(!WinUsb_SetPipePolicy(han->winUsbHan, ep, PIPE_TRANSFER_TIMEOUT,
			sizeof(ULONG), &han->eps[USB_DIR_IN].last_timeout)) {
			last_bulk_errcode = GetLastError();
			wperror("WinUsb_SetPipePolicy");
			return USB_ERR_FAILED;
		}
	}
	if(sz == 0)
		return USB_ERR_NONE;
	if(!WinUsb_ReadPipe(han->winUsbHan, ep, buffer, (ULONG)sz, &rb, NULL)) {
		last_bulk_errcode = GetLastError();
		if(last_bulk_errcode == ERROR_SEM_TIMEOUT)
			return USB_ERR_TIMEOUT;
		wperror("WinUsb_ReadPipe");
		return USB_ERR_FAILED;
	}
	last_bulk_errcode = 0;
	if(read_bytes != NULL)
		*read_bytes = rb;
	return USB_ERR_NONE;
}

void usb_free_device(usb_dev dev) {
	if(dev != NULL)
		free(dev);
}

void usb_close_device(usb_hwnd han) {
	if(han == NULL)
		return;
	WinUsb_Free(han->winUsbHan);
	CloseHandle(han->devHan);
	free(han);
}

const char *usb_last_bulk_errmsg(void) {
	return format_win32_msg(last_bulk_errcode);
}

void usb_set_msgfile(FILE *file) {
	msgfile = file;
}

void usb_set_verbose(int value) {
	verbose = value;
}

void usb_set_ignoreerrors(int value) {
	ignore_errors = value;
}
