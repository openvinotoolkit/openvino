// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#pragma comment(lib, "Setupapi.lib")
#pragma comment(lib, "cfgmgr32.lib")

#include <windows.h>

#ifndef _NTDEF_
typedef LONG NTSTATUS;
#endif

#ifdef __MINGW32__
#include <ntdef.h>
#include <winbase.h>
#endif

#ifdef __CYGWIN__
#include <ntdef.h>
#define _wcsdup wcsdup
#endif

#define MAX_STRING_WCHARS 0xFFF

#include <setupapi.h>
#include <cfgmgr32.h>
#include <winioctl.h>
#include <wchar.h>
#include <stdio.h>
#include <stdlib.h>

#include "hid-dev.h"
#include <osl.h>

#define WSTR_LENGTH 512

typedef USHORT USAGE;
typedef void* BSL_PHIDP_PREPARSED_DATA;

typedef struct _BSL_HIDP_CAPS {
  USAGE Usage;
  USAGE UsagePage;
  USHORT InputReportByteLength;
  USHORT OutputReportByteLength;
  USHORT FeatureReportByteLength;
  USHORT Reserved[17];
  USHORT fields_not_used_by_hidapi[10];
} BSL_HIDP_CAPS, *BSL_PHIDP_CAPS;

#define BSL_HIDP_STATUS_SUCCESS 0x110000

typedef struct _BSL_HIDD_ATTRIBUTES {
  ULONG Size;
  USHORT VendorID;
  USHORT ProductID;
  USHORT VersionNumber;
} BSL_HIDD_ATTRIBUTES, *BSL_PHIDD_ATTRIBUTES;

#undef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))

#ifdef _MSC_VER
/* Thanks Microsoft, but I know how to use strncpy(). */
#pragma warning(disable : 4996)
#endif

struct hidapi_device_internal {
  HANDLE device_handle;
  BOOL blocking;
  USHORT output_report_length;
  size_t input_report_length;
  void* last_error_str;
  DWORD last_error_num;
  BOOL read_pending;
  char* read_buf;
  OVERLAPPED ol;
};

typedef BOOLEAN(__stdcall* HidD_GetAttributes_Func)(HANDLE device, BSL_PHIDD_ATTRIBUTES attrib);
typedef BOOLEAN(__stdcall* HidD_GetSerialNumberString_Func)(HANDLE device, PVOID buffer, ULONG buffer_len);

typedef BOOLEAN(__stdcall* HidD_GetPreparsedData_Func)(HANDLE handle, BSL_PHIDP_PREPARSED_DATA* preparsed_data);
typedef BOOLEAN(__stdcall* HidD_FreePreparsedData_Func)(BSL_PHIDP_PREPARSED_DATA preparsed_data);
typedef NTSTATUS(__stdcall* HidP_GetCaps_Func)(BSL_PHIDP_PREPARSED_DATA preparsed_data, BSL_HIDP_CAPS* caps);
typedef BOOLEAN(__stdcall* HidD_SetNumInputBuffers_Func)(HANDLE handle, ULONG number_buffers);
typedef BOOLEAN(__stdcall* HidD_GetManufacturerString_Func)(HANDLE handle, PVOID buffer, ULONG buffer_len);
typedef BOOLEAN(__stdcall* HidD_GetProductString_Func)(HANDLE handle, PVOID buffer, ULONG buffer_len);
typedef BOOLEAN(__stdcall* HidD_GetHidGuid_Func)(LPGUID HidGuid);

static HMODULE lib_handle = NULL;
static BOOLEAN initialized = FALSE;

static HidD_GetHidGuid_Func HidD_GetHidGuid;
static HidD_GetAttributes_Func HidD_GetAttributes;
static HidD_GetSerialNumberString_Func HidD_GetSerialNumberString;
static HidD_GetManufacturerString_Func HidD_GetManufacturerString;
static HidD_GetProductString_Func HidD_GetProductString;

static HidD_GetPreparsedData_Func HidD_GetPreparsedData;
static HidD_FreePreparsedData_Func HidD_FreePreparsedData;
static HidP_GetCaps_Func HidP_GetCaps;
static HidD_SetNumInputBuffers_Func HidD_SetNumInputBuffers;

static void free_hid_device(hidapi_device_s* dev) {
  CloseHandle(dev->ol.hEvent);
  CloseHandle(dev->device_handle);
  LocalFree(dev->last_error_str);
  free(dev->read_buf);
  free(dev);
}

static int lookup_functions() {
  SetSearchPathMode(BASE_SEARCH_PATH_ENABLE_SAFE_SEARCHMODE);
  lib_handle = LoadLibraryA("hid.dll");
  if (lib_handle) {
#define RESOLVE(x)                              \
  x = (x##_Func)GetProcAddress(lib_handle, #x); \
  if (!x)                                       \
    return -1;
    RESOLVE(HidD_GetAttributes);
    RESOLVE(HidD_GetPreparsedData);
    RESOLVE(HidD_FreePreparsedData);
    RESOLVE(HidP_GetCaps);
    RESOLVE(HidD_SetNumInputBuffers);
    RESOLVE(HidD_GetSerialNumberString);
    RESOLVE(HidD_GetManufacturerString);
    RESOLVE(HidD_GetProductString);
    RESOLVE(HidD_GetHidGuid);
#undef RESOLVE
  } else
    return -1;

  return 0;
}
static void bsl__register_error(hidapi_device_s* device, const char* op) {
  WCHAR *ptr, *msg;

  FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL,
                 GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPVOID)&msg, 0 /*sz*/, NULL);

  ptr = msg;
  while (*ptr) {
    if (*ptr == '\r') {
      *ptr = 0x0000;
      break;
    }
    ptr++;
  }

  LocalFree(device->last_error_str);
  device->last_error_str = msg;
}

static HANDLE open_device(const char* path, BOOL enumerate) {
  DWORD share_mode = FILE_SHARE_READ | FILE_SHARE_WRITE;
  HANDLE handle;
  DWORD desired_access = (enumerate) ? 0 : (GENERIC_WRITE | GENERIC_READ);
  handle = CreateFileA(path, desired_access, share_mode, NULL, OPEN_EXISTING, FILE_FLAG_OVERLAPPED,
                       /*FILE_ATTRIBUTE_NORMAL,*/ 0);

  return handle;
}

static hidapi_device_s* new_hid_device() {
  hidapi_device_s* dev = (hidapi_device_s*)calloc(1, sizeof(hidapi_device_s));
  if (!dev) {
    return NULL;
  }
  dev->device_handle = INVALID_HANDLE_VALUE;
  dev->blocking = TRUE;
  dev->output_report_length = 0;
  dev->input_report_length = 0;
  dev->last_error_str = NULL;
  dev->last_error_num = 0;
  dev->read_pending = FALSE;
  dev->read_buf = NULL;
  memset(&dev->ol, 0, sizeof(dev->ol));
  dev->ol.hEvent = CreateEvent(NULL, FALSE, FALSE /*initial state f=nonsignaled*/, NULL);

  return dev;
}

int hidapi_exit(void) {
  if (lib_handle)
    FreeLibrary(lib_handle);
  lib_handle = NULL;
  initialized = FALSE;

  return 0;
}

int hidapi_init(void) {
  if (!initialized) {
    if (lookup_functions() < 0) {
      hidapi_exit();
      return -1;
    }
    initialized = TRUE;
  }

  return 0;
}

char* replace_char(char* str, char find, char replace) {
  char* current_pos = strchr(str, find);
  while (current_pos) {
    *current_pos = replace;
    current_pos = strchr(current_pos, find);
  }
  return str;
}

static void add_node_to_hid_device_info_list(struct hidapi_device_info_s** root,
                                      struct hidapi_device_info_s** current,
                                      struct hidapi_device_info_s* new_node) {
  if (*current) {
    (*current)->next = new_node;
  } else {
    *root = new_node;
  }
  *current = new_node;
}

static bool device_is_in_path_list(const char* device_path, const char* path_array[], const size_t path_array_len) {
  if (path_array_len == 0) {
    return true;
  }
  char device_path_upr[MAX_PATH_LENGTH];
  bsl_strncpy(device_path_upr, MAX_PATH_LENGTH, device_path, MAX_PATH_LENGTH);
  strupr(device_path_upr);
  replace_char(device_path_upr, '#', '\\');

  for (size_t i = 0; i < path_array_len; i++) {
    char path[MAX_PATH_LENGTH];
    bsl_strncpy(path, sizeof(path), path_array[i], MAX_PATH_LENGTH);
    strupr(path);
    if (strstr(device_path_upr, path)) {
      return true;
    }
  }
  return false;
}

static bool vid_pid_matched(BSL_HIDD_ATTRIBUTES* device_attributes,
                            unsigned short vendor_id,
                            unsigned short product_id) {
  bool vid_matched = vendor_id == 0 || vendor_id == device_attributes->VendorID;
  bool pid_matched = product_id == 0 || product_id == device_attributes->ProductID;
  bool is_our_device = vid_matched && pid_matched;
  return is_our_device;
}

static struct hidapi_device_info_s* get_tmp_node_of_device(SP_DEVICE_INTERFACE_DETAIL_DATA_A* device_detail,
                                                           HANDLE device_write_handle,
                                                           BSL_HIDD_ATTRIBUTES* attrib) {
  struct hidapi_device_info_s* tmp = malloc(sizeof(struct hidapi_device_info_s));
  bool succ = false;
  if (tmp == NULL) {
    return NULL;
  }
  memset(tmp, 0, sizeof(*tmp));

  const char* dev_path = device_detail->DevicePath;
  tmp->path = dev_path ? strdup(dev_path) : NULL;
  tmp->vendor_id = attrib->VendorID;
  tmp->product_id = attrib->ProductID;
  tmp->release_number = attrib->VersionNumber;

  wchar_t wstr[WSTR_LENGTH];
  succ = HidD_GetSerialNumberString(device_write_handle, wstr, sizeof(wstr));
  wstr[WSTR_LENGTH - 1] = 0x0000;
  if (succ) {
    tmp->serial_number = _wcsdup(wstr);
  }

  succ = HidD_GetManufacturerString(device_write_handle, wstr, sizeof(wstr));
  wstr[WSTR_LENGTH - 1] = 0x0000;
  if (succ) {
    tmp->manufacturer_string = _wcsdup(wstr);
  }

  succ = HidD_GetProductString(device_write_handle, wstr, sizeof(wstr));
  wstr[WSTR_LENGTH - 1] = 0x0000;
  if (succ) {
    tmp->product_string = _wcsdup(wstr);
  }

  BSL_PHIDP_PREPARSED_DATA pp_data = NULL;
  succ = HidD_GetPreparsedData(device_write_handle, &pp_data);
  if (succ) {
    BSL_HIDP_CAPS caps;
    if (HidP_GetCaps(pp_data, &caps) == BSL_HIDP_STATUS_SUCCESS) {
      tmp->usage = caps.Usage;
      tmp->usage_page = caps.UsagePage;
    }
    HidD_FreePreparsedData(pp_data);
  }

  tmp->interface_number = -1;
  if (tmp->path) {
    const char* interface_component = strstr(tmp->path, "&mi_");
    if (interface_component) {
      char* hex_str = interface_component + 4;
      char* endptr = NULL;
      tmp->interface_number = strtol(hex_str, &endptr, 16);
      if (endptr == hex_str) {
        tmp->interface_number = -1;
      }
    }
  }

  tmp->next = NULL;
  return tmp;
}

static SP_DEVICE_INTERFACE_DETAIL_DATA_A* get_interface_detail_data(HDEVINFO info_set,
                                                             SP_DEVICE_INTERFACE_DATA* interface_data) {
  bool succ;
  DWORD required_size = 0;
  succ = SetupDiGetDeviceInterfaceDetailA(info_set, interface_data, NULL, 0, &required_size, NULL);
  if (!required_size) {
    return NULL;
  }

  SP_DEVICE_INTERFACE_DETAIL_DATA_A* detail_data = malloc(required_size);
  if (!detail_data) {
    return NULL;
  }
  memset(detail_data, 0, required_size);
  detail_data->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA);

  succ = SetupDiGetDeviceInterfaceDetailA(info_set, interface_data, detail_data, required_size, NULL, NULL);
  if (!succ) {
    free(detail_data);
    return NULL;
  }

  return detail_data;
}

static BOOL gather_device_infos(_In_ HDEVINFO info_set,
                                _In_ SP_DEVICE_INTERFACE_DATA* interface_data,
                                _Out_ SP_DEVICE_INTERFACE_DETAIL_DATA_A** p_detail_data,
                                _Out_ HANDLE* p_write_handle,
                                _Out_ BSL_HIDD_ATTRIBUTES* p_attrib) {
  SP_DEVICE_INTERFACE_DETAIL_DATA_A* interface_detail_data = get_interface_detail_data(info_set, interface_data);
  if (interface_detail_data == NULL) {
    goto error_exit;
  }

  HANDLE dev_write_handle = open_device(interface_detail_data->DevicePath, TRUE);
  if (dev_write_handle == INVALID_HANDLE_VALUE) {
    goto error_exit;
  }

  BSL_HIDD_ATTRIBUTES attrib = {0};
  attrib.Size = sizeof(BSL_HIDD_ATTRIBUTES);
  if (!HidD_GetAttributes(dev_write_handle, &attrib)) {
    CloseHandle(dev_write_handle);
    goto error_exit;
  }

  *p_detail_data = interface_detail_data;
  *p_write_handle = dev_write_handle;
  *p_attrib = attrib;
  return true;

error_exit:
  free(interface_detail_data);
  return false;
}

struct hidapi_device_info_s* hidapi_enumerate(unsigned short vendor_id,
                                              unsigned short product_id,
                                              const char* path_array[],
                                              const size_t path_array_len) {
  struct hidapi_device_info_s* result_list_root = NULL;
  struct hidapi_device_info_s* current_tail = NULL;
  if (hidapi_init() != 0) {
    return NULL;
  }

  GUID HidGuid = {0};
  HidD_GetHidGuid(&HidGuid);
  HDEVINFO info_set = SetupDiGetClassDevsA(&HidGuid, NULL, NULL, DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);

  for (DWORD member_index = 0;; member_index++) {
    BOOL succ = FALSE;

    SP_DEVICE_INTERFACE_DATA interface_data = {0};
    interface_data.cbSize = sizeof(SP_DEVICE_INTERFACE_DATA);
    succ = SetupDiEnumDeviceInterfaces(info_set, NULL, &HidGuid, member_index, &interface_data);
    if (!succ) {
      break;
    }

    SP_DEVICE_INTERFACE_DETAIL_DATA_A* detail_data = NULL;
    HANDLE write_handle = INVALID_HANDLE_VALUE;
    BSL_HIDD_ATTRIBUTES attrib = {0};
    succ = gather_device_infos(info_set, &interface_data, &detail_data, &write_handle, &attrib);
    if (!succ) {
      continue;
    };

    if (vid_pid_matched(&attrib, vendor_id, product_id)) {
      if (device_is_in_path_list(detail_data->DevicePath, path_array, path_array_len)) {
        printf("A matched device found at %s\n", detail_data->DevicePath);
        struct hidapi_device_info_s* tmp = get_tmp_node_of_device(detail_data, write_handle, &attrib);
        add_node_to_hid_device_info_list(&result_list_root, &current_tail, tmp);
      }
    }
    CloseHandle(write_handle);
    free(detail_data);
  }

  SetupDiDestroyDeviceInfoList(info_set);
  return result_list_root;
}

hidapi_device_s* hidapi_open_path(const char* path) {
  hidapi_device_s* dev;
  BSL_HIDP_CAPS caps;
  BSL_PHIDP_PREPARSED_DATA pp_data = NULL;
  BOOLEAN res;
  NTSTATUS nt_res;

  if (hidapi_init() < 0) {
    return NULL;
  }

  dev = new_hid_device();

  if (!dev) {
    return NULL;
  }

  dev->device_handle = open_device(path, FALSE);

  if (dev->device_handle == INVALID_HANDLE_VALUE) {
    bsl__register_error(dev, "CreateFile");
    goto err;
  }

  res = HidD_SetNumInputBuffers(dev->device_handle, 64);
  if (!res) {
    bsl__register_error(dev, "HidD_SetNumInputBuffers");
    goto err;
  }

  res = HidD_GetPreparsedData(dev->device_handle, &pp_data);
  if (!res) {
    bsl__register_error(dev, "HidD_GetPreparsedData");
    goto err;
  }
  nt_res = HidP_GetCaps(pp_data, &caps);
  if (nt_res != BSL_HIDP_STATUS_SUCCESS) {
    bsl__register_error(dev, "HidP_GetCaps");
    goto err_pp_data;
  }

  dev->input_report_length = caps.InputReportByteLength;
  dev->output_report_length = caps.OutputReportByteLength;
  HidD_FreePreparsedData(pp_data);

  dev->read_buf = (char*)malloc(dev->input_report_length);

  return dev;

err_pp_data:
  HidD_FreePreparsedData(pp_data);
err:
  free_hid_device(dev);
  return NULL;
}

hidapi_device_s* hidapi_open(unsigned short vid, unsigned short pid, const wchar_t* serial_number) {
  struct hidapi_device_info_s *devs, *current_dev;
  const char* path_to_open = NULL;
  hidapi_device_s* handle = NULL;

  devs = hidapi_enumerate(vid, pid, NULL, 0);
  current_dev = devs;
  while (current_dev) {
    if (current_dev->vendor_id == vid && current_dev->product_id == pid) {
      if (serial_number) {
        if (wcscmp(serial_number, current_dev->serial_number) == 0) {
          path_to_open = current_dev->path;
          break;
        }
      } else {
        path_to_open = current_dev->path;
        break;
      }
    }
    current_dev = current_dev->next;
  }

  if (path_to_open) {
    handle = hidapi_open_path(path_to_open);
  }

  hidapi_free_enumeration(devs);

  return handle;
}
void hidapi_free_enumeration(struct hidapi_device_info_s* devs) {
  struct hidapi_device_info_s* d = devs;
  while (d) {
    struct hidapi_device_info_s* next = d->next;
    free(d->serial_number);
    free(d->manufacturer_string);
    free(d->product_string);
    free(d->path);
    free(d);
    d = next;
  }
}
int hidapi_write(hidapi_device_s* dev, const unsigned char* data, size_t length) {
  unsigned char* buf;
  OVERLAPPED ol;
  memset(&ol, 0, sizeof(ol));

  DWORD bytes_written;
  BOOL res;
  if (length >= dev->output_report_length) {
    buf = (unsigned char*)data;
  } else {
    buf = (unsigned char*)malloc(dev->output_report_length);
    memcpy(buf + dev->output_report_length - length, data, length);
    memset(buf, 0, dev->output_report_length - length);
    length = dev->output_report_length;
  }

  res = WriteFile(dev->device_handle, buf, length, NULL, &ol);

  if (!res) {
    if (GetLastError() != ERROR_IO_PENDING) {
      bsl__register_error(dev, "WriteFile");
      bytes_written = -1;
      goto end_of_function;
    }
  }

  res = GetOverlappedResult(dev->device_handle, &ol, &bytes_written, TRUE /*wait*/);
  if (!res) {
    bsl__register_error(dev, "WriteFile");
    bytes_written = -1;
    goto end_of_function;
  }

end_of_function:
  if (buf != data)
    free(buf);

  return bytes_written;
}

int hidapi_read_timeout(hidapi_device_s* hid_dev, unsigned char* data, size_t length, int milliseconds) {
  size_t copy_len = 0;

  DWORD bytes_read = 0;
  HANDLE ev = hid_dev->ol.hEvent;
  BOOL res;

  if (!hid_dev->read_pending) {
    hid_dev->read_pending = TRUE;
    memset(hid_dev->read_buf, 0, hid_dev->input_report_length);
    ResetEvent(ev);
    res = ReadFile(hid_dev->device_handle, hid_dev->read_buf, hid_dev->input_report_length, &bytes_read, &hid_dev->ol);

    if (!res) {
      if (GetLastError() != ERROR_IO_PENDING) {
        CancelIo(hid_dev->device_handle);
        hid_dev->read_pending = FALSE;
        goto end_of_function;
      }
    }
  }

  if (milliseconds >= 0) {
    res = WaitForSingleObject(ev, milliseconds);
    if (res != WAIT_OBJECT_0) {
      return 0;
    }
  }

  res = GetOverlappedResult(hid_dev->device_handle, &hid_dev->ol, &bytes_read, TRUE /*wait*/);

  hid_dev->read_pending = FALSE;

  if (res && bytes_read > 0) {
    if (hid_dev->read_buf[0] == 0x0) {
      bytes_read--;
      copy_len = length > bytes_read ? bytes_read : length;
      memcpy(data, hid_dev->read_buf + 1, copy_len);
    } else {
      copy_len = length > bytes_read ? bytes_read : length;
      memcpy(data, hid_dev->read_buf, copy_len);
    }
  }

end_of_function:
  if (!res) {
    bsl__register_error(hid_dev, "GetOverlappedResult");
    return -1;
  }

  return copy_len;
}

int hidapi_read(hidapi_device_s* hid_dev, unsigned char* data, size_t length) {
  return hidapi_read_timeout(hid_dev, data, length, (hid_dev->blocking) ? -1 : 0);
}

int hidapi_set_nonblocking(hidapi_device_s* hid_dev, int nonblock) {
  hid_dev->blocking = !nonblock;
  return 0;
}

void hidapi_close(hidapi_device_s* hid_dev) {
  if (!hid_dev)
    return;
  CancelIo(hid_dev->device_handle);
  free_hid_device(hid_dev);
}
