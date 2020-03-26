// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <assert.h>
#include <stdio.h>

#ifndef WIN32
#include <poll.h>
#endif

#include "hddl-bsl.h"
#include "hddl_bsl_priv.h"
#include "hddl_bsl_thread.h"
#include "hid-dev.h"

#define F75114_MAX_RESET_PIN_NUM (8)
#define F75114_MAX_BOARDID_PIN_NUM (3)

#define F75114_VID (0x2c42)
#define F75114_PID (0x5114)

#define HID_CMD_SIGNATURE 0x43444948

// HID UART Command
#define HID_UART_INIT 0x70
#define USB_DATA_IN 0x71
#define USB_DATA_OUT 0x72
#define HID_GET_STATUS 0x80

#define HID_PACKET_SIZE 0x20

// device id pin
#define HID_DEVICE_ID_PIN1 0x0
#define HID_DEVICE_ID_PIN2 0x1
#define HID_DEVICE_ID_PIN3 0x7
#define HID_DEVICE_ID_PIN4 0x10
#define HID_DEVICE_ID_PIN5 0x11
#define HID_DEVICE_ID_PIN6 0x12
#define HID_DEVICE_ID_PIN7 0x13
#define HID_DEVICE_ID_PIN8 0x14

// board id pin
#define HID_BOARD_ID_PIN1 2
#define HID_BOARD_ID_PIN2 3
#define HID_BOARD_ID_PIN3 4

typedef enum {
  eGPIO_Direction_In,
  eGPIO_Direction_Out,
  eGPIO_Direction_Invalid,
} eGPIO_Direction;

typedef enum {
  eGPIO_Pull_Low,
  eGPIO_Pull_High,
  eGPIO_Pull_Disable,
  eGPIO_Pull_Invalid,
} eGPIO_Pull_Mode;

#ifdef WIN32
#pragma pack(push) /* push current alignment to stack */
#pragma pack(1)    /* set alignment to 1 byte boundary */

typedef struct {
  unsigned char cmd;    // 1   command Code
  unsigned char len;    // 1   command Length
  unsigned short arg1;  // 2:
  unsigned short arg2;  // 2:
  unsigned short arg3;  // 2:  startAddress
  unsigned short arg4;  // 2:  r/w length

  unsigned int signature;  // 4:  HIDC
  unsigned short Reserve;  // 2
  unsigned char DevType;
  unsigned char FunIdx;
  unsigned char data1;
  unsigned char data2;
  unsigned char data3;
  unsigned char data4;
  unsigned char data5;
  unsigned char data6;
  unsigned int checksum;
} HID_CMD_T;

#pragma pack(pop) /* restore original alignment from stack */

#else
typedef struct {
  unsigned char cmd;    // 1   command Code
  unsigned char len;    // 1   command Length
  unsigned short arg1;  // 2:
  unsigned short arg2;  // 2:
  unsigned short arg3;  // 2:  startAddress
  unsigned short arg4;  // 2:  r/w length

  unsigned int signature;  // 4:  HIDC
  unsigned short Reserve;  // 2
  unsigned char DevType;
  unsigned char FunIdx;
  unsigned char data1;
  unsigned char data2;
  unsigned char data3;
  unsigned char data4;
  unsigned char data5;
  unsigned char data6;
  unsigned int checksum;
} __attribute__((packed)) HID_CMD_T;

#endif

static unsigned char m_hid_count = 0;

static char m_hid_path_list[MAX_HID_DEVICE_NUM][MAX_PATH_LENGTH];
static int m_hid_boardid_list[MAX_HID_DEVICE_NUM];

static bsl_mutex_t m_hid_mutex;

typedef struct {
  int pin_id;
  int device_id;
} F75114_PIN_MAP;

static const F75114_PIN_MAP m_hid_reset_pin_map[F75114_MAX_RESET_PIN_NUM] = {
    {HID_DEVICE_ID_PIN1, 0}, {HID_DEVICE_ID_PIN2, 1}, {HID_DEVICE_ID_PIN3, 2}, {HID_DEVICE_ID_PIN4, 3},
    {HID_DEVICE_ID_PIN5, 4}, {HID_DEVICE_ID_PIN6, 5}, {HID_DEVICE_ID_PIN7, 6}, {HID_DEVICE_ID_PIN8, 7}};

static const int m_hid_brd_pin_list[F75114_MAX_BOARDID_PIN_NUM] = {HID_BOARD_ID_PIN1, HID_BOARD_ID_PIN2,
                                                                   HID_BOARD_ID_PIN3};

unsigned int _cal_checksum(unsigned char* buf, unsigned int size) {
  unsigned int sum;
  int i;

  i = 0;
  sum = 0;
  while (size--) {
    sum += buf[i++];
  }

  return sum;
}

// if output_buf is null, means caller do not care the output data
int _send_buffer_to_dev(hidapi_device_s* handle, unsigned char* read_write_buff) {
  int ret = -1;

  int length = -1;
  //#ifdef DEBUG
  //  int i = 0;
  //  for (i = 0; i < HID_PACKET_SIZE; i++) {
  //    LOG(" 0x%x", read_write_buff[i]);
  //  }
  //  LOG("\n");
  //#endif
  ret = hidapi_write(handle, read_write_buff, HID_PACKET_SIZE);
  if (ret <= 0) {
    LOG("%s: line:%d\n", __func__, __LINE__);
    return -1;
  }
  LOG("[%s %d] write ret=%d\n", __func__, __LINE__, ret);

#ifdef WIN32
  usleep(2000);
#else
  struct pollfd fds;

  fds.fd = hidapi_get_fd(handle);
  fds.events = POLLIN;
  fds.revents = 0;
  ret = poll(&fds, 1, 50);  //-1 to milliseconds

  if (ret == -1 || ret == 0)
  // if (ret == -1 || (fds.revents & ~POLLOUT) != 0)
  {
    LOG("[%s:%d] failed to poll data from f75114\n", __func__, __LINE__);
    return -1;
  } else {
    /* Check for errors on the file descriptor. This will
     indicate a device disconnection. */
    if (fds.revents & (POLLERR | POLLHUP | POLLNVAL)) {
      LOG("[%s:%d] device maybe disconnect\n", __func__, __LINE__);
      return -1;
    }
  }

#endif

  length = hidapi_read(handle, read_write_buff, HID_PACKET_SIZE);

  LOG("read length: %d\n", length);

  return length > 0 ? 0 : -1;
}

// if output_buf is null, means caller do not care the output data
int _send_cmd_to_dev(hidapi_device_s* handle, HID_CMD_T* cmd, unsigned char* output_buf) {
  unsigned char buf_backup[HID_PACKET_SIZE] = {0x0};
  assert(cmd);
  cmd->checksum = _cal_checksum((unsigned char*)cmd, cmd->len);

  unsigned char* buf = output_buf;
  if (buf == NULL)
    buf = buf_backup;
  memcpy(buf, cmd, sizeof(HID_CMD_T));
  return _send_buffer_to_dev(handle, buf);
}

int _send_init_cmd(hidapi_device_s* handle) {
  HID_CMD_T cmd;
  int ret;
  unsigned char buf[HID_PACKET_SIZE] = {0x0};
  memset(&cmd, 0, sizeof(cmd));

  cmd.cmd = HID_UART_INIT;
  // cmd.len = sizeof(cmd) - 4;  /* Not include checksum */
  cmd.arg1 = 0;
  cmd.arg3 = 0;
  cmd.arg4 = 0x20;
  cmd.signature = HID_CMD_SIGNATURE;
  cmd.len = cmd.DevType = 0;
  cmd.FunIdx = 0;
  cmd.checksum = _cal_checksum((unsigned char*)&cmd, cmd.len);
  memcpy(buf, &cmd, sizeof(cmd));

  ret = _send_buffer_to_dev(handle, buf);

  if (ret != 0 || buf[0] != 0xFF) {
    LOG("[%s:%d] read result failed;ret=%d;buf[0]=%x\n", __func__, __LINE__, ret, buf[0]);
    return -1;
  }
  return 0;
}

// if (_F75114_API_USB_data_out(fd, 0, 0x20, buff))
int _send_output_cmd_with_buf(hidapi_device_s* handle, int pin_id, int func_id, unsigned char* buf) {
  HID_CMD_T cmd;

  int ret;

  cmd.cmd = USB_DATA_OUT;
  // cmd.len = sizeof(cmd) - 4;  /* Not include checksum */
  cmd.arg1 = 0;
  cmd.arg2 = 0;
  cmd.arg3 = 0;
  cmd.arg4 = 0x20;
  cmd.signature = HID_CMD_SIGNATURE;
  cmd.Reserve = 0;
  cmd.DevType = 0;
  cmd.len = (unsigned char)func_id;
  cmd.FunIdx = (unsigned char)func_id;
  cmd.data1 = buf[2];
  cmd.data2 = buf[3];
  cmd.data3 = buf[4];
  cmd.data4 = buf[5];
  cmd.data5 = buf[6];
  cmd.data6 = buf[7];
  cmd.checksum = _cal_checksum((unsigned char*)&cmd, cmd.len);
  memcpy(buf, &cmd, sizeof(cmd));

  ret = _send_buffer_to_dev(handle, buf);

  if (ret != 0 || buf[0] != 0x80) {
    LOG("[%s:%d] read result failed;ret=%d;buf[0]=%x\n", __func__, __LINE__, ret, buf[0]);
    return -1;
  }

  return 0;
}

int _send_output_cmd(hidapi_device_s* handle, int pin_id, int eMode, int func_id) {
  HID_CMD_T cmd;
  int index = 0;
  int set = 0;
  int count = 0;

  unsigned char buf[HID_PACKET_SIZE] = {0x0};

  memset(&cmd, 0, sizeof(cmd));

  set = (pin_id >> 4) & 0x0f;
  count = (pin_id >> 0) & 0x0f;
  buf[0] = 0x00;                    // GPIO
  buf[1] = (unsigned char)func_id;  // GPIO_DIR

  index = 2 * set + !eMode;
  int buf_index = index + 2;
  if (buf_index < 0 || buf_index >= HID_PACKET_SIZE - 1) {
    return -1;
  }
  buf[buf_index] = (unsigned char)(1 << count);

  return _send_output_cmd_with_buf(handle, pin_id, func_id, buf);
}

// if (_F75114_API_USB_data_in(fd, 0, 0x20, buff))
int _send_input_cmd_with_buf(hidapi_device_s* handle, int pin_id, int func_id, unsigned char* buf) {
  HID_CMD_T cmd;

  memset(&cmd, 0, sizeof(cmd));

  cmd.cmd = USB_DATA_IN;
  // cmd.len = sizeof(cmd) - 4;  /* Not include checksum */
  cmd.arg1 = 0;
  cmd.arg3 = 0;
  cmd.arg4 = 0x20;
  cmd.signature = HID_CMD_SIGNATURE;
  cmd.DevType = 0;
  cmd.len = func_id;
  cmd.FunIdx = func_id;
  cmd.data1 = buf[2];
  cmd.data2 = buf[3];
  cmd.data3 = buf[4];
  cmd.data4 = buf[5];
  cmd.data5 = buf[6];
  cmd.data6 = buf[7];
  cmd.checksum = _cal_checksum((unsigned char*)&cmd, cmd.len);
  memcpy(buf, &cmd, sizeof(cmd));

  return _send_buffer_to_dev(handle, buf);
}

//_SetGpioOutputEnableIdx
//__SetGpioOutputEnableIdx_F75114__
int _send_output_enable_cmd(hidapi_device_s* handle, int pin_id, eGPIO_Direction eMode) {
  return _send_output_cmd(handle, pin_id, (int)eMode, 0x04);
}

//_SetGpioPullMode
//__SetGpioPullMode_F75114__
int _send_pull_mode_cmd(hidapi_device_s* handle, int pin_id, eGPIO_Pull_Mode eMode) {
  if (eMode < eGPIO_Pull_Disable) {
    return _send_output_cmd(handle, pin_id, (int)eMode, 0x01);
  } else {
    int set = 0;
    int count = 0;
    unsigned char addr = 0;
    unsigned char buff[HID_PACKET_SIZE] = {0x00};
    int ret = 0;
    if (set == 0 && count < 4)
      addr = 0x60;
    else if (set == 0 && count > 3)
      addr = 0x61;
    else if (set == 1 && count < 4)
      addr = 0x62;
    else if (set == 1 && count > 3)
      addr = 0x63;
    else if (set == 2)
      addr = 0x64;
    buff[0] = 0xF0;  // For all registry
    buff[1] = 0x00;  // Read
    buff[2] = addr;
    ret = _send_input_cmd_with_buf(handle, pin_id, 0x00, buff);
    if (ret) {
      LOG("[%s:%d] send input cmd failed\n", __func__, __LINE__);
      return ret;
    }
    buff[3] = buff[0];
    // LOG("%s: pre-process buff[3]: %x\n", __func__, buff[0]);
    if ((pin_id & 0x0f) == 0 || (pin_id & 0x0f) == 4)
      buff[3] &= 0xfc;
    else if ((pin_id & 0x0f) == 1 || (pin_id & 0x0f) == 5)
      buff[3] &= 0xf3;
    else if ((pin_id & 0x0f) == 2 || (pin_id & 0x0f) == 6)
      buff[3] &= 0xcf;
    else if ((pin_id & 0x0f) == 3 || (pin_id & 0x0f) == 7)
      buff[3] &= 0x3f;
    // LOG("%s: post-process buff[3]: %x\n", __func__, buff[0]);
    buff[0] = 0xF0;  // For all registry
    buff[1] = 0x01;  // write
    buff[2] = addr;

    ret = _send_output_cmd_with_buf(handle, pin_id, 0x01, buff);
    if (ret) {
      LOG("[%s:%d] send output cmd failed\n", __func__, __LINE__);
      return ret;
    }
    return 0;
  }
}

//_SetGpioOutputDataIdx
//__SetGpioOutputDataIdx_F75114__
int _send_output_data_cmd(hidapi_device_s* handle, int pin_id, int value) {
  if (value != 0)
    value = 1;
  else
    value = 0;
  return _send_output_cmd(handle, pin_id, value, 0x02);
}

//_GetGpioInputDataIdx
//__GetGpioInputDataIdx_F75114__
int _send_get_input_data_cmd(hidapi_device_s* handle, int pin_id, unsigned char* value) {
  HID_CMD_T cmd;

  int func_id = 0x03;
  int set = 0;
  int count = 0;

  // unsigned char buff[HID_PACKET_SIZE] = { 0x0 };
  unsigned char buf[HID_PACKET_SIZE] = {0x0};

  memset(&cmd, 0, sizeof(cmd));

  set = (pin_id >> 4) & 0x0f;
  count = (pin_id >> 0) & 0x0f;
  buf[0] = 0x00;     // GPIO
  buf[1] = func_id;  // GPIO_DIR

  int ret = _send_input_cmd_with_buf(handle, pin_id, func_id, buf);
  if (ret)
    return ret;
  *value = !!(buf[set] & (1 << count));
  return 0;
}

BSL_STATUS _f75114_init() {
  for (int i = 0; i < m_hid_count; i++) {
    char* hid_path = m_hid_path_list[i];
    assert(hid_path);

    hidapi_device_s* handle = hidapi_open_path(hid_path);

    if (handle == NULL) {
      return BSL_ERROR_HID_API_OPEN_FAILED;
    }

    hidapi_set_nonblocking(handle, 1);
    _send_init_cmd(handle);
    int j = 0;
    for (; j < F75114_MAX_RESET_PIN_NUM; j++) {
      int pin_id = m_hid_reset_pin_map[j].pin_id;
      _send_output_enable_cmd(handle, pin_id, eGPIO_Direction_Out);
      _send_pull_mode_cmd(handle, pin_id, eGPIO_Pull_Disable);
      _send_output_data_cmd(handle, pin_id, 1);
    }

    //_send_output_enable_cmd(handle, HID_BOARD_ID_PIN1, eGPIO_Direction_In);
    //_send_output_enable_cmd(handle, HID_BOARD_ID_PIN2, eGPIO_Direction_In);
    //_send_output_enable_cmd(handle, HID_BOARD_ID_PIN3, eGPIO_Direction_In);

    // need read board id,and put it into m_hid_boardid_list
    // m_hid_boardid_list
    unsigned char value = 0;
    int boardid = 0;
    m_hid_boardid_list[i] = 0;

    for (j = 0; j < F75114_MAX_BOARDID_PIN_NUM; j++) {
      value = 0;
      int pin_id = m_hid_brd_pin_list[j];
      _send_output_enable_cmd(handle, pin_id, eGPIO_Direction_In);
      int ret = _send_get_input_data_cmd(handle, pin_id, &value);
      if (ret) {
        LOG("read board id failed:%d\n", ret);
      }
      LOG("board id bit %d:%d\n", j, value);
      boardid |= ((int)value & 0x1) << j;
    }

    LOG("board id is:%d\n", boardid);
    m_hid_boardid_list[i] = boardid;

    hidapi_close(handle);
  }
  return BSL_SUCCESS;
}

// config all pin as output
int m_hid_init() {
  assert(m_hid_count > 0);
  bsl_mutex_init(&m_hid_mutex);
  BSL_STATUS status = _f75114_init();
  return status;
}

int m_hid_destroy() {
  bsl_mutex_destroy(&m_hid_mutex);
  return 0;
}

// high 3 bits is baord id
// low 2 bits is real device id
static void m_hid_parse_device_id(int device_id, int* board_id, int* real_device_id) {
  char data = *((char*)&device_id);
  *real_device_id = (int)(data & 0x07);
  *board_id = (int)((data & 0xE0) >> 5);
}

static BSL_STATUS m_hid_find_resetpin_by_deviceid(int real_device_id, int* pin_id) {
  int i = 0;
  LOG("m_hid_find_resetpin_by_deviceid real_device_id=%d\n", real_device_id);
  for (; i < F75114_MAX_RESET_PIN_NUM; i++) {
    F75114_PIN_MAP item = m_hid_reset_pin_map[i];
    if (item.device_id == real_device_id) {
      *pin_id = item.pin_id;
      return BSL_SUCCESS;
    }
  }
  return BSL_ERROR_INVALID_DEVICE_ID;
}

static BSL_STATUS m_hid_reset(int device_id) {
  int target_board_id = 0;
  int real_device_id = 0;
  m_hid_parse_device_id(device_id, &target_board_id, &real_device_id);
  for (int i = 0; i < m_hid_count; i++) {
    int board_id = m_hid_boardid_list[i];
    if (board_id == target_board_id) {
      int pin_id = 0;
      BSL_STATUS status = m_hid_find_resetpin_by_deviceid(real_device_id, &pin_id);
      if (BSL_SUCCESS != status) {
        return status;
      }
      char* hid_path = m_hid_path_list[i];
      hidapi_device_s* handle = hidapi_open_path(hid_path);
      if (handle == NULL) {
        return BSL_ERROR_HID_API_OPEN_FAILED;
      }
      bsl_mutex_lock(&m_hid_mutex);
      _send_output_data_cmd(handle, pin_id, 0);
      usleep(2000);
      _send_output_data_cmd(handle, pin_id, 1);
      bsl_mutex_unlock(&m_hid_mutex);

      hidapi_close(handle);
      return BSL_SUCCESS;
    }
  }

  return BSL_ERROR_INVALID_DEVICE_ID;
}

static BSL_STATUS m_hid_discard(int device_id) {
  UNUSED(device_id);
  printf("Device Discard for HID F75114 is not supported\n");
  return BSL_ERROR_UNSUPPORTED_FUNCTION;
}

static BSL_STATUS m_hid_reset_all() {
  if (m_hid_count == 0) {
    printf("no F75114 register, need a scan?\n");
    return BSL_ERROR_NO_HID_DEVICE_FOUND;
  }

  bsl_mutex_lock(&m_hid_mutex);
  for (int i = 0; i < m_hid_count; i++) {
    hidapi_device_s* handle = hidapi_open_path(m_hid_path_list[i]);
    if (handle == NULL) {
      return BSL_ERROR_HID_API_OPEN_FAILED;
    }
    int j = 0;
    for (; j < F75114_MAX_RESET_PIN_NUM; j++) {
      int pin_id = m_hid_reset_pin_map[j].pin_id;
      _send_output_data_cmd(handle, pin_id, 0);
    }
    usleep(2000);
    for (j = 0; j < F75114_MAX_RESET_PIN_NUM; j++) {
      int pin_id = m_hid_reset_pin_map[j].pin_id;
      _send_output_data_cmd(handle, pin_id, 1);
    }
    hidapi_close(handle);
  }
  bsl_mutex_unlock(&m_hid_mutex);
  return BSL_SUCCESS;
}

void log_hidapi_device_info(const struct hidapi_device_info_s* cur_dev) {
  LOG("Device Found:\n");
  LOG("  VID:PID:        %04hx:%04hx\n", cur_dev->vendor_id, cur_dev->product_id);
  LOG("  Path:           %s\n", cur_dev->path);
  LOG("  Serial Number:  %ls\n", cur_dev->serial_number);
  LOG("\n");
  LOG("  Path:           %s\n", cur_dev->path);
  LOG("  Manufacturer:   %ls\n", cur_dev->manufacturer_string);
  LOG("  Product:        %ls\n", cur_dev->product_string);
  LOG("  Release:        %hx\n", cur_dev->release_number);
  LOG("  Interface:      %d\n", cur_dev->interface_number);
  LOG("\n");
}

static BSL_STATUS m_hid_scan_impl(int* p_device_count, const char* parent_path_list[], size_t path_num) {
  m_hid_count = 0;
  printf("scan F75114 device...\n");

  struct hidapi_device_info_s* device_list = hidapi_enumerate(F75114_VID, F75114_PID, parent_path_list, path_num);

  for (struct hidapi_device_info_s* cur_dev = device_list; cur_dev != NULL; cur_dev = cur_dev->next) {
    log_hidapi_device_info(cur_dev);
    bsl_strncpy(m_hid_path_list[m_hid_count], sizeof(m_hid_path_list[m_hid_count]), cur_dev->path,
                sizeof(m_hid_path_list[m_hid_count]));
    m_hid_count++;
  }

  hidapi_free_enumeration(device_list);

  *p_device_count = m_hid_count;
  printf("found %d F75114 device\n", m_hid_count);
  return m_hid_count > 0 ? BSL_SUCCESS : BSL_ERROR_NO_HID_DEVICE_FOUND;
}

static BSL_STATUS m_hid_scan(int* p_device_count) {
  return m_hid_scan_impl(p_device_count, NULL, 0);
}

static BSL_STATUS m_hid_get_device_num(int* p_device_count) {
  if (!p_device_count)
    return BSL_ERROR_MEMORY_ERROR;
  *p_device_count = m_hid_count;
  return BSL_SUCCESS;
}

static void fill_hid_path_array(struct json_object* config, size_t* p_path_count, const char* path_array[]) {
  struct json_object* hid_paths_obj = json_object_object_get(config, "hid_paths");
  if (hid_paths_obj == NULL) {
    return;
  }

  size_t path_count = json_object_array_length(hid_paths_obj);
  if (path_count == 0) {
    *p_path_count = 0;
    return;
  }

  if (path_count > MAX_HID_DEVICE_NUM) {
    path_count = MAX_HID_DEVICE_NUM;
  }

  for (size_t i = 0; i < path_count; i++) {
    struct json_object* path_obj = json_object_array_get_idx(hid_paths_obj, i);
    path_array[i] = json_object_get_string(path_obj);
  }
  *p_path_count = path_count;
}

static BSL_STATUS m_hid_config(struct json_object* config) {
  if (m_hid_count > 0) {
    return BSL_SUCCESS;
  }

  int count = 0;
  size_t addr_len = 0;
  const char* path_array[MAX_HID_DEVICE_NUM];
  memset(path_array, 0, sizeof(path_array));

  fill_hid_path_array(config, &addr_len, path_array);
  return m_hid_scan_impl(&count, path_array, addr_len);
}

void hid_f75114_init(HddlController_t* ctrl) {
  ctrl->device_init = m_hid_init;
  ctrl->device_reset = m_hid_reset;
  ctrl->device_reset_all = m_hid_reset_all;
  ctrl->device_add = NULL;
  ctrl->device_config = m_hid_config;
  ctrl->device_scan = m_hid_scan;
  ctrl->device_get_device_num = m_hid_get_device_num;
  ctrl->device_discard = m_hid_discard;
  ctrl->device_destroy = m_hid_destroy;
}
