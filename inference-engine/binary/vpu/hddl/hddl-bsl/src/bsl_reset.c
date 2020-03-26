// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <assert.h>
#include <errno.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hddl-bsl.h"
#include "hddl_bsl_priv.h"

static void help(void) {
  fprintf(stderr,
          "Usage: bsl_reset -d [io|mcu|hid]\n"
          "                 -i [device_id], reset device id\n"
          "                 -h help\n");
}
/*
#ifndef WIN32

static void print_i2c_busses(void)
{
  struct i2c_adap *adapters;
  int count;

  adapters = i2c_busses_gather();
  if (adapters == NULL) {
    fprintf(stderr, "Error: Out of memory!\n");
    return;
  }
  printf("--------------all i2c/smbus devices------------\n");
  for (count = 0; adapters[count].name; count++) {
    printf("i2c-%d\t%-10s\t%-32s\t%s\n",
      adapters[count].nr, adapters[count].funcs,
      adapters[count].name, adapters[count].algo);
  }

  free_adapters(adapters);
}
#endif
*/

const char* optarg = NULL;
static int bsl_getopt(int argc, char* const* argv, const char* option) {
  UNUSED(option);
  static int index = 1;
  while (index < argc) {
    if (NULL == argv + index)
      return -1;
    const char* str = *(argv + index);
    index++;            // move to next
    if (str[0] == '-')  // it is a option
    {
      if (0 == strcmp(str, "-h")) {
        return 'h';
      } else if ((0 != strcmp(str, "-i") && 0 != strcmp(str, "-d") && 0 != strcmp(str, "-a")) || index >= argc ||
                 *(*(argv + index)) == '-') {
        index = 1;
        return -1;
      }

      else {
        optarg = *(argv + index);
        return (int)str[1];
      }
    }
  }
  index = 1;
  return -1;
}

int main(int argc, char* argv[]) {
  int opt;
  int deviceid = 0xFF;
  bool drop_device = false;

  while (1) {
    opt = bsl_getopt(argc, argv, "i:d:a:h");
    if (opt < 0) {
      break;
    }
    switch (opt) {
      case 'd':
        if (strcmp("mcu", optarg) == 0)
          hddl_set_device(I2C_MCU);
        else if (strcmp("io", optarg) == 0)
          hddl_set_device(I2C_IOEXPANDER);
        else if (strcmp("hid", optarg) == 0)
          hddl_set_device(HID_F75114);
        else {
          printf("%s is not supported\n", optarg);
          exit(0);
        }
        break;
      case 'i':
        deviceid = atoi(optarg);
        break;
      case 'a':
        drop_device = true;
        deviceid = atoi(optarg);
        break;
      case 'h':
        help();
        exit(EXIT_SUCCESS);
    }
  }

  bool device_selected = deviceid != 0xFF;
  int status = 0;

  if (device_selected && drop_device) {
    status = hddl_discard(deviceid);
    goto FunctionCalled;
  }

  if (device_selected && !drop_device) {
    status = hddl_reset(deviceid);
    goto FunctionCalled;
  }

  status = hddl_reset_all();

FunctionCalled:
  if (status) {
    printf("bsl_reset fail for:\t");
    hddl_get_error_string(status);
    return EXIT_FAILURE;
  }
  printf("Success\n");
  return EXIT_SUCCESS;
}
