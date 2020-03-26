// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <Windows.h>

#include <WinIoCtl.h>
#include <iostream>
#include <string>
using namespace std;

#define FILE_DEVICE_HDDLI2C 0x0116
#define IOCTL_HDDLI2C_REGISTER_READ CTL_CODE(FILE_DEVICE_HDDLI2C, 0x700, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define IOCTL_HDDLI2C_REGISTER_WRITE CTL_CODE(FILE_DEVICE_HDDLI2C, 0x701, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define SPBHDDL_I2C_NAME L"HDDLSMBUS"
#define SPBHDDL_I2C_USERMODE_PATH L"\\\\.\\" SPBHDDL_I2C_NAME

static int start_addr = 0x18;
static int end_addr = 0x27;

static void usage(string name) {
  cout << "Usage: " << name << " [s"
       << " Address1"
       << " e Address2]" << endl;
  cout << "\t Address1 - Start device address" << endl;
  cout << "\t Address2 - End device address" << endl;
  cout << "Note: Some known write-only chips may lock SMBus if you scan all device on SMBus." << endl;
  cout << "      Default only scan HDDL card scale address" << endl;
}

int start_scan() {
  HANDLE hDevice = CreateFileW(SPBHDDL_I2C_USERMODE_PATH, GENERIC_READ | GENERIC_WRITE,
                               FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL);

  if (hDevice == INVALID_HANDLE_VALUE) {
    printf("INVALID_HANDLE_VALUE");
    return -1;
  }

  DWORD dwOutput;
  UCHAR outBuff = 0;
  UCHAR data[3] = {0x1f, 0x00, 0xff};  // Address, Register, data
  BOOL ioresult;
  const int row = 8;
  const int col = 16;
  int result[row][col] = {0};
  for (int i = start_addr; i <= end_addr; i++) {
    /* Set slave address */
    data[0] = i;
    ioresult = DeviceIoControl(hDevice, IOCTL_HDDLI2C_REGISTER_READ, data, sizeof(data), &outBuff, sizeof(outBuff),
                               &dwOutput, NULL);

    if (ioresult) {
      result[i / col][i % col] = 1;
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

  // print result table
  printf("     ");
  for (UINT j = 0; j < col; j++)
    printf("%2x  ", j);
  printf("\n");
  for (UINT i = 0; i < row; i++) {
    printf("%2x   ", i * col);
    for (UINT j = 0; j < col; j++) {
      if (result[i][j] == 1) {
        printf("%2x  ", i * col + j);
      } else {
        printf("--  ");
      }
    }
    printf("\n");
  }
  return 0;
}

int main(int argc, char* argv[]) {
  if (argc == 5) {
    if (argv[1][0] == 's') {
      try {
        start_addr = stoi(argv[2], (size_t*)0, 0);
        // cout << "Start address: " << start_addr << endl;
      } catch (const std::exception&) {
        cout << "Error: the start address parameter can't be recognized" << endl;
        goto PRINT_USAGE;
      }
    } else {
      cout << "Error: the start address parameter error, please check it and try again" << endl;
      goto PRINT_USAGE;
    }

    if (argv[3][0] == 'e') {
      try {
        end_addr = stoi(argv[4], (size_t*)0, 0);
        // cout << "End address: " << end_addr << endl;
      } catch (const std::exception&) {
        cout << "Error: the end address parameter can't be recognized" << endl;
      }
    } else {
      cout << "Error: the end address parameter error, please check it and try again" << endl;
      goto PRINT_USAGE;
    }
    if (end_addr > 0x7F) {
      cout << "Error: The address is out of scale" << endl;
      goto PRINT_USAGE;
    }
    if (end_addr < start_addr) {
      cout << "Error: The start address is big" << endl;
      goto PRINT_USAGE;
    }
  }

  if (argc == 1 || argc == 5)
    return start_scan();
PRINT_USAGE:
  usage(argv[0]);
  return -1;
}