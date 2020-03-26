// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#ifndef __I2C_LINUX_H__
#define __I2C_LINUX_H__

#include <linux/types.h>
#include <stddef.h>
#include <sys/ioctl.h>

#define FUNCTION_I2C 0x00000001u
#define FUNCTION_SMBUS_PEC 0x00000008u

#define FUNCTION_SMBUS_READ_BYTE 0x00020000u
#define FUNCTION_SMBUS_WRITE_BYTE 0x00040000u
#define FUNCTION_SMBUS_READ_BYTE_DATA 0x00080000u
#define FUNCTION_SMBUS_WRITE_BYTE_DATA 0x00100000u
#define FUNCTION_SMBUS_READ_WORD_DATA 0x00200000u
#define FUNCTION_SMBUS_WRITE_WORD_DATA 0x00400000u

#define FUNCTION_SMBUS_READ_BLOCK_DATA 0x01000000
#define FUNCTION_SMBUS_WRITE_BLOCK_DATA 0x02000000
#define FUNCTION_SMBUS_READ_I2C_BLOCK 0x04000000  /* I2C-like block xfer  */
#define FUNCTION_SMBUS_WRITE_I2C_BLOCK 0x08000000 /* w/ 1-byte reg. addr. */

#define FUNCTION_SMBUS_BYTE (FUNCTION_SMBUS_READ_BYTE | FUNCTION_SMBUS_WRITE_BYTE)
#define FUNCTION_SMBUS_BYTE_DATA (FUNCTION_SMBUS_READ_BYTE_DATA | FUNCTION_SMBUS_WRITE_BYTE_DATA)
#define FUNCTION_SMBUS_WORD_DATA (FUNCTION_SMBUS_READ_WORD_DATA | FUNCTION_SMBUS_WRITE_WORD_DATA)

#define I2C_SMBUS_BLOCK_MAX 32 /* As specified in SMBus standard */

union i2c_smbus_data {
  __u8 byte;
  __u16 word;
  __u8 block[I2C_SMBUS_BLOCK_MAX + 2]; /* block[0] is used for length */
                                       /* and one more for PEC */
};

/* smbus_access read or write markers */
#define I2C_SMBUS_READ 1
#define I2C_SMBUS_WRITE 0

/* SMBus transaction types (size parameter in the above functions)
   Note: these no longer correspond to the (arbitrary) PIIX4 internal codes! */
#define I2C_SMBUS_QUICK 0
#define I2C_SMBUS_BYTE 1
#define I2C_SMBUS_BYTE_DATA 2
#define I2C_SMBUS_WORD_DATA 3
#define I2C_SMBUS_PROC_CALL 4

#define I2C_SLAVE 0x0703 /* Use this slave address */
#define I2C_SLAVE_FORCE 0x0706

#define FUNCTIONS 0x0705 /* Get the adapter functionality mask */

#define I2C_SMBUS 0x0720 /* SMBus transfer */

struct i2c_smbus_ioctl_data {
  __u8 read_write;
  __u8 command;
  __u32 size;
  union i2c_smbus_data* data;
};

#define I2C_RDRW_IOCTL_MAX_MSGS 42

static inline __s32 __hddl_smbus_access(int file, char read_write, __u8 command, int size, union i2c_smbus_data* data) {
  struct i2c_smbus_ioctl_data args;

  args.size = size;
  args.data = data;
  args.read_write = read_write;
  args.command = command;

  return ioctl(file, I2C_SMBUS, &args);
}

static inline __s32 __i2c_smbus_read_byte_data(int file, __u8 command) {
  union i2c_smbus_data data;
  if (__hddl_smbus_access(file, I2C_SMBUS_READ, command, I2C_SMBUS_BYTE_DATA, &data))
    return -1;
  else
    return 0x0FFu & data.byte;
}

static inline __s32 __i2c_smbus_write_byte_data(int file, __u8 command, __u8 value) {
  union i2c_smbus_data data;
  data.byte = value;
  return __hddl_smbus_access(file, I2C_SMBUS_WRITE, command, I2C_SMBUS_BYTE_DATA, &data);
}

static inline __s32 __i2c_smbus_write_quick(int file, __u8 value) {
  return __hddl_smbus_access(file, value, 0, I2C_SMBUS_QUICK, NULL);
}

static inline __s32 __i2c_smbus_read_byte(int file) {
  union i2c_smbus_data data;
  if (__hddl_smbus_access(file, I2C_SMBUS_READ, 0, I2C_SMBUS_BYTE, &data))
    return -1;
  else
    return 0x0FFu & data.byte;
}

#endif