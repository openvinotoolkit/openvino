// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef PCIE_HOST_H
#define PCIE_HOST_H

#include "XLinkPlatform.h"

int pcie_init(const char *slot, void **fd);
int pcie_write(void *fd, void * buf, size_t bufSize, unsigned int timeout_ms);
int pcie_read(void *fd, void *buf, size_t bufSize, unsigned int timeout_ms);
int pcie_close(void *fd);
xLinkPlatformErrorCode_t pcie_find_device_port(int index, char* port_name, int size);
int pcie_reset_device(int fd);
int pcie_boot_device(int fd, void *buffer, size_t length);

#endif  // PCIE_HOST_H
