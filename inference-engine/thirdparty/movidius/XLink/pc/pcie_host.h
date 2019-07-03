// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef PCIE_HOST_H
#define PCIE_HOST_H

#include "XLinkPlatform.h"

int pcie_init(const char *slot, void **fd);
int pcie_write(void *fd, void * buf, size_t bufSize, int timeout);
int pcie_read(void *fd, void *buf, size_t bufSize, int timeout);
int pcie_close(void *fd);
xLinkPlatformErrorCode_t pcie_find_device_port(int index, char* port_name, int size);

#endif  // PCIE_HOST_H
