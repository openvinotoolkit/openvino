/*
* Copyright 2019 Intel Corporation.
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

#ifndef PCIE_HOST_H
#define PCIE_HOST_H

#include "XLinkPlatform.h"

int pcie_init(const char *slot, void **fd);
int pcie_write(void *fd, void * buf, size_t bufSize, int timeout);
int pcie_read(void *fd, void *buf, size_t bufSize, int timeout);
int pcie_close(void *fd);
xLinkPlatformErrorCode_t pcie_find_device_port(int index, char* port_name, int size);

#endif  // PCIE_HOST_H
