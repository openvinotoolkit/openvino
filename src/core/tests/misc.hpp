// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <stdio.h>
#include <stdlib.h>

FILE* port_open(const char* command, const char* type);
int port_close(FILE* stream);
int set_environment(const char* name, const char* value, int overwrite);
int unset_environment(const char* name);
