// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//



#include <stdio.h>

#include <emscripten.h>

EMSCRIPTEN_KEEPALIVE
void sayHi() {
    printf("Hi!\n");
}

EMSCRIPTEN_KEEPALIVE
int daysInWeek() {
    return 7;
}
