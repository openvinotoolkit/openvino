// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <iostream>

class ShapeLite {
public:
  ShapeLite(uintptr_t data, int dim);
  uintptr_t get_data();
  int get_dim();
private:
  uintptr_t data;
  int dim;
};
