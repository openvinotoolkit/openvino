// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>

class ShapeLite {
public:
    ShapeLite(uintptr_t data, int dim);
    ShapeLite(ov::Shape* shape);

    uintptr_t get_data();
    int get_dim();
    int shape_size();
    ov::Shape get_original();

private:
    ov::Shape shape;
};
