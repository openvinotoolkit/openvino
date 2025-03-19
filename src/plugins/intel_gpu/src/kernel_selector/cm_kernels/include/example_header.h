// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

inline void print_lws_gws() {
    printf("lws: %d, %d, %d\n", cm_local_size(0), cm_local_size(1), cm_local_size(2));
    printf("gws: %d, %d, %d\n", cm_group_count(0), cm_group_count(1), cm_group_count(2));
}
