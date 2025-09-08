// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_headers/fetch_data.cl"

inline uint FUNC(get_input_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint v, uint u, uint w, uint z, uint y, uint x) __attribute__((overloadable)) {
#if INPUT0_DIMS < 5
    return INPUT0_GET_INDEX(b, f, y, x);
#elif INPUT0_DIMS == 5
    return INPUT0_GET_INDEX(b, f, z, y, x);
#elif INPUT0_DIMS == 6
    return INPUT0_GET_INDEX(b, f, w, z, y, x);
#elif INPUT0_DIMS == 7
    return INPUT0_GET_INDEX(b, f, u, w, z, y, x);
#elif INPUT0_DIMS == 8
    return INPUT0_GET_INDEX(b, f, v, u, w, z, y, x);
#else
#error [GPU] Unsupported input tensor rank in get_input_index function
#endif
}

inline uint FUNC(get_input_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) __attribute__((overloadable)) {
    return FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, 0, 0, w, z, y, x);
}

inline uint FUNC(get_output_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint v, uint u, uint w, uint z, uint y, uint x) __attribute__((overloadable)) {
#if OUTPUT_DIMS < 5
    return OUTPUT_GET_INDEX(b, f, y, x);
#elif OUTPUT_DIMS == 5
    return OUTPUT_GET_INDEX(b, f, z, y, x);
#elif OUTPUT_DIMS == 6
    return OUTPUT_GET_INDEX(b, f, w, z, y, x);
#elif OUTPUT_DIMS == 7
    return OUTPUT_GET_INDEX(b, f, u, w, z, y, x);
#elif OUTPUT_DIMS == 8
    return OUTPUT_GET_INDEX(b, f, v, u, w, z, y, x);
#else
#error [GPU] Unsupported output tensor rank in get_output_index function
#endif
}

inline uint FUNC(get_output_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) __attribute__((overloadable)) {
    return FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, 0, 0, w, z, y, x);
}
