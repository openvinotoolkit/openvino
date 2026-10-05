// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// N, SPATIAL_SIZE, NUM_HEADS, EMBED_DIMS, NUM_LEVELS, NUM_QUERY and NUM_POINT
// are checked against plain input buffer element counts and supplied by the host
// as JIT constants. The plugin may collapse the original 6-D/5-D input ranks.

inline INPUT0_TYPE FUNC(ms_deform_attn_im2col_bilinear)(
    __global const INPUT0_TYPE *bottom_data, const int height, const int width,
    const int nheads, const int ed, const INPUT0_TYPE h,
    const INPUT0_TYPE w, const int m, const int c) {
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const INPUT0_TYPE lh = h - h_low;
  const INPUT0_TYPE lw = w - w_low;
  const INPUT0_TYPE hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * ed;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * ed + c;

  INPUT0_TYPE v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  INPUT0_TYPE v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  INPUT0_TYPE v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  INPUT0_TYPE v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }

  const INPUT0_TYPE w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const INPUT0_TYPE val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

KERNEL(multi_scale_deformable_attn)(
    OPTIONAL_SHAPE_INFO_ARG
    __global const INPUT0_TYPE *data_value,            //# (bs, num_keys, NUM_HEADS, EMBED_DIMS)
    __global const int *data_spatial_shapes,        //# (NUM_LEVELS, 2) Spatial shape of each feature map, last dimension 2 represent (h, w)
    __global const int *data_level_start_index,     //# (NUM_LEVELS, ) start index of each level and can be represented as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
    __global const INPUT0_TYPE *data_sampling_loc,     //# (bs ,num_queries, NUM_HEADS, NUM_LEVELS, num_points, 2), the last dimension 2 represent (x, y).
    __global const INPUT0_TYPE *data_attn_weight,      //# (bs ,num_queries, NUM_HEADS, NUM_LEVELS, num_points), weight of sampling points
    __global OUTPUT_TYPE *output) {                  //# (bs, num_queries, NUM_HEADS * EMBED_DIMS), output
  {
    int index = get_global_id(2);
    if (index >= N)
      return;

    int _temp = index;
    const int c_col = _temp % EMBED_DIMS;
    _temp /= EMBED_DIMS;
    const int sampling_index = _temp;
    const int m_col = _temp % NUM_HEADS;
    _temp /= NUM_HEADS;
    _temp /= NUM_QUERY;
    const int b_col = _temp;

    __global INPUT0_TYPE *data_col_ptr = output + index;
    int data_weight_ptr = sampling_index * NUM_LEVELS * NUM_POINT;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int qid_stride = NUM_HEADS * EMBED_DIMS;
    const int data_value_ptr_init_offset = b_col * SPATIAL_SIZE * qid_stride;
    INPUT0_TYPE col = 0;

    for (int l_col = 0; l_col < NUM_LEVELS; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      __global const INPUT0_TYPE *data_value_ptr =
          data_value +
          (data_value_ptr_init_offset + level_start_id * qid_stride);
      for (int p_col = 0; p_col < NUM_POINT; ++p_col) {
        const INPUT0_TYPE loc_w = data_sampling_loc[data_loc_w_ptr];
        const INPUT0_TYPE loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const INPUT0_TYPE weight = data_attn_weight[data_weight_ptr];

        const INPUT0_TYPE h_im = loc_h * spatial_h - 0.5;
        const INPUT0_TYPE w_im = loc_w * spatial_w - 0.5;

        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          col += FUNC_CALL(ms_deform_attn_im2col_bilinear)(data_value_ptr, spatial_h,
                                                spatial_w, NUM_HEADS, EMBED_DIMS,
                                                h_im, w_im, m_col, c_col) *
                 weight;
        }

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
    }
    *data_col_ptr = col;
  }
}