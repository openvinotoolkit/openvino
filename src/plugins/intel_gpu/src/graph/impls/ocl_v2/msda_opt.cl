#include "include/batch_headers/fetch_data.cl"

#define n INPUT0_BATCH_NUM * INPUT1_BATCH_NUM * INPUT0_SIZE_Y * INPUT0_SIZE_X
#define batch_size INPUT0_BATCH_NUM
#define spatial_size INPUT0_FEATURE_NUM
#define num_heads INPUT0_SIZE_Y
#define channels INPUT0_SIZE_X
#define num_levels INPUT1_BATCH_NUM
#define num_query INPUT3_FEATURE_NUM
#define num_point INPUT3_SIZE_Y 

INPUT0_TYPE ms_deform_attn_im2col_bilinear(
    __global const INPUT0_TYPE *bottom_data, const int height, const int width,
    const int nheads, const int embed_dims, const INPUT0_TYPE h,
    const INPUT0_TYPE w, const int m, const int c) {
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const INPUT0_TYPE lh = h - h_low;
  const INPUT0_TYPE lw = w - w_low;
  const INPUT0_TYPE hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * embed_dims;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * embed_dims + c;

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
// #ifndef MS_DEFORM_ATTN_IM2COL_MACRO
// #define MS_DEFORM_ATTN_IM2COL_MACRO

// #define MS_DEFORM_ATTN_IM2COL_BILINEAR(bottom_data, height, width, nheads, ch, h, w, m, c, result)    \
// {                                                                                                     \
//     int h_low = floor(h);                                                                            \
//     int w_low = floor(w);                                                                            \
//     int h_high = h_low + 1;                                                                          \
//     int w_high = w_low + 1;                                                                          \
//                                                                                                      \
//     INPUT0_TYPE lh = h - h_low;                                                                      \
//     INPUT0_TYPE lw = w - w_low;                                                                      \
//     INPUT0_TYPE hh = 1 - lh, hw = 1 - lw;                                                            \
//                                                                                                      \
//     int w_stride = nheads * ch;                                                                      \
//     int h_stride = width * w_stride;                                                                 \
//     int h_low_ptr_offset = h_low * h_stride;                                                         \
//     int h_high_ptr_offset = h_low_ptr_offset + h_stride;                                             \
//     int w_low_ptr_offset = w_low * w_stride;                                                         \
//     int w_high_ptr_offset = w_low_ptr_offset + w_stride;                                             \
//     int base_ptr = m * ch + c;                                                                       \
//                                                                                                      \
//     INPUT0_TYPE v1 = 0;                                                                              \
//     if (h_low >= 0 && w_low >= 0) {                                                                  \
//         int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;                                  \
//         v1 = bottom_data[ptr1];                                                                      \
//     }                                                                                                \
//     INPUT0_TYPE v2 = 0;                                                                              \
//     if (h_low >= 0 && w_high <= width - 1) {                                                         \
//         int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;                                 \
//         v2 = bottom_data[ptr2];                                                                      \
//     }                                                                                                \
//     INPUT0_TYPE v3 = 0;                                                                              \
//     if (h_high <= height - 1 && w_low >= 0) {                                                        \
//         int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;                                 \
//         v3 = bottom_data[ptr3];                                                                      \
//     }                                                                                                \
//     INPUT0_TYPE v4 = 0;                                                                              \
//     if (h_high <= height - 1 && w_high <= width - 1) {                                               \
//         int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;                                \
//         v4 = bottom_data[ptr4];                                                                      \
//     }                                                                                                \
//                                                                                                      \
//     INPUT0_TYPE w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;                              \
//                                                                                                      \
//     result = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);                                                 \
// }

// #endif  // MS_DEFORM_ATTN_IM2COL_MACRO

// KERNEL(multi_scale_deformable_attn)(
//     const int n,
//     __global const INPUT0_TYPE *data_value,            //# (bs, num_keys, num_heads, channels)
//     __global const int *data_spatial_shapes,        //# (num_levels, 2) Spatial shape of each feature map, last dimension 2 represent (h, w)
//     __global const int *data_level_start_index,     //# (num_levels, ) start index of each level and can be represented as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
//     __global const INPUT0_TYPE *data_sampling_loc,     //# (bs ,num_queries, num_heads, num_levels, num_points, 2), the last dimension 2 represent (x, y).
//     __global const INPUT0_TYPE *data_attn_weight,      //# (bs ,num_queries, num_heads, num_levels, num_points), weight of sampling points
//     const int batch_size,
//     const int spatial_size, const int num_heads, const int channels,
//     const int num_levels, const int num_query, const int num_point,
//     __global INPUT0_TYPE *data_col) {                  //# (bs, num_keys, num_heads, channels), output

KERNEL(multi_scale_deformable_attn)(
    OPTIONAL_SHAPE_INFO_ARG
    __global const INPUT0_TYPE *data_value,            //# (bs, num_keys, num_heads, channels)
    __global const int *data_spatial_shapes,        //# (num_levels, 2) Spatial shape of each feature map, last dimension 2 represent (h, w)
    __global const int *data_level_start_index,     //# (num_levels, ) start index of each level and can be represented as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
    __global const INPUT0_TYPE *data_sampling_loc,     //# (bs ,num_queries, num_heads, num_levels, num_points, 2), the last dimension 2 represent (x, y).
    __global const INPUT0_TYPE *data_attn_weight,      //# (bs ,num_queries, num_heads, num_levels, num_points), weight of sampling points
    __global INPUT0_TYPE *data_col) {                  //# (bs, num_keys, num_heads, channels), output
#define sglid          (uint) get_sub_group_local_id()
#define sgid           (uint) get_sub_group_id()
  // CUDA_1D_KERNEL_LOOP(index, n) {
  // for (int index = 0; index < n; index++)
  {
    if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0) {
    printf("batch_size=%d, spatial_size=%d, num_heads=%d, embed_dims=%d, num_levels=%d, num_query=%d, num_point=%d\n",
        batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point);
    }
if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0) {
    // INPUT0: data_value [B, Lv, H, C]
    printf("INPUT0 (value): shape = [%d, %d, %d, %d]\n", INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_Y, INPUT0_SIZE_X);

    // INPUT1: data_spatial_shapes [L, 2]
    printf("INPUT1 (spatial_shapes): shape = [%d, %d, %d, %d]\n", INPUT1_BATCH_NUM, INPUT1_FEATURE_NUM, INPUT1_SIZE_Y, INPUT1_SIZE_X);

    // INPUT2: level_start_index [L]
    printf("INPUT2 (level_start_index): shape = [%d, %d, %d, %d]\n", INPUT2_BATCH_NUM, INPUT2_FEATURE_NUM, INPUT2_SIZE_Y, INPUT2_SIZE_X);

    // INPUT3: sampling_locations [B, Q, H, L, P, 2]
    printf("INPUT3 (sampling_locations): shape = [%d, %d, %d, %d, %d, %d]\n",
           INPUT3_BATCH_NUM, INPUT3_FEATURE_NUM, INPUT3_SIZE_W, INPUT3_SIZE_Z, INPUT3_SIZE_Y, INPUT3_SIZE_X);

    // INPUT4: attn_weights [B, Q, H, L, P]
    printf("INPUT4 (attn_weights): shape = [%d, %d, %d, %d, %d]\n",
           INPUT4_BATCH_NUM, INPUT4_FEATURE_NUM, INPUT4_SIZE_Z, INPUT4_SIZE_Y, INPUT4_SIZE_X);

    // OUTPUT0: result [B, Q, H*C]
    printf("OUTPUT0 (data_col): shape = [%d, %d, %d, %d]\n", OUTPUT_BATCH_NUM, OUTPUT_FEATURE_NUM, OUTPUT_SIZE_Y, OUTPUT_SIZE_X);
}

    int index = get_global_id(2);
    // printf("[%ld]][%ld][%d][%d] indx = %d/%d\n", get_group_id(2), get_local_id(2), sgid, sglid, index, n);

    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    __global INPUT0_TYPE *data_col_ptr = data_col + index;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    INPUT0_TYPE col = 0;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      __global const INPUT0_TYPE *data_value_ptr =
          data_value +
          (data_value_ptr_init_offset + level_start_id * qid_stride);
      for (int p_col = 0; p_col < num_point; ++p_col) {
        const INPUT0_TYPE loc_w = data_sampling_loc[data_loc_w_ptr];
        const INPUT0_TYPE loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const INPUT0_TYPE weight = data_attn_weight[data_weight_ptr];

        const INPUT0_TYPE h_im = loc_h * spatial_h - 0.5;
        const INPUT0_TYPE w_im = loc_w * spatial_w - 0.5;

        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          // INPUT0_TYPE bilinear_val = 0;
          // MS_DEFORM_ATTN_IM2COL_BILINEAR(data_value_ptr, spatial_h, spatial_w, num_heads, channels,
          //                                h_im, w_im, m_col, c_col, bilinear_val);
          INPUT0_TYPE bilinear_val = ms_deform_attn_im2col_bilinear(data_value_ptr, spatial_h, spatial_w, num_heads, channels,
                                         h_im, w_im, m_col, c_col);
          INPUT0_TYPE a = bilinear_val * weight;
          col += a;
        }

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
    }
    *data_col_ptr = col;
  }
}
