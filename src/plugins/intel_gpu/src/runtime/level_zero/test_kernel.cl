
kernel void vector_add(global int *src1, global int *src2) 
{
  const int id = get_global_id(0);
  src1[id] = src1[id] + src2[id];
}

kernel void local_read_from_remote(global int *src1, global int *src2,
  const int src_offset_x, const int src_offset_y,
  const int stride_x, const int stride_y,
  const int width, const int size)
{
  const int gid_x = get_global_id(0);
  const int gid_y = get_global_id(1);

  const int offset_x = src_offset_x + gid_x * stride_x;
  const int offset_y = src_offset_y + gid_y * stride_y;

  const int orig = gid_x + gid_y * width;
  const int offset = offset_x + offset_y * width;

  if (offset < size) {
    src1[orig] = src2[offset];

    if (gid_x < 10 && gid_y < 10) {
      printf("local_read_from_remote gid: (%d,%d), orig/offset: (%d,%d), src/dst: (%d,%d), size: %d \n", gid_x, gid_y, orig, offset, src2[offset], src1[orig], size);
    }
  }
}

// kernel void local_write_to_remote(global unsigned char *src1, global unsigned char *src2,
kernel void local_write_to_remote(global int *src1, global int *src2,
  const int src_offset_x, const int src_offset_y,
  const int stride_x, const int stride_y,
  const int width, const int size)
{
  const int gid_x = get_global_id(0);
  const int gid_y = get_global_id(1);

  const int offset_x = src_offset_x + gid_x * stride_x;
  const int offset_y = src_offset_y + gid_y * stride_y;

  const int orig = gid_x + gid_y * width;
  const int offset = offset_x + offset_y * width;

  if (offset < size) {
    src2[orig] = src1[offset];

    if (gid_x < 10 && gid_y < 10) {
      printf("local_write_to_remote gid: (%d,%d), orig/offset: (%d,%d), src/dst: (%d,%d), size: %d \n", gid_x, gid_y, orig, offset, src1[offset], src2[orig], size);
    }
  }

}
