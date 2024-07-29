
kernel void vector_add(global int *src1, global int *src2) 
{
  const int id = get_global_id(0);
  src1[id] = src1[id] + src2[id];
}

kernel void local_read_from_remote(global int *src1, global int *src2, int width)
{
  const int id = get_global_id(0);
  printf("local_read_from_remote id: %d, width: %d \n", id, width);
  src1[id] = src2[id];
}

kernel void local_write_to_remote(global int *src1, global int *src2) 
{
  const int id = get_global_id(0);
  src2[id] = src1[id];
}
