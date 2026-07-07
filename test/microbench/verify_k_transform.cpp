// Standalone GPU probe: determine the EXACT (key, head) layout produced by
// intel_sub_group_2d_block_read_transform_8b_32r16x1c when reading K memory
// (row-major [key, head], 1 byte/elem). This tells us whether the 8b transform
// read gives the DPAS-A operand layout (lane=head, elems=keys) WITHOUT a subgroup
// shuffle -- the make-or-break question for Step 2 (see sdpa-ocl-int8-perf memory).
//
// Build: g++ verify_k_transform.cpp -lOpenCL -o /tmp/vk && /tmp/vk
// Pattern planted: K[key*HEAD + head] = (key & 0x1F)*... encoded so we can recover
// both key and head from the byte value. We use value = key*16 + head with HEAD=16
// wide read window, key in 0..31 -> fits in a byte for key<16; to disambiguate all
// 32 keys we split: byteval = ((key & 7) << 4) | head_low ... simpler: just store
// key in high nibble region won't fit. Instead store TWO probes:
//   probe A: value = head   (0..15)  -> reveals which head each lane/byte carries
//   probe B: value = key    (0..31)  -> reveals which key each uint/byte carries
// Run both, cross them.
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>

static const char* SRC = R"CLC(
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void vk(const global uchar* K, int w, int h, int pitch, global int* out) {
    const int lane = get_sub_group_local_id();
    uint vt[8];
    intel_sub_group_2d_block_read_transform_8b_32r16x1c(
        (global void*)K, w, h, pitch, (int2)(0, 0), (private uint*)&vt[0]);
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
        uchar4 b = as_uchar4(vt[u]);
        out[(lane*8 + u)*4 + 0] = b.s0;
        out[(lane*8 + u)*4 + 1] = b.s1;
        out[(lane*8 + u)*4 + 2] = b.s2;
        out[(lane*8 + u)*4 + 3] = b.s3;
    }
}
)CLC";

#define CK(x) do{ cl_int e=(x); if(e){ printf("CL err %d at %s:%d\n", e, __FILE__, __LINE__); exit(1);} }while(0)

int main() {
    const int HEAD = 16;   // read window width (cols)
    const int KEYS = 32;   // 32r
    // K memory: [KEYS x HEAD] bytes, row-major, pitch = HEAD bytes.
    std::vector<unsigned char> Ka(KEYS*HEAD), Kb(KEYS*HEAD);
    for (int key=0; key<KEYS; ++key)
      for (int hd=0; hd<HEAD; ++hd) {
        Ka[key*HEAD+hd] = (unsigned char)hd;    // probe A: value == head
        Kb[key*HEAD+hd] = (unsigned char)key;   // probe B: value == key
      }

    cl_platform_id plats[8]; cl_uint np=0; CK(clGetPlatformIDs(8,plats,&np));
    cl_device_id dev=nullptr;
    for (cl_uint i=0;i<np && !dev;i++){ cl_uint nd=0; if(clGetDeviceIDs(plats[i],CL_DEVICE_TYPE_GPU,1,&dev,&nd)!=CL_SUCCESS) dev=nullptr; }
    if(!dev){ printf("no GPU\n"); return 1; }
    char nm[256]; clGetDeviceInfo(dev,CL_DEVICE_NAME,sizeof(nm),nm,0); printf("Device: %s\n", nm);

    cl_context ctx=clCreateContext(0,1,&dev,0,0,0);
    cl_command_queue q=clCreateCommandQueue(ctx,dev,0,0);
    cl_program prog=clCreateProgramWithSource(ctx,1,&SRC,0,0);
    if(clBuildProgram(prog,1,&dev,"",0,0)!=CL_SUCCESS){
        size_t n; clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,0,0,&n);
        std::vector<char> log(n); clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,n,log.data(),0);
        printf("build log:\n%s\n", log.data()); return 1;
    }
    cl_kernel k=clCreateKernel(prog,"vk",0);

    auto run=[&](std::vector<unsigned char>& Kdata, const char* label){
        cl_mem Kb_=clCreateBuffer(ctx,CL_MEM_COPY_HOST_PTR|CL_MEM_READ_ONLY,Kdata.size(),Kdata.data(),0);
        std::vector<int> out(16*8*4, -1);
        cl_mem Ob=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,out.size()*sizeof(int),0,0);
        int w=HEAD, h=KEYS, pitch=HEAD;
        CK(clSetKernelArg(k,0,sizeof(cl_mem),&Kb_));
        CK(clSetKernelArg(k,1,sizeof(int),&w));
        CK(clSetKernelArg(k,2,sizeof(int),&h));
        CK(clSetKernelArg(k,3,sizeof(int),&pitch));
        CK(clSetKernelArg(k,4,sizeof(cl_mem),&Ob));
        size_t gws=16, lws=16;
        CK(clEnqueueNDRangeKernel(q,k,1,0,&gws,&lws,0,0,0));
        CK(clEnqueueReadBuffer(q,Ob,CL_TRUE,0,out.size()*sizeof(int),out.data(),0,0,0));
        clReleaseMemObject(Kb_); clReleaseMemObject(Ob);
        return out;
    };

    auto A=run(Ka,"head");   // A[(lane*8+u)*4+bb] = head value carried there
    auto B=run(Kb,"key");    // B[...] = key value carried there

    // For the DPAS A operand we want, per key-block of 8 keys: k_raw[mb][off], lane=head.
    // Print the decoded (key,head) at each (lane,u,byte). If lane==head everywhere and
    // the (u,byte)->key mapping is a clean linear order, NO shuffle is needed.
    printf("\n(lane,u,byte) -> (key=B, head=A):  [expect head==lane if lane carries head]\n");
    bool lane_is_head = true;
    for (int lane=0; lane<16; ++lane) {
      for (int u=0; u<8; ++u) {
        for (int bb=0; bb<4; ++bb) {
          int idx=(lane*8+u)*4+bb;
          int hd=A[idx], key=B[idx];
          if (hd != lane) lane_is_head=false;
          if (lane<2 && u<8) // print first 2 lanes fully to see the key order
            printf("  L%02d u%d b%d -> key=%2d head=%2d\n", lane,u,bb,key,hd);
        }
      }
    }
    printf("\nlane==head for ALL (lane,u,byte)? %s\n", lane_is_head ? "YES (no shuffle needed for A operand)" : "NO");

    // Show, for lane 0, the sequence of keys across u=0..7,byte=0..3 (32 keys):
    printf("lane0 key order (u0b0..u7b3): ");
    for (int u=0;u<8;u++) for(int bb=0;bb<4;bb++) printf("%d ", B[(0*8+u)*4+bb]);
    printf("\n");
    return 0;
}
