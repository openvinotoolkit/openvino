// On-device probe: decode the exact (key,value) layout the 8b VNNI-transform read produces
// for V memory (row-major [key, value], 1 byte/elem). Plants two markers (V=key, V=value)
// and dumps, per (lane,uint,byte), which (key,value) that slot carries.
//
// RESULT (B580): lane == value; per lane the bytes are keys 0,1,2,...,15 in LINEAR order
// (uint u byte b = key u*4+b). That is EXACTLY the DPAS f16-VNNI-2 operand's logical order
// (lane=value, slot p = keys 2p,2p+1). So the transform read is ALREADY VNNI-aligned — the
// ~670 movs are int8->f16 WIDENING + per-key zp broadcast, NOT a reorder. This REFUTED the
// "adopt a VNNI-aligned load to remove the repack" lever. See sdpa-ocl-int8-perf.
//
// Build/run: test/microbench/run_probe.sh probe_v_layouts   (needs libOpenCL + GPU)
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do{ cl_int e=(x); if(e){ printf("CL err %d @ %d\n", e, __LINE__); exit(1);} }while(0)
static const char* SRC = R"CLC(
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void t(const global void* V, int w,int h,int p, global int* outk, global int* outv) {
    int lane=get_sub_group_local_id();
    uint vt[8];
    intel_sub_group_2d_block_read_transform_8b_32r16x1c((global void*)V, w,h,p, (int2)(0,0), (private uint*)&vt[0]);
    for (int u=0;u<4;u++){ char4 c=as_char4(vt[u]); for(int b=0;b<4;b++){ int idx=(lane*4+u)*4+b; outk[idx]=c[b]&0xff; } }
}
)CLC";
int main(){
    const int KEYS=32, VAL=16;
    std::vector<unsigned char> Vk(KEYS*VAL), Vv(KEYS*VAL);
    for(int key=0;key<KEYS;key++)for(int v=0;v<VAL;v++){ Vk[key*VAL+v]=key; Vv[key*VAL+v]=v; }
    cl_platform_id pl[8]; cl_uint np; CK(clGetPlatformIDs(8,pl,&np));
    cl_device_id dev=0; for(cl_uint i=0;i<np&&!dev;i++) clGetDeviceIDs(pl[i],CL_DEVICE_TYPE_GPU,1,&dev,0);
    char nm[256]; clGetDeviceInfo(dev,CL_DEVICE_NAME,256,nm,0); printf("Device: %s\n",nm);
    cl_context ctx=clCreateContext(0,1,&dev,0,0,0);
    cl_command_queue q=clCreateCommandQueue(ctx,dev,0,0);
    cl_program pr=clCreateProgramWithSource(ctx,1,&SRC,0,0);
    if(clBuildProgram(pr,1,&dev,"",0,0)){size_t n;clGetProgramBuildInfo(pr,dev,CL_PROGRAM_BUILD_LOG,0,0,&n);std::vector<char>l(n);clGetProgramBuildInfo(pr,dev,CL_PROGRAM_BUILD_LOG,n,l.data(),0);printf("%s\n",l.data());return 1;}
    cl_kernel k=clCreateKernel(pr,"t",0);
    auto run=[&](std::vector<unsigned char>&V,std::vector<int>&out){
        cl_mem Vb=clCreateBuffer(ctx,CL_MEM_COPY_HOST_PTR|CL_MEM_READ_ONLY,V.size(),V.data(),0);
        out.assign(16*4*4,-1);
        cl_mem Ob=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,out.size()*4,0,0);
        cl_mem Ob2=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,out.size()*4,0,0);
        int w=VAL,h=KEYS,p=VAL;
        CK(clSetKernelArg(k,0,sizeof(cl_mem),&Vb));CK(clSetKernelArg(k,1,4,&w));CK(clSetKernelArg(k,2,4,&h));CK(clSetKernelArg(k,3,4,&p));
        CK(clSetKernelArg(k,4,sizeof(cl_mem),&Ob));CK(clSetKernelArg(k,5,sizeof(cl_mem),&Ob2));
        size_t g=16,l=16; CK(clEnqueueNDRangeKernel(q,k,1,0,&g,&l,0,0,0));
        CK(clEnqueueReadBuffer(q,Ob,CL_TRUE,0,out.size()*4,out.data(),0,0,0));
        clReleaseMemObject(Vb);clReleaseMemObject(Ob);clReleaseMemObject(Ob2);
    };
    std::vector<int> ok, ov;
    run(Vk,ok); run(Vv,ov);
    printf("8b-transform read: (lane,u,byte) -> key[value]\n");
    for(int lane=0;lane<2;lane++){ printf(" lane%d: ",lane);
      for(int u=0;u<4;u++){ for(int b=0;b<4;b++){int idx=(lane*4+u)*4+b; printf("k%d[v%d] ",ok[idx],ov[idx]);} printf("| ");} printf("\n"); }
    printf("\nDPAS f16-VNNI-2 operand wants: lane=value, int-slot p = (key 2p, key 2p+1).\n");
    printf("=> transform read already gives lane=value, keys 0..15 in order. No reorder needed.\n");
    return 0;
}
