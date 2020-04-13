#pragma OPENCL EXTENSION cl_khr_fp16 : enable

ushort extract_weights(uchar val, int bit)
{
    return ((val >> bit) & 1);
}

__kernel void binary_convolution(
                        const __global half*  restrict src_data,
                        const __global uchar* restrict weights_data,
                        const __global half*  restrict dst_data,
                                 float pad_value,

                                       int    IW,
                                       int    IH,
                                       int    IC,

                                       int    DW,
                                       int    DH,

                                       int    GC,

                                       int    KW,
                                       int    KH,

                                       int    PW,
                                       int    PH,

                                       int    SW,
                                       int    SH,

                                       int    OW,
                        const __local  half*  restrict src_local,
                              __local  half*  restrict dst_local)
{
    int oh = get_global_id(0);
    int oc = get_global_id(1);
    int OH = get_global_size(0);
    int OC = get_global_size(1);

    half pad_value_half = convert_half(pad_value);

    //padding row
    if (oh * SH > IH - 1)
    {
        __local half* dst = src_local;
        for(int c = 0; c < IC/GC; c++)
        {
            #pragma unroll 8
            for(int j = 0; j < IW; j++)
            {
                dst[j] = pad_value_half;
            }
            dst += IW;
        }
    } 
  
    int OWS = SW * OW;
    ushort8 in;

    for (int ows8 = 0; ows8 < (OWS+7)/8; ows8++)
    {
        ushort8 val = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int ic = 0; ic < IC/GC; ++ic)
        {
            __local half* src = (__local half*)((__local half8*)(src_local + ic * IW) + ows8);
            int weight_pos = oc * IC/GC + ic;
            ushort w = extract_weights(weights_data[((weight_pos + 0)) / 8], ((weight_pos + 0) % 8));

            if ((ows8 * 8) <= IW - 1)
            {
                in = *((__local ushort8*)(src)); 
            }

            //padding column
            if (ows8 * 8 + 7 > IW - 1)
            {
                int boundary = (IW - 1) - ows8 * 8 + 1;
                boundary = boundary < 0 ? 0 : boundary;
                for (int offset = boundary; offset < 8; offset++)
                {
                    *((half*)(&in) + offset) = pad_value_half;
                }
            }

            ushort8 w8 = (ushort8)(w);

            ushort8 cond = (((in) < (ushort8)0x8000) && (in > (ushort8)0x0000)) ? (ushort8)(1) : (ushort8)(0);
            
            val += (cond ^ w8);
        } 
        
        ushort8 val_shift = val << 1;
        int boundary = (ows8 * 8 + 7) / SW < OW - 1 ? (ows8 * 8 + 7) / SW : OW - 1;
        for (int ow = (ows8 * 8 + SW - 1) / SW; ow <= boundary; ow++)
        {
            *(dst_local + ow) = (half)(IC/GC - *((ushort*)(&val_shift) + ow * SW - ows8 * 8));
        }
    }
}

__kernel void __dma_preload_binary_convolution(
                        const __global half*  restrict src_data,
                        const __global uchar* restrict weights_data,
                        const __global half*  restrict dst_data,
                                 float pad_value,

                                       int    IW,
                                       int    IH,
                                       int    IC,

                                       int    DW,
                                       int    DH,

                                       int    GC,

                                       int    KW,
                                       int    KH,

                                       int    PW,
                                       int    PH,

                                       int    SW,
                                       int    SH,

                                       int    OW,
                              __local  half*  restrict src_local,
                        const __local  half*  restrict dst_local)
{
    const int oh = get_group_id(0);
    const int oc = get_group_id(1);
    const int OC = get_global_size(1);

    const int gc = oc / (OC/GC);

    if (oh * SH >= 0 && oh * SH <= IH - 1)
    {
        const __global half* src = src_data + (gc * IC/GC) * IW * IH + (SH * oh) * IW;
        WorkGroupDmaCreateStrideTransaction(
            src, // src
            src_local, // dst
            IW * sizeof(half), // src width
            IW * sizeof(half), // dst width
            IH * IW * sizeof(half), // src stride
            IW * sizeof(half),  // dst stride
            IW * IC/GC * sizeof(half), //total size
            0
            );  
    }
}
__kernel void __dma_postwrite_binary_convolution(
                        const __global half*  restrict src_data,
                        const __global uchar* restrict weights_data,
                              __global half*  restrict dst_data,
                                 float pad_value,

                                       int    IW,
                                       int    IH,
                                       int    IC,

                                       int    DW,
                                       int    DH,

                                       int    GC,

                                       int    KW,
                                       int    KH,

                                       int    PW,
                                       int    PH,

                                       int    SW,
                                       int    SH,

                                       int    OW,
                        const __local  half*  restrict src_local,
                        const __local  half*  restrict dst_local)
{
    const int oh = get_group_id(0);
    const int oc = get_group_id(1);
    const int OH = get_global_size(0);

    async_work_group_copy(dst_data + oc*OW*OH + oh*OW, dst_local, OW, 0);
}