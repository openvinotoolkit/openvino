
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
    if (oh * SH - 1 < 0 || oh * SH - 1 > IH - 1)
    {
        __local half* dst = src_local;
        for(int c = 0; c < IC/GC; c++)
        {
            #pragma unroll 8
            for(int j = 0; j < IW; j++)
            {
                dst[j] = pad_value_half;
            }
            dst += 3 * IW;
        }
    }
    if (oh * SH + DH - 1 > IH - 1)
    {
        __local half* dst = src_local + IW;
        for(int c = 0; c < IC/GC; c++)
        {
            #pragma unroll 8
            for(int j = 0; j < IW; j++)
            {
                dst[j] = pad_value_half;
            }
            dst += 3 * IW;
        }
    }
    if (oh * SH + DH + DH - 1 > IH - 1)
    {
        __local half* dst = src_local + 2 * IW;
        for(int c = 0; c < IC/GC; c++)
        {
            #pragma unroll 8
            for(int j = 0; j < IW; j++)
            {
                dst[j] = pad_value_half;
            }
            dst += 3 * IW;
        }
    }

    int OWS = SW * OW;

    ushort8 in00;
    ushort8 in01;
    ushort8 in02;
    ushort8 in10;
    ushort8 in11;
    ushort8 in12;
    ushort8 in20;
    ushort8 in21;
    ushort8 in22;

    for (int ows8 = 0; ows8 < (OWS+7)/8; ows8++)
    {
        ushort8 val = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int ic = 0; ic < IC/GC; ++ic)
        {
            __local half* src = (__local half*)((__local half8*)(src_local + ic * IW * 3 + IW + DW - 1) + ows8);
            int weight_pos = oc*IC/GC*3*3 + ic*3*3;
            ushort w0 = extract_weights(weights_data[((weight_pos + 0)) / 8], ((weight_pos + 0) % 8));
            ushort w1 = extract_weights(weights_data[((weight_pos + 1)) / 8], ((weight_pos + 1) % 8));
            ushort w2 = extract_weights(weights_data[((weight_pos + 2)) / 8], ((weight_pos + 2) % 8));
            ushort w3 = extract_weights(weights_data[((weight_pos + 3)) / 8], ((weight_pos + 3) % 8));
            ushort w4 = extract_weights(weights_data[((weight_pos + 4)) / 8], ((weight_pos + 4) % 8));
            ushort w5 = extract_weights(weights_data[((weight_pos + 5)) / 8], ((weight_pos + 5) % 8));
            ushort w6 = extract_weights(weights_data[((weight_pos + 6)) / 8], ((weight_pos + 6) % 8));
            ushort w7 = extract_weights(weights_data[((weight_pos + 7)) / 8], ((weight_pos + 7) % 8));
            ushort w8 = extract_weights(weights_data[((weight_pos + 8)) / 8], ((weight_pos + 8) % 8));

            if ((ows8 * 8) - 1 <= IW - 1)
            {
                in00 = *((__local ushort8*)(src - IW - DW));
                in01 = *((__local ushort8*)(src - IW));
                in02 = *((__local ushort8*)(src - IW + DW));

                in10 = *((__local ushort8*)(src - DW));
                in11 = *((__local ushort8*)(src));
                in12 = *((__local ushort8*)(src + DW));

                in20 = *((__local ushort8*)(src + IW - DW));
                in21 = *((__local ushort8*)(src + IW));
                in22 = *((__local ushort8*)(src + IW + DW));
            }

            //padding column
            if (ows8 * 8 - 1 < 0)
            {
                int boundary = 1 - ows8 * 8;
                boundary = boundary > 8 ? 8 : boundary;
                for (int offset = 0; offset < boundary; offset++)
                {
                    *((half*)(&in00) + offset) = pad_value_half;
                    *((half*)(&in10) + offset) = pad_value_half;
                    *((half*)(&in20) + offset) = pad_value_half;
                }
            }            
            if ((ows8 * 8 + 7) + DW + DW - 1 > IW - 1)
            {
                int boundary = (IW - DW - 1 - DW + 1) - ows8 * 8 + 1;
                boundary = boundary < 0 ? 0 : boundary;
                for (int offset = boundary; offset < 8; offset++)
                {
                    *((half*)(&in02) + offset) = pad_value_half;
                    *((half*)(&in12) + offset) = pad_value_half;
                    *((half*)(&in22) + offset) = pad_value_half;
                }
            }            
            if ((ows8 * 8 + 7) + DW - 1 > IW - 1)
            {   
                int boundary = (IW - 1 - DW + 1) - ows8 * 8 + 1;
                boundary = boundary < 0 ? 0 : boundary;
                for (int offset = boundary; offset < 8; offset++)
                {
                    *((half*)(&in01) + offset) = pad_value_half;
                    *((half*)(&in11) + offset) = pad_value_half;
                    *((half*)(&in21) + offset) = pad_value_half;
                }
            }
            if ((ows8 * 8 + 7) - 1 > IW - 1)
            {
                int boundary = (IW - 1 + 1) - ows8 * 8 + 1;
                boundary = boundary < 0 ? 0 : boundary;
                for (int offset = boundary; offset < 8; offset++)
                {
                    *((half*)(&in00) + offset) = pad_value_half;
                    *((half*)(&in10) + offset) = pad_value_half;
                    *((half*)(&in20) + offset) = pad_value_half;
                }
            }

            ushort8 w00 = (ushort8)(w0);
            ushort8 w01 = (ushort8)(w1);
            ushort8 w02 = (ushort8)(w2);
            ushort8 w10 = (ushort8)(w3);
            ushort8 w11 = (ushort8)(w4);
            ushort8 w12 = (ushort8)(w5);
            ushort8 w20 = (ushort8)(w6);
            ushort8 w21 = (ushort8)(w7);
            ushort8 w22 = (ushort8)(w8);

            ushort8 cond0 = (((in00) < (ushort8)0x8000) && (in00 > (ushort8)0x0000)) ? (ushort8)(1) : (ushort8)(0);
            ushort8 cond1 = (((in01) < (ushort8)0x8000) && (in01 > (ushort8)0x0000)) ? (ushort8)(1) : (ushort8)(0);
            ushort8 cond2 = (((in02) < (ushort8)0x8000) && (in02 > (ushort8)0x0000)) ? (ushort8)(1) : (ushort8)(0);
            ushort8 cond3 = (((in10) < (ushort8)0x8000) && (in10 > (ushort8)0x0000)) ? (ushort8)(1) : (ushort8)(0);
            ushort8 cond4 = (((in11) < (ushort8)0x8000) && (in11 > (ushort8)0x0000)) ? (ushort8)(1) : (ushort8)(0);
            ushort8 cond5 = (((in12) < (ushort8)0x8000) && (in12 > (ushort8)0x0000)) ? (ushort8)(1) : (ushort8)(0);
            ushort8 cond6 = (((in20) < (ushort8)0x8000) && (in20 > (ushort8)0x0000)) ? (ushort8)(1) : (ushort8)(0);
            ushort8 cond7 = (((in21) < (ushort8)0x8000) && (in21 > (ushort8)0x0000)) ? (ushort8)(1) : (ushort8)(0);
            ushort8 cond8 = (((in22) < (ushort8)0x8000) && (in22 > (ushort8)0x0000)) ? (ushort8)(1) : (ushort8)(0);
        
            val += (cond0 ^ w00);
            val += (cond1 ^ w01);
            val += (cond2 ^ w02);
            val += (cond3 ^ w10);
            val += (cond4 ^ w11);
            val += (cond5 ^ w12);
            val += (cond6 ^ w20);
            val += (cond7 ^ w21);
            val += (cond8 ^ w22);
        }

        ushort8 val_shift = val << 1;
        int boundary = (ows8 * 8 + 7) / SW <= OW - 1 ? (ows8 * 8 + 7) / SW : OW - 1;
        for (int ow = (ows8 * 8 + SW - 1) / SW; ow <= boundary; ow++)
        {
            *(dst_local + ow) = (half)(IC/GC*KH*KW - *((ushort*)(&val_shift) + ow * SW - ows8 * 8));
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
    const int OH = get_global_size(0);
    const int OC = get_global_size(1);

    const int gc = oc / (OC/GC);

    if (oh * SH - 1 >= 0 && oh * SH + DH + DH - 1 <= IH - 1) //dma for 3 rows
    {
        const __global half* src = src_data + (gc * IC/GC) * IW * IH + (SH * oh - 1) * IW;
        WorkGroupDmaCreate3DTransaction(
            src, //src,
            src_local, //dst,
            IW * sizeof(half), //src width,
            IW * sizeof(half), //dst width,
            DH * IW * sizeof(half), //src stride,
            IW * sizeof(half), //dst stride,
            IC/GC, //num planes //hang when > 256
            IH * IW * sizeof(half), //src plane stride,
            3 * IW * sizeof(half), //dst plane stride,
            3 * IW * sizeof(half), //plane size,
            0
            ); 
  
    }
    else
    {
        int ih = oh * SH - 1;
        if (ih >= 0 && ih <= IH - 1) //dma for first row
        {
            const __global half* src = src_data + (gc * IC/GC) * IW * IH + ih * IW;
            __local half* dst = src_local;
            WorkGroupDmaCreateStrideTransaction(
                src, // src
                dst, // dst
                IW * sizeof(half), // src width
                IW * sizeof(half), // dst width
                IH * IW * sizeof(half), // src stride
                3 * IW * sizeof(half),  // dst stride
                IW * IC/GC * sizeof(half), //total size
                0
                );
        }
        ih = oh * SH - 1 + DH;
        if (ih >= 0 && ih <= IH - 1) //dma for second row
        {
            const __global half* src = src_data + (gc * IC/GC) * IW * IH + ih * IW;
            __local half* dst = src_local + IW;
            WorkGroupDmaCreateStrideTransaction(
                src, // src
                dst, // dst
                IW * sizeof(half), // src width
                IW * sizeof(half), // dst width
                IH * IW * sizeof(half), // src stride
                3 * IW * sizeof(half),  // dst stride
                IW * IC/GC * sizeof(half), //total size
                0
                );
        }
        ih = oh * SH - 1 + 2 * DH;
        if (ih >= 0 && ih <= IH - 1) //dma for third row
        {
            const __global half* src = src_data + (gc * IC/GC) * IW * IH + ih * IW;
            __local half* dst = src_local + 2 * IW;
            WorkGroupDmaCreateStrideTransaction(
                src, // src
                dst, // dst
                IW * sizeof(half), // src width
                IW * sizeof(half), // dst width
                IH * IW * sizeof(half), // src stride
                3 * IW * sizeof(half),  // dst stride
                IW * IC/GC * sizeof(half), //total size
                0
                );
        }
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