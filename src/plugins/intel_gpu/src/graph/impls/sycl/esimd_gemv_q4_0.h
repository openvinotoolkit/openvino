#include <ext/intel/esimd.hpp>
using fp16 = ::sycl::half;

using namespace ::sycl::ext::intel::esimd;
using namespace ::sycl;
using namespace ::sycl::ext::intel::esimd;
using namespace ::sycl::ext::intel::esimd::xmx;

template <uint32_t pixelPerGroupShift>
ESIMD_INLINE void matrixMulCommonDim4096Int4NoReshapeNx16V3_ipex2(
    uint8_t* a,
    uint8_t* b,
    uint8_t* c,
    uint8_t* d,
    nd_item<1>& ndi) {

  constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
  constexpr uint32_t quantPerGroup = 4096 / 32 * pixelPerGroup;
  // constexpr uint32_t baseOffsetInc16[16] = {
  //     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  constexpr uint32_t baseOffsetInc8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  __ESIMD_NS::slm_init(16 * 16 * sizeof(float));
  int hh = ndi.get_local_id(0); // 0-16
  int h = ndi.get_group(0); // [0, 256)
  // int rowSize = ndi.get_group_range(0) * pixelPerGroup;
  // weight offset
  int offsetABase = (h * pixelPerGroup * 4096 + hh * 64) >> 1;
  // scale offset
  int offsetQuanBase = h * quantPerGroup * sizeof(fp16) + hh * 2 * sizeof(fp16);
  // input offset
  int offsetB = hh * 64 * sizeof(fp16);
  // output offset
  int outputOffset = pixelPerGroup * h;
  simd<unsigned char, 128> aaa;
  simd<fp16, 16> quant;
  simd<float, 8> fp32Quant;
  simd<float, 256> bb;
  simd<fp16, 256> bb_fp16;
  simd<float, 16 * 16> aa;
  simd<float, 16> cc(0.0f);
  simd<uint32_t, 8> offsetA(baseOffsetInc8);
  simd<uint32_t, 8> offsetQuan(baseOffsetInc8);
  simd_mask<8> quantPred = 1;
  quantPred[4] = 0;
  quantPred[5] = 0;
  quantPred[6] = 0;
  quantPred[7] = 0;
  // gather read 8, 4*8=32 byte=64 int4
  offsetA = offsetA * 4 + offsetABase;
  // 4096/32 (chunk size)/4 (4 split per thread) = 32 stride
  offsetQuan = offsetQuan * 32 * sizeof(fp16) + offsetQuanBase;

// 4 times for 1024*4
#pragma unroll
  for (int k = 0; k < 4; k++) {
    // 128 uint8_t=64 fp16, 128*k is uint8 offset at bb_fp16, offsetB is input offset
    bb_fp16.template bit_cast_view<uint8_t>().template select<128, 1>(128 * k) =
    // block load
        __ESIMD_ENS::lsc_block_load<
            uint8_t,
            128,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);

    offsetB += 1024 * sizeof(fp16);
  }
  // to fp32
  bb.select<256, 1>(0) = bb_fp16.select<256, 1>(0);

  //
  for (int n = 0; n < pixelPerGroup; n++) {
    cc = 0.0f;
    // 8 gather read, only use first 4 element (2 fp16)
    quant.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)d, offsetQuan, quantPred);

    aaa.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)a, offsetA);
    offsetA += 512;

    aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)a, offsetA);
    offsetA += 512; // 2048 - 16 * sizeof(uint32_t)

    aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8 * 2) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)a, offsetA);
    offsetA += 512; // 2048 - 16 * sizeof(uint32_t)

    aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8 * 3) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)a, offsetA);
    offsetA += 512; // 2048 - 16 * sizeof(uint32_t)

#pragma unroll
    for (int k = 0; k < 8; k++) {
      aa.select<16, 2>(32 * k) = aaa.select<16, 1>(16 * k) & 0x0F;
      aa.select<16, 2>(32 * k + 1) = (aaa.select<16, 1>(16 * k) & 0xF0) >> 4;
    }

    aa = aa - 8.0f;
    fp32Quant = quant.select<8, 1>(0);

#pragma unroll
    for (int k = 0; k < 8; k++) {
      aa.select<32, 1>(32 * k) = fp32Quant[k] * aa.select<32, 1>(32 * k);
    }

#pragma unroll
    for (int k = 0; k < 16; k++) {
      cc += aa.select<16, 1>(16 * k) * bb.select<16, 1>(16 * k);
    }

    cc.select<8, 1>(0) += cc.select<8, 1>(8);
    cc.select<4, 1>(0) += cc.select<4, 1>(4);
    cc.select<2, 1>(0) += cc.select<2, 1>(2);
    simd<float, 1> slmAccumulationTemp = cc[0] + cc[1];
    uint32_t slmAccumulationOffset = (hh * pixelPerGroup + n) * sizeof(float);
    // slm_scalar_store(slmAccumulationOffset, slmAccumulationTemp);
    slm_block_store<float, 1>(slmAccumulationOffset, slmAccumulationTemp);
    offsetQuan += 128 * sizeof(fp16);
  }
  barrier();

  if (hh == 0) {
    if constexpr (pixelPerGroupShift == 4) {
#pragma unroll
      for (int k = 0; k < 4; k++) {
        bb.select<64, 1>(64 * k) =
            slm_block_load<float, 64>(64 * k * sizeof(float));
      }
#pragma unroll
      for (int k = 1; k < 16; k++) {
        bb.select<16, 1>(0) += bb.select<16, 1>(16 * k);
      }

    } else if constexpr (pixelPerGroupShift == 3) {
#pragma unroll
      for (int k = 0; k < 2; k++) {
        bb.select<64, 1>(64 * k) =
            slm_block_load<float, 64>(64 * k * sizeof(float));
      }
#pragma unroll
      for (int k = 1; k < 8; k++) {
        bb.select<16, 1>(0) += bb.select<16, 1>(16 * k);
      }
      bb.select<8, 1>(0) += bb.select<8, 1>(8);
    } else if constexpr (pixelPerGroupShift == 2) {
      bb.select<64, 1>(0) = slm_block_load<float, 64>(0);
#pragma unroll
      for (int k = 1; k < 4; k++) {
        bb.select<16, 1>(0) += bb.select<16, 1>(16 * k);
      }
      bb.select<8, 1>(0) += bb.select<8, 1>(8);
      bb.select<4, 1>(0) += bb.select<4, 1>(4);
    } else if constexpr (pixelPerGroupShift == 1) {
      bb.select<32, 1>(0) = slm_block_load<float, 32>(0);
      bb.select<16, 1>(0) += bb.select<16, 1>(16 * 1);
      bb.select<8, 1>(0) += bb.select<8, 1>(8);
      bb.select<4, 1>(0) += bb.select<4, 1>(4);
      bb.select<2, 1>(0) += bb.select<2, 1>(2);
    } else if constexpr (pixelPerGroupShift == 0) {
      bb.select<16, 1>(0) = slm_block_load<float, 16>(0);
      bb.select<8, 1>(0) += bb.select<8, 1>(8);
      bb.select<4, 1>(0) += bb.select<4, 1>(4);
      bb.select<2, 1>(0) += bb.select<2, 1>(2);
      bb.select<1, 1>(0) += bb.select<1, 1>(1);
    }

    bb_fp16.select<pixelPerGroup, 1>(0) = bb.select<pixelPerGroup, 1>(0);

    __ESIMD_ENS::lsc_block_store<
        fp16,
        pixelPerGroup,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>(
        (fp16*)c + outputOffset, bb_fp16.select<pixelPerGroup, 1>(0));
  }
}


template <uint32_t pixelPerGroupShift>
ESIMD_INLINE void matrixMulCommonDim11008Int4NoReshapeNx16V2_ipex(
    uint8_t* a,
    uint8_t* b,
    uint8_t* c,
    uint8_t* d,
    nd_item<1>& ndi) {
  constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
  constexpr uint32_t quantPerGroup = 11008 / 32 * pixelPerGroup;
  constexpr uint32_t sumThreads = pixelPerGroup / 16;
  // constexpr uint32_t baseOffsetInc16[16] = {
  //     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  constexpr uint32_t baseOffsetInc8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  __ESIMD_NS::slm_init(16 * 128 * sizeof(float));
  int hh = ndi.get_local_id(0); // [0, 64)
  int h = ndi.get_group(0); // [0, 256)
  int rowSize = ndi.get_group_range(0) * pixelPerGroup;
  int offsetABase = (h * pixelPerGroup * 11008 + hh * 8 * 8) >> 1;
  int offsetQuanBase = /*rowSize * 5504 +*/ h * quantPerGroup * sizeof(fp16) +
      hh * 2 * sizeof(fp16);
  int offsetB = hh * 64 * sizeof(fp16);
  int outputOffset = pixelPerGroup * h;
  simd<uint8_t, 64> aaa;
  simd<fp16, 32> quant;
  simd<fp16, 704> bb;
  simd<float, 256> bb_f32;
  simd<float, 8 * 16> aa;
  simd<float, 16> cc(0.0f);
  simd<uint32_t, 8> offsetA(baseOffsetInc8);
  simd<uint32_t, 8> offsetQuan(baseOffsetInc8);
  offsetA = offsetA * sizeof(uint32_t) + offsetABase;
  offsetQuan = offsetQuan * 32 * sizeof(fp16) + offsetQuanBase;

#pragma unroll
  for (int k = 0; k < 10; k++) {
    bb.template bit_cast_view<unsigned char>().template select<128, 1>(
        128 * k) =
        __ESIMD_ENS::lsc_block_load<
            uint8_t,
            128,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);

    offsetB += 1024 * sizeof(fp16);
  }

  if (hh < 12) {
    bb.template bit_cast_view<unsigned char>().template select<128, 1>(
        128 * 10) =
        __ESIMD_ENS::lsc_block_load<
            uint8_t,
            128,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);
  } else {
    bb.template bit_cast_view<unsigned char>().template select<128, 1>(
        128 * 10) = 0;
  }

  for (int n = 0; n < pixelPerGroup; n++) {
    cc = 0.0f;
    offsetQuan = baseOffsetInc8;
    offsetQuan = offsetQuan * 32 * sizeof(fp16) + offsetQuanBase +
        n * 344 * sizeof(fp16);
    quant.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)d, offsetQuan);

    offsetQuan += 32 * sizeof(fp16) * 8;
    quant.template bit_cast_view<uint32_t>().template select<8, 1>(8) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)d, offsetQuan);

    offsetA = baseOffsetInc8;
    offsetA = offsetA * sizeof(uint32_t) + offsetABase + n * 5504;

#pragma unroll
    for (int k = 0; k < 5; k++) {
      simd<float, 4> fp32Q = quant.select<4, 1>(4 * k);
      aaa.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
          __ESIMD_ENS::lsc_gather<
              uint32_t,
              1,
              __ESIMD_ENS::lsc_data_size::u32,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached,
              8,
              uint32_t>((uint32_t*)a, offsetA);
      offsetA += 512;

      aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8) =
          __ESIMD_ENS::lsc_gather<
              uint32_t,
              1,
              __ESIMD_ENS::lsc_data_size::u32,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached,
              8,
              uint32_t>((uint32_t*)a, offsetA);
      offsetA += 512;

#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        // aa.select<16, 1>(32 * kk) = aaa.select<16, 1>(16 * kk) & 0xf;
        // aa.select<16, 1>(32 * kk + 16) = aaa.select<16, 1>(16 * kk) >> 4;
        aa.select<16, 2>(32 * kk) = aaa.select<16, 1>(16 * kk) & 0xf;
        aa.select<16, 2>(32 * kk + 1) = aaa.select<16, 1>(16 * kk) >> 4;;
      }

      aa = aa - 8.0f;
#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        aa.select<32, 1>(32 * kk) = fp32Q[kk] * aa.select<32, 1>(32 * kk);
      }

#pragma unroll
      for (int kk = 0; kk < 8; kk++) {
        cc += aa.select<16, 1>(16 * kk) * bb.select<16, 1>(16 * kk + 128 * k);
      }
    }

    if (hh < 12) {
      simd<float, 2> fp32Q = quant.select<2, 1>(4 * 5);
      aaa.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
          __ESIMD_ENS::lsc_gather<
              uint32_t,
              1,
              __ESIMD_ENS::lsc_data_size::u32,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached,
              8,
              uint32_t>((uint32_t*)a, offsetA);
      offsetA += 512;

#pragma unroll
      for (int kk = 0; kk < 2; kk++) {
        aa.select<16, 2>(32 * kk) = aaa.select<16, 1>(16 * kk) & 0xf;
        aa.select<16, 2>(32 * kk + 1) = aaa.select<16, 1>(16 * kk) >> 4;
        aa.select<32, 1>(32 * kk) = aa.select<32, 1>(32 * kk) - 8.0f;
      }

#pragma unroll
      for (int kk = 0; kk < 2; kk++) {
        aa.select<32, 1>(32 * kk) = fp32Q[kk] * aa.select<32, 1>(32 * kk);
      }

#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        cc += aa.select<16, 1>(16 * kk) * bb.select<16, 1>(16 * kk + 128 * 5);
      }
    }

    cc.select<8, 1>(0) += cc.select<8, 1>(8);
    cc.select<4, 1>(0) += cc.select<4, 1>(4);
    cc.select<2, 1>(0) += cc.select<2, 1>(2);
    simd<float, 1> slmAccumulationTemp = cc[0] + cc[1];
    uint32_t slmGroup = n >> 4;
    uint32_t slmInnerGroupOffset = n & 0xf;
    uint32_t slmAccumulationOffset =
        (slmGroup * 16 * 16 + hh * 16 + slmInnerGroupOffset) * sizeof(float);
    // slm_scalar_store(slmAccumulationOffset, slmAccumulationTemp);
    slm_block_store<float, 1>(slmAccumulationOffset, slmAccumulationTemp);
  }
  barrier();
  if (hh < sumThreads) {
    uint32_t slmSumPhase1LoadOffset = hh * 16 * 16 * sizeof(float);
#pragma unroll
    for (int k = 0; k < 4; k++) {
      bb_f32.select<64, 1>(64 * k) = slm_block_load<float, 64>(
          slmSumPhase1LoadOffset + 64 * k * sizeof(float));
    }

#pragma unroll
    for (int k = 1; k < 16; k++) {
      bb_f32.select<16, 1>(0) =
          bb_f32.select<16, 1>(0) + bb_f32.select<16, 1>(16 * k);
    }

    bb.select<16, 1>(0) = bb_f32.select<16, 1>(0);

    __ESIMD_ENS::lsc_block_store<
        fp16,
        16,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>(
        (fp16*)c + outputOffset + hh * 16, bb.select<16, 1>(0));
  }
}


template <uint32_t pixelPerGroupShift>
ESIMD_INLINE void matrixMulCommonDim4096Int4NoReshapeNx16V3_ipex2_FP32out(
    uint8_t* a,
    uint8_t* b,
    uint8_t* c,
    uint8_t* d,
    nd_item<1>& ndi) {

  constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
  constexpr uint32_t quantPerGroup = 4096 / 32 * pixelPerGroup;
  // constexpr uint32_t baseOffsetInc16[16] = {
  //     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  constexpr uint32_t baseOffsetInc8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  __ESIMD_NS::slm_init(16 * 16 * sizeof(float));
  int hh = ndi.get_local_id(0); // 0-16
  int h = ndi.get_group(0); // [0, 256)
  // int rowSize = ndi.get_group_range(0) * pixelPerGroup;
  // weight offset
  int offsetABase = (h * pixelPerGroup * 4096 + hh * 64) >> 1;
  // scale offset
  int offsetQuanBase = h * quantPerGroup * sizeof(fp16) + hh * 2 * sizeof(fp16);
  // input offset
  int offsetB = hh * 64 * sizeof(fp16);
  // output offset
  int outputOffset = pixelPerGroup * h;
  simd<unsigned char, 128> aaa;
  simd<fp16, 16> quant;
  simd<float, 8> fp32Quant;
  simd<float, 256> bb;
  simd<fp16, 256> bb_fp16;
  simd<float, 16 * 16> aa;
  simd<float, 16> cc(0.0f);
  simd<uint32_t, 8> offsetA(baseOffsetInc8);
  simd<uint32_t, 8> offsetQuan(baseOffsetInc8);
  simd_mask<8> quantPred = 1;
  quantPred[4] = 0;
  quantPred[5] = 0;
  quantPred[6] = 0;
  quantPred[7] = 0;
  // gather read 8, 4*8=32 byte=64 int4
  offsetA = offsetA * 4 + offsetABase;
  // 4096/32 (chunk size)/4 (4 split per thread) = 32 stride
  offsetQuan = offsetQuan * 32 * sizeof(fp16) + offsetQuanBase;

// 4 times for 1024*4
#pragma unroll
  for (int k = 0; k < 4; k++) {
    // 128 uint8_t=64 fp16, 128*k is uint8 offset at bb_fp16, offsetB is input offset
    bb_fp16.template bit_cast_view<uint8_t>().template select<128, 1>(128 * k) =
    // block load
        __ESIMD_ENS::lsc_block_load<
            uint8_t,
            128,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);

    offsetB += 1024 * sizeof(fp16);
  }
  // to fp32
  bb.select<256, 1>(0) = bb_fp16.select<256, 1>(0);

  //
  for (int n = 0; n < pixelPerGroup; n++) {
    cc = 0.0f;
    // 8 gather read, only use first 4 element (2 fp16)
    quant.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)d, offsetQuan, quantPred);

    aaa.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)a, offsetA);
    offsetA += 512;

    aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)a, offsetA);
    offsetA += 512; // 2048 - 16 * sizeof(uint32_t)

    aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8 * 2) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)a, offsetA);
    offsetA += 512; // 2048 - 16 * sizeof(uint32_t)

    aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8 * 3) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)a, offsetA);
    offsetA += 512; // 2048 - 16 * sizeof(uint32_t)

#pragma unroll
    for (int k = 0; k < 8; k++) {
      aa.select<16, 2>(32 * k) = aaa.select<16, 1>(16 * k) & 0x0F;
      aa.select<16, 2>(32 * k + 1) = (aaa.select<16, 1>(16 * k) & 0xF0) >> 4;
    }

    aa = aa - 8.0f;
    fp32Quant = quant.select<8, 1>(0);

#pragma unroll
    for (int k = 0; k < 8; k++) {
      aa.select<32, 1>(32 * k) = fp32Quant[k] * aa.select<32, 1>(32 * k);
    }

#pragma unroll
    for (int k = 0; k < 16; k++) {
      cc += aa.select<16, 1>(16 * k) * bb.select<16, 1>(16 * k);
    }

    cc.select<8, 1>(0) += cc.select<8, 1>(8);
    cc.select<4, 1>(0) += cc.select<4, 1>(4);
    cc.select<2, 1>(0) += cc.select<2, 1>(2);
    simd<float, 1> slmAccumulationTemp = cc[0] + cc[1];
    uint32_t slmAccumulationOffset = (hh * pixelPerGroup + n) * sizeof(float);
    // slm_scalar_store(slmAccumulationOffset, slmAccumulationTemp);
    slm_block_store<float, 1>(slmAccumulationOffset, slmAccumulationTemp);
    offsetQuan += 128 * sizeof(fp16);
  }
  barrier();

  if (hh == 0) {
    if constexpr (pixelPerGroupShift == 4) {
#pragma unroll
      for (int k = 0; k < 4; k++) {
        bb.select<64, 1>(64 * k) =
            slm_block_load<float, 64>(64 * k * sizeof(float));
      }
#pragma unroll
      for (int k = 1; k < 16; k++) {
        bb.select<16, 1>(0) += bb.select<16, 1>(16 * k);
      }

    } else if constexpr (pixelPerGroupShift == 3) {
#pragma unroll
      for (int k = 0; k < 2; k++) {
        bb.select<64, 1>(64 * k) =
            slm_block_load<float, 64>(64 * k * sizeof(float));
      }
#pragma unroll
      for (int k = 1; k < 8; k++) {
        bb.select<16, 1>(0) += bb.select<16, 1>(16 * k);
      }
      bb.select<8, 1>(0) += bb.select<8, 1>(8);
    } else if constexpr (pixelPerGroupShift == 2) {
      bb.select<64, 1>(0) = slm_block_load<float, 64>(0);
#pragma unroll
      for (int k = 1; k < 4; k++) {
        bb.select<16, 1>(0) += bb.select<16, 1>(16 * k);
      }
      bb.select<8, 1>(0) += bb.select<8, 1>(8);
      bb.select<4, 1>(0) += bb.select<4, 1>(4);
    } else if constexpr (pixelPerGroupShift == 1) {
      bb.select<32, 1>(0) = slm_block_load<float, 32>(0);
      bb.select<16, 1>(0) += bb.select<16, 1>(16 * 1);
      bb.select<8, 1>(0) += bb.select<8, 1>(8);
      bb.select<4, 1>(0) += bb.select<4, 1>(4);
      bb.select<2, 1>(0) += bb.select<2, 1>(2);
    } else if constexpr (pixelPerGroupShift == 0) {
      bb.select<16, 1>(0) = slm_block_load<float, 16>(0);
      bb.select<8, 1>(0) += bb.select<8, 1>(8);
      bb.select<4, 1>(0) += bb.select<4, 1>(4);
      bb.select<2, 1>(0) += bb.select<2, 1>(2);
      bb.select<1, 1>(0) += bb.select<1, 1>(1);
    }

    bb_fp16.select<pixelPerGroup, 1>(0) = bb.select<pixelPerGroup, 1>(0);

    __ESIMD_ENS::lsc_block_store<
        float,
        pixelPerGroup,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>(
        (float*)c + outputOffset, bb_fp16.select<pixelPerGroup, 1>(0));
  }
}


template <uint32_t pixelPerGroupShift>
ESIMD_INLINE void matrixMulCommonDim11008Int4NoReshapeNx16V2_ipex_FP32out(
    uint8_t* a,
    uint8_t* b,
    uint8_t* c,
    uint8_t* d,
    nd_item<1>& ndi) {
  constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
  constexpr uint32_t quantPerGroup = 11008 / 32 * pixelPerGroup;
  constexpr uint32_t sumThreads = pixelPerGroup / 16;
  // constexpr uint32_t baseOffsetInc16[16] = {
  //     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  constexpr uint32_t baseOffsetInc8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  __ESIMD_NS::slm_init(16 * 128 * sizeof(float));
  int hh = ndi.get_local_id(0); // [0, 64)
  int h = ndi.get_group(0); // [0, 256)
  int rowSize = ndi.get_group_range(0) * pixelPerGroup;
  int offsetABase = (h * pixelPerGroup * 11008 + hh * 8 * 8) >> 1;
  int offsetQuanBase = /*rowSize * 5504 +*/ h * quantPerGroup * sizeof(fp16) +
      hh * 2 * sizeof(fp16);
  int offsetB = hh * 64 * sizeof(fp16);
  int outputOffset = pixelPerGroup * h;
  simd<uint8_t, 64> aaa;
  simd<fp16, 32> quant;
  simd<fp16, 704> bb;
  simd<float, 256> bb_f32;
  simd<float, 8 * 16> aa;
  simd<float, 16> cc(0.0f);
  simd<uint32_t, 8> offsetA(baseOffsetInc8);
  simd<uint32_t, 8> offsetQuan(baseOffsetInc8);
  offsetA = offsetA * sizeof(uint32_t) + offsetABase;
  offsetQuan = offsetQuan * 32 * sizeof(fp16) + offsetQuanBase;

#pragma unroll
  for (int k = 0; k < 10; k++) {
    bb.template bit_cast_view<unsigned char>().template select<128, 1>(
        128 * k) =
        __ESIMD_ENS::lsc_block_load<
            uint8_t,
            128,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);

    offsetB += 1024 * sizeof(fp16);
  }

  if (hh < 12) {
    bb.template bit_cast_view<unsigned char>().template select<128, 1>(
        128 * 10) =
        __ESIMD_ENS::lsc_block_load<
            uint8_t,
            128,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);
  } else {
    bb.template bit_cast_view<unsigned char>().template select<128, 1>(
        128 * 10) = 0;
  }

  for (int n = 0; n < pixelPerGroup; n++) {
    cc = 0.0f;
    offsetQuan = baseOffsetInc8;
    offsetQuan = offsetQuan * 32 * sizeof(fp16) + offsetQuanBase +
        n * 344 * sizeof(fp16);
    quant.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)d, offsetQuan);

    offsetQuan += 32 * sizeof(fp16) * 8;
    quant.template bit_cast_view<uint32_t>().template select<8, 1>(8) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            1,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            8,
            uint32_t>((uint32_t*)d, offsetQuan);

    offsetA = baseOffsetInc8;
    offsetA = offsetA * sizeof(uint32_t) + offsetABase + n * 5504;

#pragma unroll
    for (int k = 0; k < 5; k++) {
      simd<float, 4> fp32Q = quant.select<4, 1>(4 * k);
      aaa.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
          __ESIMD_ENS::lsc_gather<
              uint32_t,
              1,
              __ESIMD_ENS::lsc_data_size::u32,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached,
              8,
              uint32_t>((uint32_t*)a, offsetA);
      offsetA += 512;

      aaa.template bit_cast_view<uint32_t>().template select<8, 1>(8) =
          __ESIMD_ENS::lsc_gather<
              uint32_t,
              1,
              __ESIMD_ENS::lsc_data_size::u32,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached,
              8,
              uint32_t>((uint32_t*)a, offsetA);
      offsetA += 512;

#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        // aa.select<16, 1>(32 * kk) = aaa.select<16, 1>(16 * kk) & 0xf;
        // aa.select<16, 1>(32 * kk + 16) = aaa.select<16, 1>(16 * kk) >> 4;
        aa.select<16, 2>(32 * kk) = aaa.select<16, 1>(16 * kk) & 0xf;
        aa.select<16, 2>(32 * kk + 1) = aaa.select<16, 1>(16 * kk) >> 4;;
      }

      aa = aa - 8.0f;
#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        aa.select<32, 1>(32 * kk) = fp32Q[kk] * aa.select<32, 1>(32 * kk);
      }

#pragma unroll
      for (int kk = 0; kk < 8; kk++) {
        cc += aa.select<16, 1>(16 * kk) * bb.select<16, 1>(16 * kk + 128 * k);
      }
    }

    if (hh < 12) {
      simd<float, 2> fp32Q = quant.select<2, 1>(4 * 5);
      aaa.template bit_cast_view<uint32_t>().template select<8, 1>(0) =
          __ESIMD_ENS::lsc_gather<
              uint32_t,
              1,
              __ESIMD_ENS::lsc_data_size::u32,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached,
              8,
              uint32_t>((uint32_t*)a, offsetA);
      offsetA += 512;

#pragma unroll
      for (int kk = 0; kk < 2; kk++) {
        aa.select<16, 2>(32 * kk) = aaa.select<16, 1>(16 * kk) & 0xf;
        aa.select<16, 2>(32 * kk + 1) = aaa.select<16, 1>(16 * kk) >> 4;
        aa.select<32, 1>(32 * kk) = aa.select<32, 1>(32 * kk) - 8.0f;
      }

#pragma unroll
      for (int kk = 0; kk < 2; kk++) {
        aa.select<32, 1>(32 * kk) = fp32Q[kk] * aa.select<32, 1>(32 * kk);
      }

#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        cc += aa.select<16, 1>(16 * kk) * bb.select<16, 1>(16 * kk + 128 * 5);
      }
    }

    cc.select<8, 1>(0) += cc.select<8, 1>(8);
    cc.select<4, 1>(0) += cc.select<4, 1>(4);
    cc.select<2, 1>(0) += cc.select<2, 1>(2);
    simd<float, 1> slmAccumulationTemp = cc[0] + cc[1];
    uint32_t slmGroup = n >> 4;
    uint32_t slmInnerGroupOffset = n & 0xf;
    uint32_t slmAccumulationOffset =
        (slmGroup * 16 * 16 + hh * 16 + slmInnerGroupOffset) * sizeof(float);
    // slm_scalar_store(slmAccumulationOffset, slmAccumulationTemp);
    slm_block_store<float, 1>(slmAccumulationOffset, slmAccumulationTemp);
  }
  barrier();
  if (hh < sumThreads) {
    uint32_t slmSumPhase1LoadOffset = hh * 16 * 16 * sizeof(float);
#pragma unroll
    for (int k = 0; k < 4; k++) {
      bb_f32.select<64, 1>(64 * k) = slm_block_load<float, 64>(
          slmSumPhase1LoadOffset + 64 * k * sizeof(float));
    }

#pragma unroll
    for (int k = 1; k < 16; k++) {
      bb_f32.select<16, 1>(0) =
          bb_f32.select<16, 1>(0) + bb_f32.select<16, 1>(16 * k);
    }

    bb.select<16, 1>(0) = bb_f32.select<16, 1>(0);

    __ESIMD_ENS::lsc_block_store<
        float,
        16,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>(
        (float*)c + outputOffset + hh * 16, bb.select<16, 1>(0));
  }
}

ESIMD_INLINE void matrixMulCommonDim4096Int4NoReshape(uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* d, nd_item<1>& ndi) {
  __ESIMD_NS::slm_init(4096 * sizeof(fp16) + 64 * sizeof(fp16));
  int hhh = ndi.get_global_id(0); // [0, 512*64)
  int hh = ndi.get_local_linear_id(); // [0, 64)
  int h = hhh >> 6; // [0, 512)
  int offsetA = (h * 8 * 4096 + hh * 64 * 8) >> 1;
  int offsetQuan = h * 64 * 16 * sizeof(fp16) + hh * 16 * sizeof(fp16);
  int offsetB = hh * 64 * sizeof(fp16);
  int outputOffset = 8 * h;
  simd<unsigned char, 256> aaa;
  simd<fp16, 16> quant;
  simd<fp16, 64> bb;
  simd<fp16, 32 * 16> aa;
  simd<fp16, 16> cc(0.0f);

  bb.template bit_cast_view<unsigned char>().template select<128, 1>(0) =
    __ESIMD_ENS::lsc_block_load<
    uint8_t,
    128,
    __ESIMD_ENS::lsc_data_size::default_size,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);

  aaa.template bit_cast_view<unsigned char>().template select<256, 1>(0) =
    __ESIMD_ENS::lsc_block_load<
    uint8_t,
    256,
    __ESIMD_ENS::lsc_data_size::default_size,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA);

  quant.template bit_cast_view<unsigned char>().template select<32, 1>(0) =
    __ESIMD_ENS::lsc_block_load<
    uint8_t,
    32,
    __ESIMD_ENS::lsc_data_size::default_size,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);

#pragma unroll
  for (int k = 0; k < 16; k++) {
    aa.select<16, 2>(32 * k) = aaa.select<16, 1>(16 * k) & 0xf;
    aa.select<16, 2>(32 * k + 1) = aaa.select<16, 1>(16 * k) >> 4;
  }

  aa = aa - 8.0f;
#pragma unroll
  for (int k = 0; k < 16; k++) {
    aa.select<32, 1>(32 * k) = quant[k] * aa.select<32, 1>(32 * k);
  }

  slm_block_store<fp16, 64>(hh * 64 * sizeof(fp16), bb);
  barrier();
  int inputVectSlmOffset = hh & 0x7;
  inputVectSlmOffset = inputVectSlmOffset * 512 * sizeof(fp16);
#pragma unroll
  for (int ll = 0; ll < 2; ll++) {
    simd<fp16, 16 * 16> fBb;
#pragma unroll
    for (int k = 0; k < 2; k++) {
      fBb.template select<128, 1>(128 * k) = slm_block_load<fp16, 128>(inputVectSlmOffset + 128 * k * sizeof(fp16));
    }

#pragma unroll
    for (int k = 0; k < 16; k++) {
      cc += aa.select<16, 1>(16 * k + ll * 256) * fBb.select<16, 1>(16 * k);
    }
    inputVectSlmOffset += 256 * sizeof(fp16);
  }

  cc.select<8, 1>(0) += cc.select<8, 1>(8);
  cc.select<4, 1>(0) += cc.select<4, 1>(4);
  cc.select<2, 1>(0) += cc.select<2, 1>(2);
  simd<fp16, 1> slmAccumulationTemp = cc[0] + cc[1];
  uint32_t slmAccumulationOffsetGroup = hh >> 3;
  uint32_t slmAccumulationOffsetSimdSlot = hh & 0x7;
  uint32_t slmAccumulationOffset = 4096 * sizeof(fp16) + (slmAccumulationOffsetGroup + slmAccumulationOffsetSimdSlot * 8) * sizeof(fp16);

  //slm_scalar_store(slmAccumulationOffset, slmAccumulationTemp);
  slm_block_store<fp16, 1>(slmAccumulationOffset, slmAccumulationTemp);
  barrier();

  if (hh == 0) {
    simd<fp16, 64> sum;
    //simd<fp16, 8> sumOut;

    sum = slm_block_load<fp16, 64>(4096 * sizeof(fp16));

#pragma unroll
    for (int k = 1; k < 8; k++) {
      sum.select<8, 1>(0) += sum.select<8, 1>(8 * k);
    }

    //sumOut = sum.select<8, 1>(0);

    __ESIMD_ENS::lsc_block_store<
      fp16,
      8,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)c + outputOffset, sum.select<8, 1>(0));
  }
}

ESIMD_INLINE void matrixMulCommonDim4096Int4NoReshape_FP32out(uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* d, nd_item<1>& ndi) {
  __ESIMD_NS::slm_init(4096 * sizeof(fp16) + 64 * sizeof(fp16));
  int hhh = ndi.get_global_id(0); // [0, 512*64)
  int hh = ndi.get_local_linear_id(); // [0, 64)
  int h = hhh >> 6; // [0, 512)
  int offsetA = (h * 8 * 4096 + hh * 64 * 8) >> 1;
  int offsetQuan = h * 64 * 16 * sizeof(fp16) + hh * 16 * sizeof(fp16);
  int offsetB = hh * 64 * sizeof(fp16);
  int outputOffset = 8 * h;
  simd<unsigned char, 256> aaa;
  simd<fp16, 16> quant;
  simd<fp16, 64> bb;
  simd<fp16, 32 * 16> aa;
  simd<fp16, 16> cc(0.0f);

  bb.template bit_cast_view<unsigned char>().template select<128, 1>(0) =
    __ESIMD_ENS::lsc_block_load<
    uint8_t,
    128,
    __ESIMD_ENS::lsc_data_size::default_size,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);

  aaa.template bit_cast_view<unsigned char>().template select<256, 1>(0) =
    __ESIMD_ENS::lsc_block_load<
    uint8_t,
    256,
    __ESIMD_ENS::lsc_data_size::default_size,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA);

  quant.template bit_cast_view<unsigned char>().template select<32, 1>(0) =
    __ESIMD_ENS::lsc_block_load<
    uint8_t,
    32,
    __ESIMD_ENS::lsc_data_size::default_size,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);

#pragma unroll
  for (int k = 0; k < 16; k++) {
    aa.select<16, 2>(32 * k) = aaa.select<16, 1>(16 * k) & 0xf;
    aa.select<16, 2>(32 * k + 1) = aaa.select<16, 1>(16 * k) >> 4;
  }

  aa = aa - 8.0f;
#pragma unroll
  for (int k = 0; k < 16; k++) {
    aa.select<32, 1>(32 * k) = quant[k] * aa.select<32, 1>(32 * k);
  }

  //simd<fp16, 64> fp16Shuffle;
  //fp16Shuffle = bb;
  //fp16Shuffle.select<32, 1>(0) = bb.select<32, 2>(0);
  //fp16Shuffle.select<32, 1>(32) = bb.select<32, 2>(1);
  slm_block_store<fp16, 64>(hh * 64 * sizeof(fp16), bb);
  barrier();
  int inputVectSlmOffset = hh & 0x7;
  inputVectSlmOffset = inputVectSlmOffset * 512 * sizeof(fp16);
#pragma unroll
  for (int ll = 0; ll < 2; ll++) {
    simd<fp16, 16 * 16> fBb;
#pragma unroll
    for (int k = 0; k < 2; k++) {
      fBb.template select<128, 1>(128 * k) = slm_block_load<fp16, 128>(inputVectSlmOffset + 128 * k * sizeof(fp16));
    }

#pragma unroll
    for (int k = 0; k < 16; k++) {
      cc += aa.select<16, 1>(16 * k + ll * 256) * fBb.select<16, 1>(16 * k);
    }
    inputVectSlmOffset += 256 * sizeof(fp16);
  }

  cc.select<8, 1>(0) += cc.select<8, 1>(8);
  cc.select<4, 1>(0) += cc.select<4, 1>(4);
  cc.select<2, 1>(0) += cc.select<2, 1>(2);
  simd<float, 1> slmAccumulationTemp = cc[0] + cc[1];
  uint32_t slmAccumulationOffsetGroup = hh >> 3;
  uint32_t slmAccumulationOffsetSimdSlot = hh & 0x7;
  uint32_t slmAccumulationOffset = 4096 * sizeof(fp16) + (slmAccumulationOffsetGroup + slmAccumulationOffsetSimdSlot * 8) * sizeof(float);

  //slm_scalar_store(slmAccumulationOffset, slmAccumulationTemp);
  slm_block_store<float, 1>(slmAccumulationOffset, slmAccumulationTemp);
  barrier();

  if (hh == 0) {
    simd<float, 64> sum;
    //simd<fp16, 8> sumOut;

    sum = slm_block_load<float, 64>(4096 * sizeof(fp16));

#pragma unroll
    for (int k = 1; k < 8; k++) {
      sum.select<8, 1>(0) += sum.select<8, 1>(8 * k);
    }

    //sumOut = sum.select<8, 1>(0);

    __ESIMD_ENS::lsc_block_store<
      float,
      8,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((float*)c + outputOffset, sum.select<8, 1>(0));
  }
}


ESIMD_INLINE void matrixMulCommonDim11008Int4NoReshape(uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* d, nd_item<2>& ndi) {
  __ESIMD_NS::slm_init(11008 * sizeof(fp16) + 88 * sizeof(fp16));
  int hh = ndi.get_local_id(0); // [0, 11)
  int vv = ndi.get_local_id(1); // [0, 4)
  int h = ndi.get_group(0); // [0, 11)
  int localLinearId = vv * 11 + hh;
  int offsetA = (h * 8 * 11008 + vv * 11008 + hh * 1024) >> 1;
  int offsetQuant = h * 344 * 8 * sizeof(fp16) + (vv * 344 + hh * 32) * sizeof(fp16);
  int offsetB = localLinearId * 256 * sizeof(fp16);
  int outputOffset = 8 * h;
  simd<unsigned char, 512> aaa;
  simd<fp16, 32> quant;
  simd<fp16, 256> bb;
  simd<fp16, 64 * 16> aa;
  simd<fp16, 16> cc(0.0f);

  aaa.template bit_cast_view<unsigned char>().template select<256, 1>(0) =
    __ESIMD_ENS::lsc_block_load<
    uint8_t,
    256,
    __ESIMD_ENS::lsc_data_size::default_size,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA);

  aaa.template bit_cast_view<unsigned char>().template select<128, 1>(256) =
    __ESIMD_ENS::lsc_block_load<
    uint8_t,
    128,
    __ESIMD_ENS::lsc_data_size::default_size,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA + 256);

  if (hh != 10) {
    aaa.template bit_cast_view<unsigned char>().template select<128, 1>(256 + 128) =
      __ESIMD_ENS::lsc_block_load<
      uint8_t,
      128,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA + 256 + 128);

  }

  quant.template bit_cast_view<unsigned char>().template select<64, 1>(0) =
    __ESIMD_ENS::lsc_block_load<
    uint8_t,
    64,
    __ESIMD_ENS::lsc_data_size::default_size,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuant);

  if (localLinearId < 43) {
#pragma unroll
    for (int k = 0; k < 4; k++) {
      bb.template bit_cast_view<unsigned char>().template select<128, 1>(128 * k) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        128,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 128 * k);
    }
  }

#pragma unroll
  for (int k = 0; k < 24; k++) {
    aa.select<16, 2>(32 * k) = aaa.select<16, 1>(16 * k) & 0xf;
    aa.select<16, 2>(32 * k + 1) = aaa.select<16, 1>(16 * k) >> 4;
  }

  if (hh != 10) {
#pragma unroll
    for (int k = 24; k < 32; k++) {
      aa.select<16, 2>(32 * k) = aaa.select<16, 1>(16 * k) & 0xf;
      aa.select<16, 2>(32 * k + 1) = aaa.select<16, 1>(16 * k) >> 4;
    }
  }

  aa = aa - 8.0f;
#pragma unroll
  for (int k = 0; k < 24; k++) {
    aa.select<32, 1>(32 * k) = aa.select<32, 1>(32 * k) * quant[k];
  }

  if (hh != 10) {
#pragma unroll
    for (int k = 24; k < 32; k++) {
      aa.select<32, 1>(32 * k) = aa.select<32, 1>(32 * k) * quant[k];
    }
  }

  if (localLinearId < 43) {
    simd<fp16, 64> fBbShuffle;
#pragma unroll
    for (int k = 0; k < 4; k++) {
      fBbShuffle = bb.select<64, 1>(64 * k);
      //fBbShuffle.select<32, 1>(0) = bb.select<32, 2>(64 * k);
      //fBbShuffle.select<32, 1>(32) = bb.select<32, 2>(64 * k + 1);
      slm_block_store<fp16, 64>((localLinearId * 256 + k * 64) * sizeof(fp16), fBbShuffle.select<64, 1>(0));
    }
  }
  barrier();
  int inputVectSlmOffset = hh * 1024 * sizeof(fp16);
  simd<fp16, 16 * 16> fBb;
#pragma unroll
  for (int ll = 0; ll < 3; ll++) {
#pragma unroll
    for (int k = 0; k < 2; k++) {
      fBb.select<128, 1>(128 * k) = slm_block_load<fp16, 128>(inputVectSlmOffset + k * 128 * sizeof(fp16));
    }

#pragma unroll
    for (int k = 0; k < 16; k++) {
      cc += aa.select<16, 1>(16 * k + ll * 256) * fBb.select<16, 1>(16 * k);
    }
    inputVectSlmOffset += 256 * sizeof(fp16);
  }

  if (hh != 10) {
#pragma unroll
    for (int k = 0; k < 2; k++) {
      fBb.select<128, 1>(128 * k) = slm_block_load<fp16, 128>(inputVectSlmOffset + k * 128 * sizeof(fp16));
    }

#pragma unroll
    for (int k = 0; k < 16; k++) {
      cc += aa.select<16, 1>(16 * k + 3 * 256) * fBb.select<16, 1>(16 * k);
    }
    inputVectSlmOffset += 256 * sizeof(fp16);
  }

  cc.select<8, 1>(0) += cc.select<8, 1>(8);
  cc.select<4, 1>(0) += cc.select<4, 1>(4);
  cc.select<2, 1>(0) += cc.select<2, 1>(2);
  simd<fp16, 1> slmAccumulationTemp = cc[0] + cc[1];
  uint32_t slmAccumulationOffsetGroup = vv;
  uint32_t slmAccumulationOffsetSimdSlot = hh;
  uint32_t slmAccumulationOffset = 11008 * sizeof(fp16) + (slmAccumulationOffsetGroup + slmAccumulationOffsetSimdSlot * 8) * sizeof(fp16);
  //slm_scalar_store(slmAccumulationOffset, slmAccumulationTemp);
  slm_block_store<fp16, 1>(slmAccumulationOffset, slmAccumulationTemp);
  cc = (fp16)0.0f;
  offsetA += 11008 * 4 / 2;
  offsetQuant += 344 * 4 * sizeof(fp16);

  aaa.template bit_cast_view<unsigned char>().template select<256, 1>(0) =
    __ESIMD_ENS::lsc_block_load<
    uint8_t,
    256,
    __ESIMD_ENS::lsc_data_size::default_size,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA);

  aaa.template bit_cast_view<unsigned char>().template select<128, 1>(256) =
    __ESIMD_ENS::lsc_block_load<
    uint8_t,
    128,
    __ESIMD_ENS::lsc_data_size::default_size,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA + 256);

  if (hh != 10) {
    aaa.template bit_cast_view<unsigned char>().template select<128, 1>(256 + 128) =
      __ESIMD_ENS::lsc_block_load<
      uint8_t,
      128,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA + 256 + 128);
  }

  quant.template bit_cast_view<unsigned char>().template select<64, 1>(0) =
    __ESIMD_ENS::lsc_block_load<
    uint8_t,
    64,
    __ESIMD_ENS::lsc_data_size::default_size,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuant);

#pragma unroll
  for (int k = 0; k < 24; k++) {
    aa.select<16, 2>(32 * k) = aaa.select<16, 1>(16 * k) & 0xf;
    aa.select<16, 2>(32 * k + 1) = aaa.select<16, 1>(16 * k) >> 4;
  }

  if (hh != 10) {
#pragma unroll
    for (int k = 24; k < 32; k++) {
      aa.select<16, 2>(32 * k) = aaa.select<16, 1>(16 * k) & 0xf;
      aa.select<16, 2>(32 * k + 1) = aaa.select<16, 1>(16 * k) >> 4;
    }
  }

  aa = aa - 8.0f;
#pragma unroll
  for (int k = 0; k < 24; k++) {
    aa.select<32, 1>(32 * k) = aa.select<32, 1>(32 * k) * quant[k];
  }

  if (hh != 10) {
#pragma unroll
    for (int k = 24; k < 32; k++) {
      aa.select<32, 1>(32 * k) = aa.select<32, 1>(32 * k) * quant[k];
    }
  }

  inputVectSlmOffset = hh * 1024 * sizeof(fp16);
#pragma unroll
  for (int ll = 0; ll < 3; ll++) {
#pragma unroll
    for (int k = 0; k < 2; k++) {
      fBb.select<128, 1>(128 * k) = slm_block_load<fp16, 128>(inputVectSlmOffset + k * 128 * sizeof(fp16));
    }

#pragma unroll
    for (int k = 0; k < 16; k++) {
      cc += aa.select<16, 1>(16 * k + ll * 256) * fBb.select<16, 1>(16 * k);
    }
    inputVectSlmOffset += 256 * sizeof(fp16);
  }

  if (hh != 10) {
#pragma unroll
    for (int k = 0; k < 2; k++) {
      fBb.select<128, 1>(128 * k) = slm_block_load<fp16, 128>(inputVectSlmOffset + k * 128 * sizeof(fp16));
    }

#pragma unroll
    for (int k = 0; k < 16; k++) {
      cc += aa.select<16, 1>(16 * k + 3 * 256) * fBb.select<16, 1>(16 * k);
    }
    inputVectSlmOffset += 256 * sizeof(fp16);
  }

  cc.select<8, 1>(0) += cc.select<8, 1>(8);
  cc.select<4, 1>(0) += cc.select<4, 1>(4);
  cc.select<2, 1>(0) += cc.select<2, 1>(2);
  slmAccumulationTemp = cc[0] + cc[1];
  slmAccumulationOffsetGroup = vv + 4;
  slmAccumulationOffsetSimdSlot = hh;
  slmAccumulationOffset = 11008 * sizeof(fp16) + (slmAccumulationOffsetGroup + slmAccumulationOffsetSimdSlot * 8) * sizeof(fp16);
  //slm_scalar_store(slmAccumulationOffset, slmAccumulationTemp);
  slm_block_store<fp16, 1>(slmAccumulationOffset, slmAccumulationTemp);
  barrier();

  if (localLinearId == 0) {
    simd<fp16, 88> sum;
    //simd<fp16, 8> sumOut;
    sum.select<64, 1>(0) = slm_block_load<fp16, 64>(11008 * sizeof(fp16));
    sum.select<16, 1>(64) = slm_block_load<fp16, 16>(11008 * sizeof(fp16) + 64 * sizeof(fp16));
    sum.select<8, 1>(64 + 16) = slm_block_load<fp16, 8>(11008 * sizeof(fp16) + (64 + 16) * sizeof(fp16));

#pragma unroll
    for (int k = 1; k < 11; k++) {
      sum.select<8, 1>(0) += sum.select<8, 1>(8 * k);
    }

    //sumOut = sum.select<8, 1>(0);

    __ESIMD_ENS::lsc_block_store<
      fp16,
      8,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)c + outputOffset, sum.select<8, 1>(0));
  }
}

ESIMD_INLINE void matrixMulCommonDim11008Int4NoReshape_FP32out(uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* d, nd_item<2>& ndi) {
  __ESIMD_NS::slm_init(11008 * sizeof(fp16) + 88 * sizeof(fp16));
  int hh = ndi.get_local_id(0); // [0, 11)
  int vv = ndi.get_local_id(1); // [0, 4)
  int h = ndi.get_group(0); // [0, 11)
  int localLinearId = vv * 11 + hh;
  int offsetA = (h * 8 * 11008 + vv * 11008 + hh * 1024) >> 1;
  int offsetQuant = h * 344 * 8 * sizeof(fp16) + (vv * 344 + hh * 32) * sizeof(fp16);
  int offsetB = localLinearId * 256 * sizeof(fp16);
  int outputOffset = 8 * h;
  simd<unsigned char, 512> aaa;
  simd<fp16, 32> quant;
  simd<fp16, 256> bb;
  simd<fp16, 64 * 16> aa;
  simd<fp16, 16> cc(0.0f);

  aaa.template bit_cast_view<unsigned char>().template select<256, 1>(0) =
    __ESIMD_ENS::lsc_block_load<
    uint8_t,
    256,
    __ESIMD_ENS::lsc_data_size::default_size,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA);

  aaa.template bit_cast_view<unsigned char>().template select<128, 1>(256) =
    __ESIMD_ENS::lsc_block_load<
    uint8_t,
    128,
    __ESIMD_ENS::lsc_data_size::default_size,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA + 256);

  if (hh != 10) {
    aaa.template bit_cast_view<unsigned char>().template select<128, 1>(256 + 128) =
      __ESIMD_ENS::lsc_block_load<
      uint8_t,
      128,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA + 256 + 128);

  }

  quant.template bit_cast_view<unsigned char>().template select<64, 1>(0) =
    __ESIMD_ENS::lsc_block_load<
    uint8_t,
    64,
    __ESIMD_ENS::lsc_data_size::default_size,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuant);

  if (localLinearId < 43) {
#pragma unroll
    for (int k = 0; k < 4; k++) {
      bb.template bit_cast_view<unsigned char>().template select<128, 1>(128 * k) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        128,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 128 * k);
    }
  }

#pragma unroll
  for (int k = 0; k < 24; k++) {
    aa.select<16, 2>(32 * k) = aaa.select<16, 1>(16 * k) & 0xf;
    aa.select<16, 2>(32 * k + 1) = aaa.select<16, 1>(16 * k) >> 4;
  }

  if (hh != 10) {
#pragma unroll
    for (int k = 24; k < 32; k++) {
      aa.select<16, 2>(32 * k) = aaa.select<16, 1>(16 * k) & 0xf;
      aa.select<16, 2>(32 * k + 1) = aaa.select<16, 1>(16 * k) >> 4;
    }
  }

  aa = aa - 8.0f;
#pragma unroll
  for (int k = 0; k < 24; k++) {
    aa.select<32, 1>(32 * k) = aa.select<32, 1>(32 * k) * quant[k];
  }

  if (hh != 10) {
#pragma unroll
    for (int k = 24; k < 32; k++) {
      aa.select<32, 1>(32 * k) = aa.select<32, 1>(32 * k) * quant[k];
    }
  }

  if (localLinearId < 43) {
    simd<fp16, 64> fBbShuffle;
#pragma unroll
    for (int k = 0; k < 4; k++) {
      fBbShuffle = bb.select<64, 1>(64 * k);
      //fBbShuffle.select<32, 1>(0) = bb.select<32, 2>(64 * k);
      //fBbShuffle.select<32, 1>(32) = bb.select<32, 2>(64 * k + 1);
      slm_block_store<fp16, 64>((localLinearId * 256 + k * 64) * sizeof(fp16), fBbShuffle.select<64, 1>(0));
    }
  }
  barrier();
  int inputVectSlmOffset = hh * 1024 * sizeof(fp16);
  simd<fp16, 16 * 16> fBb;
#pragma unroll
  for (int ll = 0; ll < 3; ll++) {
#pragma unroll
    for (int k = 0; k < 2; k++) {
      fBb.select<128, 1>(128 * k) = slm_block_load<fp16, 128>(inputVectSlmOffset + k * 128 * sizeof(fp16));
    }

#pragma unroll
    for (int k = 0; k < 16; k++) {
      cc += aa.select<16, 1>(16 * k + ll * 256) * fBb.select<16, 1>(16 * k);
    }
    inputVectSlmOffset += 256 * sizeof(fp16);
  }

  if (hh != 10) {
#pragma unroll
    for (int k = 0; k < 2; k++) {
      fBb.select<128, 1>(128 * k) = slm_block_load<fp16, 128>(inputVectSlmOffset + k * 128 * sizeof(fp16));
    }

#pragma unroll
    for (int k = 0; k < 16; k++) {
      cc += aa.select<16, 1>(16 * k + 3 * 256) * fBb.select<16, 1>(16 * k);
    }
    inputVectSlmOffset += 256 * sizeof(fp16);
  }

  cc.select<8, 1>(0) += cc.select<8, 1>(8);
  cc.select<4, 1>(0) += cc.select<4, 1>(4);
  cc.select<2, 1>(0) += cc.select<2, 1>(2);
  simd<fp16, 1> slmAccumulationTemp = cc[0] + cc[1];
  uint32_t slmAccumulationOffsetGroup = vv;
  uint32_t slmAccumulationOffsetSimdSlot = hh;
  uint32_t slmAccumulationOffset = 11008 * sizeof(fp16) + (slmAccumulationOffsetGroup + slmAccumulationOffsetSimdSlot * 8) * sizeof(fp16);
  //slm_scalar_store(slmAccumulationOffset, slmAccumulationTemp);
  slm_block_store<fp16, 1>(slmAccumulationOffset, slmAccumulationTemp);
  cc = (fp16)0.0f;
  offsetA += 11008 * 4 / 2;
  offsetQuant += 344 * 4 * sizeof(fp16);

  aaa.template bit_cast_view<unsigned char>().template select<256, 1>(0) =
    __ESIMD_ENS::lsc_block_load<
    uint8_t,
    256,
    __ESIMD_ENS::lsc_data_size::default_size,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA);

  aaa.template bit_cast_view<unsigned char>().template select<128, 1>(256) =
    __ESIMD_ENS::lsc_block_load<
    uint8_t,
    128,
    __ESIMD_ENS::lsc_data_size::default_size,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA + 256);

  if (hh != 10) {
    aaa.template bit_cast_view<unsigned char>().template select<128, 1>(256 + 128) =
      __ESIMD_ENS::lsc_block_load<
      uint8_t,
      128,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA + 256 + 128);
  }

  quant.template bit_cast_view<unsigned char>().template select<64, 1>(0) =
    __ESIMD_ENS::lsc_block_load<
    uint8_t,
    64,
    __ESIMD_ENS::lsc_data_size::default_size,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuant);

#pragma unroll
  for (int k = 0; k < 24; k++) {
    aa.select<16, 2>(32 * k) = aaa.select<16, 1>(16 * k) & 0xf;
    aa.select<16, 2>(32 * k + 1) = aaa.select<16, 1>(16 * k) >> 4;
  }

  if (hh != 10) {
#pragma unroll
    for (int k = 24; k < 32; k++) {
      aa.select<16, 2>(32 * k) = aaa.select<16, 1>(16 * k) & 0xf;
      aa.select<16, 2>(32 * k + 1) = aaa.select<16, 1>(16 * k) >> 4;
    }
  }

  aa = aa - 8.0f;
#pragma unroll
  for (int k = 0; k < 24; k++) {
    aa.select<32, 1>(32 * k) = aa.select<32, 1>(32 * k) * quant[k];
  }

  if (hh != 10) {
#pragma unroll
    for (int k = 24; k < 32; k++) {
      aa.select<32, 1>(32 * k) = aa.select<32, 1>(32 * k) * quant[k];
    }
  }

  inputVectSlmOffset = hh * 1024 * sizeof(fp16);
#pragma unroll
  for (int ll = 0; ll < 3; ll++) {
#pragma unroll
    for (int k = 0; k < 2; k++) {
      fBb.select<128, 1>(128 * k) = slm_block_load<fp16, 128>(inputVectSlmOffset + k * 128 * sizeof(fp16));
    }

#pragma unroll
    for (int k = 0; k < 16; k++) {
      cc += aa.select<16, 1>(16 * k + ll * 256) * fBb.select<16, 1>(16 * k);
    }
    inputVectSlmOffset += 256 * sizeof(fp16);
  }

  if (hh != 10) {
#pragma unroll
    for (int k = 0; k < 2; k++) {
      fBb.select<128, 1>(128 * k) = slm_block_load<fp16, 128>(inputVectSlmOffset + k * 128 * sizeof(fp16));
    }

#pragma unroll
    for (int k = 0; k < 16; k++) {
      cc += aa.select<16, 1>(16 * k + 3 * 256) * fBb.select<16, 1>(16 * k);
    }
    inputVectSlmOffset += 256 * sizeof(fp16);
  }

  cc.select<8, 1>(0) += cc.select<8, 1>(8);
  cc.select<4, 1>(0) += cc.select<4, 1>(4);
  cc.select<2, 1>(0) += cc.select<2, 1>(2);
  slmAccumulationTemp = cc[0] + cc[1];
  slmAccumulationOffsetGroup = vv + 4;
  slmAccumulationOffsetSimdSlot = hh;
  slmAccumulationOffset = 11008 * sizeof(fp16) + (slmAccumulationOffsetGroup + slmAccumulationOffsetSimdSlot * 8) * sizeof(fp16);
  //slm_scalar_store(slmAccumulationOffset, slmAccumulationTemp);
  slm_block_store<fp16, 1>(slmAccumulationOffset, slmAccumulationTemp);
  barrier();

  if (localLinearId == 0) {
    simd<fp16, 88> sum;
    //simd<fp16, 8> sumOut;
    sum.select<64, 1>(0) = slm_block_load<fp16, 64>(11008 * sizeof(fp16));
    sum.select<16, 1>(64) = slm_block_load<fp16, 16>(11008 * sizeof(fp16) + 64 * sizeof(fp16));
    sum.select<8, 1>(64 + 16) = slm_block_load<fp16, 8>(11008 * sizeof(fp16) + (64 + 16) * sizeof(fp16));

#pragma unroll
    for (int k = 1; k < 11; k++) {
      sum.select<8, 1>(0) += sum.select<8, 1>(8 * k);
    }

    //sumOut = sum.select<8, 1>(0);

    __ESIMD_ENS::lsc_block_store<
      float,
      8,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((float*)c + outputOffset, sum.select<8, 1>(0));
  }
}
