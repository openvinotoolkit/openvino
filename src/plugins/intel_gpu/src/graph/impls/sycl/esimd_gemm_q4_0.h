#include <ext/intel/esimd.hpp>
using fp16 = ::sycl::half;

using namespace ::sycl::ext::intel::esimd;
using namespace ::sycl;
using namespace ::sycl::ext::intel::esimd;
using namespace ::sycl::ext::intel::esimd::xmx;

ESIMD_INLINE void gemmReduce2048WeightsQ40InputFp16_ipex(
    uint8_t* a,
    uint8_t* b,
    uint8_t* c,
    uint8_t* d,
    int hiddenDim,
    int tokenSize,
    int reduceIdx,
    int lastReduce,
    nd_item<2>& ndi) {
  constexpr uint32_t baseOffsetInc16[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  // constexpr uint32_t baseOffsetInc8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  // constexpr uint32_t baseOffsetInc4[4] = {0, 1, 2, 3};
  __ESIMD_NS::slm_init(64 * 8 * 16 * sizeof(fp16) + 16 * 32 * sizeof(fp16));
  int hh = ndi.get_local_linear_id(); // [0, 64)
  int h = ndi.get_group(0); // [0, (row + 15) / 16)
  int v = reduceIdx; // [0, (row + 15) / 16)
  int outputRow = ndi.get_group_range(0) * 16;
  int hiddenDimInt4Size = hiddenDim >> 1;
  int hiddenDimDequantSize = hiddenDim >> 5;
  uint32_t globalOffset = v * 2048 + h * hiddenDim * 16 + hh * 32;
  uint32_t baseOffsetA = globalOffset >> 1;
  uint32_t baseOffsetQuant =
      /*hiddenDimInt4Size * outputRow +*/ (globalOffset >> 4);
  uint32_t baseOffsetB = (v * 2048 + hh * 32) * sizeof(fp16);
  uint32_t offsetC = h * 16 + hh * outputRow;
  simd<fp16, 8 * 32> bb;
  simd<fp16, 8 * 32> bb_fp16;
  simd<fp16, 32 * 16> aa;
  simd<fp16, 8 * 16> cc;
  simd<fp16, 8 * 16> cc_fp16;
  simd<uint32_t, 16> offset(baseOffsetInc16);
  simd<uint32_t, 16> offsetQuant(baseOffsetInc16);
  uint32_t loopCount = (tokenSize + 7) >> 3;

  offsetQuant =
      offsetQuant * hiddenDimDequantSize * sizeof(fp16) + baseOffsetQuant;

  {
    cc.template bit_cast_view<fp16>().template select<16, 1>(16 * 0) =
        __ESIMD_ENS::lsc_gather<
            fp16,
            1,
            __ESIMD_ENS::lsc_data_size::u16,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::uncached,
            16,
            uint32_t>((fp16*)d, offsetQuant);

    offsetQuant += 32 * sizeof(fp16);
  }

  cc.select<16, 1>(16) =
      cc.template bit_cast_view<fp16>().template select<16, 1>(0);

  offset = offset * hiddenDimInt4Size + baseOffsetA;

  {
    bb.template bit_cast_view<uint32_t>().template select<64, 1>(64 * 0) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            4,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::uncached,
            __ESIMD_ENS::cache_hint::uncached,
            16,
            uint32_t>((uint32_t*)a, offset);

// #pragma unroll
//     for (int kk = 0; kk < 4; kk++) {
//       simd<int8_t, 16> bitShiftTemp = bb.template
//       bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 0); aa.select<16,
//       1>((4 * kk + 0) * 16) = bitShiftTemp & 0xf; aa.select<16, 1>((4 * kk +
//       0) * 16 + 16 * 16) = bitShiftTemp >> 4; bitShiftTemp = bb.template
//       bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 1); aa.select<16,
//       1>((4 * kk + 1) * 16) = bitShiftTemp & 0xf; aa.select<16, 1>((4 * kk +
//       1) * 16 + 16 * 16) = bitShiftTemp >> 0x4; bitShiftTemp = bb.template
//       bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 2); aa.select<16,
//       1>((4 * kk + 2) * 16) = bitShiftTemp & 0xf; aa.select<16, 1>((4 * kk +
//       2) * 16 + 16 * 16) = bitShiftTemp >> 4; bitShiftTemp = bb.template
//       bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 3); aa.select<16,
//       1>((4 * kk + 3) * 16) = bitShiftTemp & 0xf; aa.select<16, 1>((4 * kk +
//       3) * 16 + 16 * 16) = bitShiftTemp >> 4;
//     }
//     offset += 128 * sizeof(uint32_t);
//   }
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      simd<uint8_t, 16> bitShiftTemp =
          bb.template bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 0);

      aa.select<16, 1>((4 * 2 * kk + 0) * 16) = bitShiftTemp & 0xf;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 1 * 16) = bitShiftTemp >> 4;

      // ==========================================================================================
      bitShiftTemp =
          bb.template bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 1);

      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 2 * 16) = bitShiftTemp & 0xf;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 3 * 16) = bitShiftTemp >> 4;

      // ==========================================================================================
      bitShiftTemp =
          bb.template bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 2);

      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 4 * 16) = bitShiftTemp & 0xf;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 5 * 16) = bitShiftTemp >> 4;

      // ==========================================================================================
      bitShiftTemp =
          bb.template bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 3);

      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 6 * 16) = bitShiftTemp & 0xf;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 7 * 16) = bitShiftTemp >> 4;
    }
    offset += 128 * sizeof(uint32_t);
  }

  aa = aa - 8.0f;
#pragma unroll
  for (int k = 0; k < 32; k++) {
    aa.select<16, 1>(16 * k) = aa.select<16, 1>(16 * k) * cc.select<16, 1>(16);
  }

  for (int nn = 0; nn < loopCount; nn++) {
    cc = 0;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      bb_fp16.template bit_cast_view<uint8_t>().template select<64, 1>(64 * k) =
          __ESIMD_ENS::lsc_block_load<
              uint8_t,
              64,
              __ESIMD_ENS::lsc_data_size::default_size,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + baseOffsetB);

#pragma unroll
      for (int kkk = 0; kkk < 32; kkk++) {
        cc.select<16, 1>(16 * k) +=
            aa.select<16, 1>(kkk * 16) * bb_fp16[k * 32 + kkk];
      }

      baseOffsetB += hiddenDim * sizeof(fp16);
    }

#pragma unroll
    for (int k = 0; k < 8; k++) {
      slm_block_store<fp16, 16>(
          (hh * 16 + k * 16 * 64) * sizeof(fp16), cc.select<16, 1>(16 * k));
    }

    barrier();

    if (hh < 32) {
      uint32_t slmOffset = hh * 16 * 16 * sizeof(fp16);
#pragma unroll
      for (int k = 0; k < 4; k++) {
        bb.select<64, 1>(64 * k) =
            slm_block_load<fp16, 64>(slmOffset + k * 64 * sizeof(fp16));
      }

#pragma unroll
      for (int k = 1; k < 8; k++) {
        bb.select<32, 1>(0) += bb.select<32, 1>(32 * k);
      }
      bb.select<16, 1>(0) += bb.select<16, 1>(16);
      slm_block_store<fp16, 16>(
          8 * 16 * 64 * sizeof(fp16) + hh * 16 * sizeof(fp16),
          bb.select<16, 1>(0));
    }

    barrier();

    if (hh < 8) {
      if (8 * nn + hh < tokenSize) {
        if (v != 0) {
          cc_fp16.select<16, 1>(0) = __ESIMD_ENS::lsc_block_load<
              fp16,
              16,
              __ESIMD_ENS::lsc_data_size::default_size,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached>((fp16*)c + offsetC);
        } else {
          cc_fp16.select<16, 1>(0) = 0;
        }
        uint32_t slmOffset =
            hh * 16 * 4 * sizeof(fp16) + 16 * 64 * 8 * sizeof(fp16);
        bb.template bit_cast_view<fp16>().template select<64, 1>(0) =
            slm_block_load<fp16, 64>(slmOffset);

#pragma unroll
        for (int k = 0; k < 4; k++) {
          cc_fp16.select<16, 1>(0) += bb.select<16, 1>(16 * k);
        }

        __ESIMD_ENS::lsc_block_store<
            fp16,
            16,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::write_back,
            __ESIMD_ENS::cache_hint::write_back>(
            (fp16*)c + offsetC, cc_fp16.select<16, 1>(0));

        offsetC += 8 * outputRow;
      }
    }
  }
}

ESIMD_INLINE void gemmReduce768WeightsQ40InputFp16_ipex(
    uint8_t* a,
    uint8_t* b,
    uint8_t* c,
    uint8_t* d,
    int hiddenDim,
    int tokenSize,
    int reduceIdx,
    int lastReduce,
    nd_item<2>& ndi) {
  constexpr uint32_t baseOffsetInc16[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  // constexpr uint32_t baseOffsetInc8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  // constexpr uint32_t baseOffsetInc4[4] = {0, 1, 2, 3};
  __ESIMD_NS::slm_init(24 * 8 * 16 * sizeof(float) + 16 * 32 * sizeof(float));
  int hh = ndi.get_local_linear_id(); // [0, 64)
  int h = ndi.get_group(0); // [0, (row + 15) / 16)
  int v = reduceIdx; // [0, (row + 15) / 16)
  int outputRow = ndi.get_group_range(0) * 16;
  int hiddenDimInt4Size = hiddenDim >> 1;
  int hiddenDimDequantSize = hiddenDim >> 5;
  uint32_t globalOffset = v * 2048 + h * hiddenDim * 16 + hh * 32;
  uint32_t baseOffsetA = globalOffset >> 1;
  uint32_t baseOffsetQuant =
      /*hiddenDimInt4Size * outputRow +*/ (globalOffset >> 4);
  uint32_t baseOffsetB = (v * 2048 + hh * 32) * sizeof(fp16);
  uint32_t offsetC = h * 16 + hh * outputRow;
  simd<float, 8 * 32> bb;
  simd<fp16, 8 * 32> bb_fp16;
  simd<float, 32 * 16> aa;
  simd<float, 8 * 16> cc;
  simd<fp16, 8 * 16> cc_fp16;
  simd<uint32_t, 16> offset(baseOffsetInc16);
  simd<uint32_t, 16> offsetQuant(baseOffsetInc16);
  uint32_t loopCount = (tokenSize + 7) >> 3;

  offsetQuant =
      offsetQuant * hiddenDimDequantSize * sizeof(fp16) + baseOffsetQuant;

  {
    cc.template bit_cast_view<fp16>().template select<16, 1>(16 * 0) =
        __ESIMD_ENS::lsc_gather<
            fp16,
            1,
            __ESIMD_ENS::lsc_data_size::u16,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::uncached,
            16,
            uint32_t>((fp16*)d, offsetQuant);

    offsetQuant += 12 * sizeof(fp16);
  }

  cc.select<16, 1>(16) =
      cc.template bit_cast_view<fp16>().template select<16, 1>(0);
  offset = offset * hiddenDimInt4Size + baseOffsetA;

  {
    bb.template bit_cast_view<uint32_t>().template select<64, 1>(64 * 0) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            4,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::uncached,
            __ESIMD_ENS::cache_hint::uncached,
            16,
            uint32_t>((uint32_t*)a, offset);

#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      simd<u_char, 16> bitShiftTemp =
          bb.template bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 0);

      aa.select<16, 1>((4 * 2 * kk + 0) * 16) = bitShiftTemp & 0xf;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 1 * 16) = bitShiftTemp >> 4;

      // ==========================================================================================
      bitShiftTemp =
          bb.template bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 1);


      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 2 * 16) = bitShiftTemp & 0xf;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 3 * 16) = bitShiftTemp >> 4;

      // ==========================================================================================
      bitShiftTemp =
          bb.template bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 2);


      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 4 * 16) = bitShiftTemp & 0xf;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 5 * 16) = bitShiftTemp >> 4;

      // ==========================================================================================
      bitShiftTemp =
          bb.template bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 3);

      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 6 * 16) = bitShiftTemp & 0xf;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 7 * 16) = bitShiftTemp >> 4;
    }
    offset += 48 * sizeof(uint32_t);
  }

  aa = aa - 8.0f;
#pragma unroll
  for (int k = 0; k < 32; k++) {
    aa.select<16, 1>(16 * k) = aa.select<16, 1>(16 * k) * cc.select<16, 1>(16);
  }

  for (int nn = 0; nn < loopCount; nn++) {
    cc = 0;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      bb_fp16.template bit_cast_view<uint8_t>().template select<64, 1>(64 * k) =
          __ESIMD_ENS::lsc_block_load<
              uint8_t,
              64,
              __ESIMD_ENS::lsc_data_size::default_size,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + baseOffsetB);

#pragma unroll
      for (int kkk = 0; kkk < 32; kkk++) {
        cc.select<16, 1>(16 * k) +=
            aa.select<16, 1>(kkk * 16) * bb_fp16[k * 32 + kkk];
      }

      baseOffsetB += hiddenDim * sizeof(fp16);
    }

    barrier();

#pragma unroll
    for (int k = 0; k < 8; k++) {
      slm_block_store<float, 16>(
          (hh * 16 + k * 16 * 24) * sizeof(float), cc.select<16, 1>(16 * k));
    }

    barrier();

    if (hh < 8) {
      if (8 * nn + hh < tokenSize) {
        uint32_t slmOffset = hh * 16 * 24 * sizeof(float);
        if (v != 0) {
          cc_fp16.select<16, 1>(0) = __ESIMD_ENS::lsc_block_load<
              fp16,
              16,
              __ESIMD_ENS::lsc_data_size::default_size,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached>((fp16*)c + offsetC);
        } else {
          cc_fp16.select<16, 1>(0) = 0;
        }
#pragma unroll
        for (int k = 0; k < 4; k++) {
          bb.template bit_cast_view<float>().template select<64, 1>(64 * k) =
              slm_block_load<float, 64>(slmOffset);
          slmOffset += 64 * sizeof(float);
        }

#pragma unroll
        for (int k = 0; k < 16; k++) {
          cc_fp16.select<16, 1>(0) += bb.select<16, 1>(16 * k);
        }

#pragma unroll
        for (int k = 0; k < 2; k++) {
          bb.template bit_cast_view<float>().template select<64, 1>(64 * k) =
              slm_block_load<float, 64>(slmOffset);
          slmOffset += 64 * sizeof(float);
        }

#pragma unroll
        for (int k = 0; k < 8; k++) {
          cc_fp16.select<16, 1>(0) += bb.select<16, 1>(16 * k);
        }

        __ESIMD_ENS::lsc_block_store<
            fp16,
            16,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::write_back,
            __ESIMD_ENS::cache_hint::write_back>(
            (fp16*)c + offsetC, cc_fp16.select<16, 1>(0));

        offsetC += 8 * outputRow;
      }
    }
  }
}


ESIMD_INLINE void gemmReduce2048WeightsQ40InputFp16_ipex_FP32out(
    uint8_t* a,
    uint8_t* b,
    uint8_t* c,
    uint8_t* d,
    int hiddenDim,
    int tokenSize,
    int reduceIdx,
    int lastReduce,
    nd_item<2>& ndi) {
  constexpr uint32_t baseOffsetInc16[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  // constexpr uint32_t baseOffsetInc8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  // constexpr uint32_t baseOffsetInc4[4] = {0, 1, 2, 3};
  __ESIMD_NS::slm_init(64 * 8 * 16 * sizeof(fp16) + 16 * 32 * sizeof(fp16));
  int hh = ndi.get_local_linear_id(); // [0, 64)
  int h = ndi.get_group(0); // [0, (row + 15) / 16)
  int v = reduceIdx; // [0, (row + 15) / 16)
  int outputRow = ndi.get_group_range(0) * 16;
  int hiddenDimInt4Size = hiddenDim >> 1;
  int hiddenDimDequantSize = hiddenDim >> 5;
  uint32_t globalOffset = v * 2048 + h * hiddenDim * 16 + hh * 32;
  uint32_t baseOffsetA = globalOffset >> 1;
  uint32_t baseOffsetQuant =
      /*hiddenDimInt4Size * outputRow +*/ (globalOffset >> 4);
  uint32_t baseOffsetB = (v * 2048 + hh * 32) * sizeof(fp16);
  uint32_t offsetC = h * 16 + hh * outputRow;
  simd<fp16, 8 * 32> bb;
  simd<fp16, 8 * 32> bb_fp16;
  simd<fp16, 32 * 16> aa;
  simd<fp16, 8 * 16> cc;
  simd<fp16, 16> cc_fp16;
  simd<float, 16> cc_fp32;
  simd<uint32_t, 16> offset(baseOffsetInc16);
  simd<uint32_t, 16> offsetQuant(baseOffsetInc16);
  uint32_t loopCount = (tokenSize + 7) >> 3;

  offsetQuant =
      offsetQuant * hiddenDimDequantSize * sizeof(fp16) + baseOffsetQuant;

  {
    cc.template bit_cast_view<fp16>().template select<16, 1>(16 * 0) =
        __ESIMD_ENS::lsc_gather<
            fp16,
            1,
            __ESIMD_ENS::lsc_data_size::u16,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::uncached,
            16,
            uint32_t>((fp16*)d, offsetQuant);

    offsetQuant += 32 * sizeof(fp16);
  }

  cc.select<16, 1>(16) =
      cc.template bit_cast_view<fp16>().template select<16, 1>(0);

  offset = offset * hiddenDimInt4Size + baseOffsetA;

  {
    bb.template bit_cast_view<uint32_t>().template select<64, 1>(64 * 0) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            4,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::uncached,
            __ESIMD_ENS::cache_hint::uncached,
            16,
            uint32_t>((uint32_t*)a, offset);

// #pragma unroll
//     for (int kk = 0; kk < 4; kk++) {
//       simd<int8_t, 16> bitShiftTemp = bb.template
//       bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 0); aa.select<16,
//       1>((4 * kk + 0) * 16) = bitShiftTemp & 0xf; aa.select<16, 1>((4 * kk +
//       0) * 16 + 16 * 16) = bitShiftTemp >> 4; bitShiftTemp = bb.template
//       bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 1); aa.select<16,
//       1>((4 * kk + 1) * 16) = bitShiftTemp & 0xf; aa.select<16, 1>((4 * kk +
//       1) * 16 + 16 * 16) = bitShiftTemp >> 0x4; bitShiftTemp = bb.template
//       bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 2); aa.select<16,
//       1>((4 * kk + 2) * 16) = bitShiftTemp & 0xf; aa.select<16, 1>((4 * kk +
//       2) * 16 + 16 * 16) = bitShiftTemp >> 4; bitShiftTemp = bb.template
//       bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 3); aa.select<16,
//       1>((4 * kk + 3) * 16) = bitShiftTemp & 0xf; aa.select<16, 1>((4 * kk +
//       3) * 16 + 16 * 16) = bitShiftTemp >> 4;
//     }
//     offset += 128 * sizeof(uint32_t);
//   }
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      simd<uint8_t, 16> bitShiftTemp =
          bb.template bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 0);

      aa.select<16, 1>((4 * 2 * kk + 0) * 16) = bitShiftTemp & 0xf;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 1 * 16) = bitShiftTemp >> 4;

      // ==========================================================================================
      bitShiftTemp =
          bb.template bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 1);

      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 2 * 16) = bitShiftTemp & 0xf;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 3 * 16) = bitShiftTemp >> 4;

      // ==========================================================================================
      bitShiftTemp =
          bb.template bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 2);

      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 4 * 16) = bitShiftTemp & 0xf;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 5 * 16) = bitShiftTemp >> 4;

      // ==========================================================================================
      bitShiftTemp =
          bb.template bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 3);

      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 6 * 16) = bitShiftTemp & 0xf;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 7 * 16) = bitShiftTemp >> 4;
    }
    offset += 128 * sizeof(uint32_t);
  }

  aa = aa - 8.0f;
#pragma unroll
  for (int k = 0; k < 32; k++) {
    aa.select<16, 1>(16 * k) = aa.select<16, 1>(16 * k) * cc.select<16, 1>(16);
  }

  for (int nn = 0; nn < loopCount; nn++) {
    cc = 0;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      bb_fp16.template bit_cast_view<uint8_t>().template select<64, 1>(64 * k) =
          __ESIMD_ENS::lsc_block_load<
              uint8_t,
              64,
              __ESIMD_ENS::lsc_data_size::default_size,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + baseOffsetB);

#pragma unroll
      for (int kkk = 0; kkk < 32; kkk++) {
        cc.select<16, 1>(16 * k) +=
            aa.select<16, 1>(kkk * 16) * bb_fp16[k * 32 + kkk];
      }

      baseOffsetB += hiddenDim * sizeof(fp16);
    }

#pragma unroll
    for (int k = 0; k < 8; k++) {
      slm_block_store<fp16, 16>(
          (hh * 16 + k * 16 * 64) * sizeof(fp16), cc.select<16, 1>(16 * k));
    }

    barrier();

    if (hh < 32) {
      uint32_t slmOffset = hh * 16 * 16 * sizeof(fp16);
#pragma unroll
      for (int k = 0; k < 4; k++) {
        bb.select<64, 1>(64 * k) =
            slm_block_load<fp16, 64>(slmOffset + k * 64 * sizeof(fp16));
      }

#pragma unroll
      for (int k = 1; k < 8; k++) {
        bb.select<32, 1>(0) += bb.select<32, 1>(32 * k);
      }
      bb.select<16, 1>(0) += bb.select<16, 1>(16);
      slm_block_store<fp16, 16>(
          8 * 16 * 64 * sizeof(fp16) + hh * 16 * sizeof(fp16),
          bb.select<16, 1>(0));
    }

    barrier();

    if (hh < 8) {
      if (8 * nn + hh < tokenSize) {
        if (v != 0) {
          cc_fp32.select<16, 1>(0) = __ESIMD_ENS::lsc_block_load<
              float,
              16,
              __ESIMD_ENS::lsc_data_size::default_size,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached>((float*)c + offsetC);
        } else {
          cc_fp32.select<16, 1>(0) = 0;
        }
        cc_fp16.select<16, 1>(0) = cc_fp32.select<16, 1>(0);
        uint32_t slmOffset =
            hh * 16 * 4 * sizeof(fp16) + 16 * 64 * 8 * sizeof(fp16);
        bb.template bit_cast_view<fp16>().template select<64, 1>(0) =
            slm_block_load<fp16, 64>(slmOffset);

#pragma unroll
        for (int k = 0; k < 4; k++) {
          cc_fp16.select<16, 1>(0) += bb.select<16, 1>(16 * k);
        }

        cc_fp32.select<16, 1>(0) = cc_fp16.select<16, 1>(0);
        __ESIMD_ENS::lsc_block_store<
            float,
            16,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::write_back,
            __ESIMD_ENS::cache_hint::write_back>(
            (float*)c + offsetC, cc_fp32.select<16, 1>(0));

        offsetC += 8 * outputRow;
      }
    }
  }
}

ESIMD_INLINE void gemmReduce768WeightsQ40InputFp16_ipex_FP32out(
    uint8_t* a,
    uint8_t* b,
    uint8_t* c,
    uint8_t* d,
    int hiddenDim,
    int tokenSize,
    int reduceIdx,
    int lastReduce,
    nd_item<2>& ndi) {
  constexpr uint32_t baseOffsetInc16[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  // constexpr uint32_t baseOffsetInc8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  // constexpr uint32_t baseOffsetInc4[4] = {0, 1, 2, 3};
  __ESIMD_NS::slm_init(24 * 8 * 16 * sizeof(float) + 16 * 32 * sizeof(float));
  int hh = ndi.get_local_linear_id(); // [0, 64)
  int h = ndi.get_group(0); // [0, (row + 15) / 16)
  int v = reduceIdx; // [0, (row + 15) / 16)
  int outputRow = ndi.get_group_range(0) * 16;
  int hiddenDimInt4Size = hiddenDim >> 1;
  int hiddenDimDequantSize = hiddenDim >> 5;
  uint32_t globalOffset = v * 2048 + h * hiddenDim * 16 + hh * 32;
  uint32_t baseOffsetA = globalOffset >> 1;
  uint32_t baseOffsetQuant =
      /*hiddenDimInt4Size * outputRow +*/ (globalOffset >> 4);
  uint32_t baseOffsetB = (v * 2048 + hh * 32) * sizeof(fp16);
  uint32_t offsetC = h * 16 + hh * outputRow;
  simd<float, 8 * 32> bb;
  simd<fp16, 8 * 32> bb_fp16;
  simd<float, 32 * 16> aa;
  simd<float, 8 * 16> cc;
  simd<fp16, 16> cc_fp16;
  simd<fp16, 16> cc_fp32;
  simd<uint32_t, 16> offset(baseOffsetInc16);
  simd<uint32_t, 16> offsetQuant(baseOffsetInc16);
  uint32_t loopCount = (tokenSize + 7) >> 3;

  offsetQuant =
      offsetQuant * hiddenDimDequantSize * sizeof(fp16) + baseOffsetQuant;

  {
    cc.template bit_cast_view<fp16>().template select<16, 1>(16 * 0) =
        __ESIMD_ENS::lsc_gather<
            fp16,
            1,
            __ESIMD_ENS::lsc_data_size::u16,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::uncached,
            16,
            uint32_t>((fp16*)d, offsetQuant);

    offsetQuant += 12 * sizeof(fp16);
  }

  cc.select<16, 1>(16) =
      cc.template bit_cast_view<fp16>().template select<16, 1>(0);
  offset = offset * hiddenDimInt4Size + baseOffsetA;

  {
    bb.template bit_cast_view<uint32_t>().template select<64, 1>(64 * 0) =
        __ESIMD_ENS::lsc_gather<
            uint32_t,
            4,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::uncached,
            __ESIMD_ENS::cache_hint::uncached,
            16,
            uint32_t>((uint32_t*)a, offset);

#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      simd<u_char, 16> bitShiftTemp =
          bb.template bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 0);

      aa.select<16, 1>((4 * 2 * kk + 0) * 16) = bitShiftTemp & 0xf;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 1 * 16) = bitShiftTemp >> 4;

      // ==========================================================================================
      bitShiftTemp =
          bb.template bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 1);


      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 2 * 16) = bitShiftTemp & 0xf;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 3 * 16) = bitShiftTemp >> 4;

      // ==========================================================================================
      bitShiftTemp =
          bb.template bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 2);


      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 4 * 16) = bitShiftTemp & 0xf;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 5 * 16) = bitShiftTemp >> 4;

      // ==========================================================================================
      bitShiftTemp =
          bb.template bit_cast_view<uint8_t>().select<16, 4>(64 * kk + 3);

      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 6 * 16) = bitShiftTemp & 0xf;
      aa.select<16, 1>((4 * 2 * kk + 0) * 16 + 7 * 16) = bitShiftTemp >> 4;
    }
    offset += 48 * sizeof(uint32_t);
  }

  aa = aa - 8.0f;
#pragma unroll
  for (int k = 0; k < 32; k++) {
    aa.select<16, 1>(16 * k) = aa.select<16, 1>(16 * k) * cc.select<16, 1>(16);
  }

  for (int nn = 0; nn < loopCount; nn++) {
    cc = 0;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      bb_fp16.template bit_cast_view<uint8_t>().template select<64, 1>(64 * k) =
          __ESIMD_ENS::lsc_block_load<
              uint8_t,
              64,
              __ESIMD_ENS::lsc_data_size::default_size,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + baseOffsetB);

#pragma unroll
      for (int kkk = 0; kkk < 32; kkk++) {
        cc.select<16, 1>(16 * k) +=
            aa.select<16, 1>(kkk * 16) * bb_fp16[k * 32 + kkk];
      }

      baseOffsetB += hiddenDim * sizeof(fp16);
    }

    barrier();

#pragma unroll
    for (int k = 0; k < 8; k++) {
      slm_block_store<float, 16>(
          (hh * 16 + k * 16 * 24) * sizeof(float), cc.select<16, 1>(16 * k));
    }

    barrier();

    if (hh < 8) {
      if (8 * nn + hh < tokenSize) {
        uint32_t slmOffset = hh * 16 * 24 * sizeof(float);
        if (v != 0) {
          cc_fp32.select<16, 1>(0) = __ESIMD_ENS::lsc_block_load<
              float,
              16,
              __ESIMD_ENS::lsc_data_size::default_size,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached>((float*)c + offsetC);
        } else {
          cc_fp32.select<16, 1>(0) = 0;
        }
        cc_fp16.select<16, 1>(0) = cc_fp32.select<16, 1>(0);
#pragma unroll
        for (int k = 0; k < 4; k++) {
          bb.template bit_cast_view<float>().template select<64, 1>(64 * k) =
              slm_block_load<float, 64>(slmOffset);
          slmOffset += 64 * sizeof(float);
        }

#pragma unroll
        for (int k = 0; k < 16; k++) {
          cc_fp16.select<16, 1>(0) += bb.select<16, 1>(16 * k);
        }

#pragma unroll
        for (int k = 0; k < 2; k++) {
          bb.template bit_cast_view<float>().template select<64, 1>(64 * k) =
              slm_block_load<float, 64>(slmOffset);
          slmOffset += 64 * sizeof(float);
        }

#pragma unroll
        for (int k = 0; k < 8; k++) {
          cc_fp16.select<16, 1>(0) += bb.select<16, 1>(16 * k);
        }

        cc_fp32.select<16, 1>(0) = cc_fp16.select<16, 1>(0);
        __ESIMD_ENS::lsc_block_store<
            float,
            16,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::write_back,
            __ESIMD_ENS::cache_hint::write_back>(
            (float*)c + offsetC, cc_fp32.select<16, 1>(0));

        offsetC += 8 * outputRow;
      }
    }
  }
}
