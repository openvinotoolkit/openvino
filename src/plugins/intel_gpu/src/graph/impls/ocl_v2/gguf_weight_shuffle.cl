#ifndef GGUF_SHUFFLE_FORMAT
#error GGUF_SHUFFLE_FORMAT must select a supported GGUF block format
#endif

inline void gguf_scatter_plane(__global uchar* destination,
                               const __private uchar* source,
                               size_t bytes,
                               size_t lane) {
    for (size_t offset = 0; offset < bytes; ++offset)
        destination[(offset / 4) * 64 + lane * 4 + offset % 4] = source[offset];
}

__kernel void gguf_weight_shuffle(__global const uchar* source,
                                  __global uchar* destination,
                                  ulong blocks_per_row,
                                  ulong total_blocks) {
    const size_t block_index = get_global_id(0);
    if (block_index >= total_blocks)
        return;
    const size_t row = block_index / blocks_per_row;
    const size_t column = block_index % blocks_per_row;
    const size_t lane = row % 16;
    const size_t entry = (row / 16) * blocks_per_row + column;
    uchar low_plane[256];

#if GGUF_SHUFFLE_FORMAT == 0 || GGUF_SHUFFLE_FORMAT == 1
#if GGUF_SHUFFLE_FORMAT == 0
    const size_t block_bytes = 144;
    const size_t quant_offset = 16;
#else
    const size_t block_bytes = 176;
    const size_t quant_offset = 48;
#endif
    __global const uchar* block = source + block_index * block_bytes;
    uchar quants[256];
    uchar scales[8];
    uchar minimums[8];
    for (size_t section = 0; section < 4; ++section) {
        for (size_t index = 0; index < 32; ++index) {
            const uchar packed = block[quant_offset + section * 32 + index];
            quants[section * 64 + index] = packed & 15;
            quants[section * 64 + 32 + index] = packed >> 4;
#if GGUF_SHUFFLE_FORMAT == 1
            quants[section * 64 + index] |= ((block[16 + index] >> (2 * section)) & 1) << 4;
            quants[section * 64 + 32 + index] |= ((block[16 + index] >> (2 * section + 1)) & 1) << 4;
#endif
        }
    }
    for (size_t section = 0; section < 8; ++section) {
        if (section < 4) {
            scales[section] = block[4 + section] & 63;
            minimums[section] = block[8 + section] & 63;
        } else {
            scales[section] = (block[8 + section] & 15) | ((block[section] >> 6) << 4);
            minimums[section] = (block[8 + section] >> 4) | ((block[4 + section] >> 6) << 4);
        }
        for (size_t index = 0; index < 16; ++index)
            low_plane[section * 16 + index] = (quants[section * 32 + index] & 15) |
                                                           ((quants[section * 32 + 16 + index] & 15) << 4);
    }
    gguf_scatter_plane(destination + entry * 16 * 128, low_plane, 128, lane);
#if GGUF_SHUFFLE_FORMAT == 1
    uchar high_plane[32];
    for (size_t section = 0; section < 8; ++section) {
        for (size_t index = 0; index < 4; ++index) {
            uchar packed = 0;
            for (size_t bit = 0; bit < 8; ++bit)
                packed |= ((quants[section * 32 + index + bit * 4] >> 4) & 1) << bit;
            high_plane[section * 4 + index] = packed;
        }
    }
    gguf_scatter_plane(destination + total_blocks * 128 + entry * 16 * 32, high_plane, 32, lane);
    const size_t metadata_base = total_blocks * 160 + entry * 16 * 16;
#else
    const size_t metadata_base = total_blocks * 128 + entry * 16 * 16;
#endif
    for (size_t index = 0; index < 4; ++index) {
        destination[metadata_base + lane * 4 + index] = (scales[index * 2] & 15) | ((scales[index * 2 + 1] & 15) << 4);
        destination[metadata_base + 64 + lane * 4 + index] = (minimums[index * 2] & 15) | ((minimums[index * 2 + 1] & 15) << 4);
    }
    for (size_t index = 0; index < 2; ++index) {
        const size_t first = index * 4;
        destination[metadata_base + 128 + lane * 2 + index] = (scales[first] >> 4) |
            ((scales[first + 1] >> 4) << 2) | ((scales[first + 2] >> 4) << 4) | ((scales[first + 3] >> 4) << 6);
        destination[metadata_base + 160 + lane * 2 + index] = (minimums[first] >> 4) |
            ((minimums[first + 1] >> 4) << 2) | ((minimums[first + 2] >> 4) << 4) | ((minimums[first + 3] >> 4) << 6);
        destination[metadata_base + 192 + lane * 2 + index] = block[index];
        destination[metadata_base + 224 + lane * 2 + index] = block[2 + index];
    }
#elif GGUF_SHUFFLE_FORMAT == 2
    __global const uchar* block = source + block_index * 210;
    uchar quants[256];
    uchar high_plane[64];
    for (size_t half_index = 0; half_index < 2; ++half_index) {
        for (size_t index = 0; index < 64; ++index) {
            const uchar packed = block[half_index * 64 + index];
            quants[half_index * 128 + index] = packed & 15;
            quants[half_index * 128 + 64 + index] = packed >> 4;
        }
        for (size_t index = 0; index < 32; ++index) {
            const uchar packed = block[128 + half_index * 32 + index];
            for (size_t section = 0; section < 4; ++section)
                quants[half_index * 128 + section * 32 + index] |= ((packed >> (section * 2)) & 3) << 4;
        }
    }
    for (size_t section = 0; section < 8; ++section)
        for (size_t index = 0; index < 16; ++index)
            low_plane[section * 16 + index] = (quants[section * 32 + index] & 15) |
                                                           ((quants[section * 32 + 16 + index] & 15) << 4);
    for (size_t section = 0; section < 16; ++section) {
        for (size_t index = 0; index < 4; ++index) {
            uchar packed = 0;
            for (size_t part = 0; part < 4; ++part)
                packed |= ((quants[section * 16 + index + part * 4] >> 4) & 3) << (part * 2);
            high_plane[section * 4 + index] = packed;
        }
    }
    gguf_scatter_plane(destination + entry * 16 * 128, low_plane, 128, lane);
    gguf_scatter_plane(destination + total_blocks * 128 + entry * 16 * 64, high_plane, 64, lane);
    for (size_t index = 0; index < 16; ++index)
        destination[total_blocks * 192 + entry * 16 * 16 + index * 16 + lane] = block[192 + index];
    for (size_t index = 0; index < 2; ++index)
        destination[total_blocks * 208 + entry * 16 * 2 + lane * 2 + index] = block[208 + index];
#else
#if GGUF_SHUFFLE_FORMAT == 3
    const size_t block_bytes = 18;
    const size_t quant_offset = 2;
    const size_t plane_bytes = 128;
#elif GGUF_SHUFFLE_FORMAT == 4
    const size_t block_bytes = 20;
    const size_t quant_offset = 4;
    const size_t plane_bytes = 128;
#elif GGUF_SHUFFLE_FORMAT == 5
    const size_t block_bytes = 34;
    const size_t quant_offset = 2;
    const size_t plane_bytes = 256;
#else
#error Unsupported GGUF_SHUFFLE_FORMAT
#endif
    __global const uchar* super_block = source + block_index * 8 * block_bytes;
    for (size_t section = 0; section < 8; ++section) {
        __global const uchar* block = super_block + section * block_bytes;
        for (size_t index = 0; index < plane_bytes / 8; ++index)
            low_plane[section * (plane_bytes / 8) + index] = block[quant_offset + index];
        for (size_t index = 0; index < 2; ++index) {
            destination[total_blocks * plane_bytes + entry * 16 * 16 + section * 32 + lane * 2 + index] = block[index];
#if GGUF_SHUFFLE_FORMAT == 4
            destination[total_blocks * (plane_bytes + 16) + entry * 16 * 16 + section * 32 + lane * 2 + index] = block[2 + index];
#endif
        }
    }
    gguf_scatter_plane(destination + entry * 16 * plane_bytes, low_plane, plane_bytes, lane);
#endif
}