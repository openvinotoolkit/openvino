/*******************************************************************************
 * Copyright 2024 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#ifndef GPU_OCL_TILE_OPS_H
#define GPU_OCL_TILE_OPS_H

float __builtin_IB_atomic_max_local_f32(__local float *, float);

__attribute__((overloadable)) float local_atomic_max(local float *p, float v) {
    return __builtin_IB_atomic_max_local_f32(p, v);
}

__attribute__((overloadable)) half local_atomic_max(
        local half *p, half v) { /* not implemented */
    return v;
}

__attribute__((overloadable)) uint local_atomic_max(local uint *p, uint v) {
    return atomic_max(p, v);
}

__attribute__((overloadable)) int local_atomic_max(local int *p, int v) {
    return atomic_max(p, v);
}

#define DEF_BLOCK_LOAD_STORE(type, itype, suffix, n) \
    __attribute__((overloadable)) type##n block_load( \
            const global type *p, int vlen) \
            __attribute__((enable_if(vlen == n, "wrong vector length"))) { \
        return as_##type##n( \
                intel_sub_group_block_read##suffix##n((global void *)p)); \
    } \
    __attribute__((overloadable)) void block_store( \
            global type *p, type##n v) { \
        intel_sub_group_block_write##suffix##n( \
                (global itype *)p, as_##itype##n(v)); \
    }

#define DEF_BLOCK_LOAD_STORE1(type, itype, suffix) \
    __attribute__((overloadable)) \
            type##1 block_load(const global type *p, int vlen) __attribute__( \
                    (enable_if(vlen == 1, "wrong vector length"))) { \
        type##1 x; \
        x[0] = as_##type( \
                intel_sub_group_block_read##suffix((global void *)p)); \
        return x; \
    } \
    __attribute__((overloadable)) void block_store( \
            global type *p, type##1 v) { \
        intel_sub_group_block_write##suffix( \
                (global itype *)p, as_##itype(v[0])); \
    }

#define DEF_BLOCK_LOAD_STORE16(type, itype, suffix) \
    __attribute__((overloadable)) \
            type##16 block_load(const global type *p, int vlen) __attribute__( \
                    (enable_if(vlen == 16, "wrong vector length"))) { \
        type##16 x; \
        x.s01234567 = as_##type##8( \
                intel_sub_group_block_read8##suffix((global void *)p)); \
        x.s89abcdef = as_##type##8( \
                intel_sub_group_block_read8##suffix((global void *)(p + 8 * get_sub_group_size()))); \
        return x; \
    } \
    __attribute__((overloadable)) void block_store( \
            global type *p, type##16 v) { \
        intel_sub_group_block_write8##suffix( \
                (global itype *)p, as_##itype##8(v.s01234567)); \
        intel_sub_group_block_write8##suffix( \
                (global itype *)(p + 8 * get_sub_group_size()), as_##itype##8(v.s89abcdef)); \
    }

DEF_BLOCK_LOAD_STORE1(half, ushort, _us)
DEF_BLOCK_LOAD_STORE(half, ushort, _us, 2)
DEF_BLOCK_LOAD_STORE(half, ushort, _us, 4)
DEF_BLOCK_LOAD_STORE(half, ushort, _us, 8)
DEF_BLOCK_LOAD_STORE(half, ushort, _us, 16)
DEF_BLOCK_LOAD_STORE1(uint, uint, )
DEF_BLOCK_LOAD_STORE(uint, uint, , 2)
DEF_BLOCK_LOAD_STORE(uint, uint, , 4)
DEF_BLOCK_LOAD_STORE(uint, uint, , 8)
DEF_BLOCK_LOAD_STORE16(uint, uint, )

#define DEF_BLOCK2D_LOAD_STORE(type, itype, vl, SG, suffix, BR, BC) \
    itype##vl __builtin_IB_subgroup_block_read_flat_##suffix( \
            long, int, int, int, int2); \
    void __builtin_IB_subgroup_block_write_flat_##suffix( \
            long, int, int, int, int2, itype##vl); \
    __attribute__((overloadable)) type##vl block2d_load(const global type *p, \
            int w, int h, int ld, int x, int y, int br, int bc, \
            int sg) __attribute__((enable_if(br == BR, "wrong #rows"))) \
            __attribute__((enable_if(bc == BC, "wrong #columns"))) \
                    __attribute__( \
                            (enable_if(sg == SG, "wrong subgroup size"))) { \
        ulong pp = as_long(p); \
        ulong prem = pp & 0x3F; \
        pp &= ~0x3F; \
        x += (prem / sizeof(type)); \
        w += prem; \
        int2 coord = {x, y}; \
        return as_##type##vl(__builtin_IB_subgroup_block_read_flat_##suffix( \
                pp, w - 1, h - 1, ld - 1, coord)); \
    } \
    __attribute__((overloadable)) void block2d_store(type##vl v, \
            global type *p, int w, int h, int ld, int x, int y, int br, \
            int bc, \
            int sg) __attribute__((enable_if(br == BR, "wrong #rows"))) \
            __attribute__((enable_if(bc == BC, "wrong #columns"))) \
                    __attribute__( \
                            (enable_if(sg == SG, "wrong subgroup size"))) { \
        ulong pp = as_long(p); \
        ulong prem = pp & 0x3F; \
        pp &= ~0x3F; \
        x += (prem / sizeof(type)); \
        w += prem; \
        int2 coord = {x, y}; \
        __builtin_IB_subgroup_block_write_flat_##suffix( \
                pp, w - 1, h - 1, ld - 1, coord, as_##itype##vl(v)); \
    }

DEF_BLOCK2D_LOAD_STORE(half, ushort, 8, 16, u16_m8k16v1, 16, 8)
DEF_BLOCK2D_LOAD_STORE(half, ushort, 8, 16, u16_m4k32v1, 32, 4)
DEF_BLOCK2D_LOAD_STORE(half, ushort, 16, 16, u16_m8k32v1, 32, 8)

#define tile_fill(t, v) \
    do { \
        _Pragma("unroll") for (int i = 0; i < sizeof(t.x) / sizeof(t.x[0]); \
                               i++) t.x[i] \
                = v; \
    } while (0)

#define tile_elementwise(t, f) \
    do { \
        _Pragma("unroll") for (int i = 0; i < sizeof(t.x) / sizeof(t.x[0]); \
                               i++) t.x[i] \
                = f(t.x[i]); \
    } while (0)

#define tile_elementwise_s(t, f) \
    do { \
        _Pragma("unroll") for (int i = 0; i < sizeof(t.x) / sizeof(t.x[0]); \
                               i++) { \
            _Pragma("unroll") for (int s = 0; \
                                   s < sizeof(t.x[0]) / sizeof(t.x[0][0]); \
                                   s++) t.x[i][s] \
                    = f(t.x[i][s]); \
        } \
    } while (0)

#define tile_binary(t, t2, f) \
    do { \
        _Pragma("unroll") for (int i = 0; i < sizeof(t.x) / sizeof(t.x[0]); \
                               i++) t.x[i] \
                = f(t.x[i], t2.x[i]); \
    } while (0)

#define tile_copy(t, t_new) \
    do { \
        _Pragma("unroll") for (int i = 0; i < sizeof(t.x) / sizeof(t.x[0]); \
                               i++) t_new.x[i] \
                = __builtin_convertvector(t.x[i], __typeof__(t_new.x[i])); \
    } while (0)

#define tile_copy_to_half2(t, t_new) \
    do { \
        _Pragma("unroll") for (int i = 0; i < sizeof(t.x) / sizeof(t.x[0]); \
                               i++) { \
            _Pragma("unroll") for (int s = 0; \
                                   s < sizeof(t.x[0]) / sizeof(t.x[0][0]) / 2; \
                                   s++) { \
                half2 v = {t.x[i][2 * s], t.x[i][2 * s + 1]}; \
                t_new.x[i][s] = as_uint(v); \
            } \
        } \
    } while (0)

#define tile_access(t, i0, j, sg, br, bc, nbr) \
    (t).x[(i0) / (br) + (nbr) * ((j) / (bc))] \
         [((i0) % (br)) / (sg) + ((j) % (bc)) * ((br) / (sg))]

#define xlane_tile_access(t, i, j, sg, br, bc, nbr) \
    sub_group_broadcast(tile_access(t, i, j, sg, br, bc, nbr), i % sg)

#define tile_predicated_assignment_t( \
        t, sg_offset_r, sg_offset_c, predicate, value, sg, br, bc, nbr, nbc) \
    do { \
        for (int j = 0; j < (bc * nbc); j++) { \
            for (int i0 = 0; i0 < (br * nbr); i0 += sg) { \
                int i = i0 + get_sub_group_local_id(); \
                int offset_r = sg_offset_r + j; \
                int offset_c = sg_offset_c + i; \
                if (predicate(offset_r, offset_c)) { \
                    tile_access(t, i0, j, sg, br, bc, nbr) = value; \
                } \
            } \
        } \
    } while (0)

#define DECLARE_2D_TILE_OPS(tile_type, element_type, sg, br, bc, nbr, nbc) \
    __attribute__((overloadable)) void tile_load_full(tile_type *t, \
            const global element_type *ptr, int ld, int offset_r, \
            int offset_c) { \
        ptr += ld * offset_c + offset_r; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr += ld) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                int i = i0 + get_sub_group_local_id(); \
                tile_access(*t, i0, j, sg, br, bc, nbr) = ptr[i]; \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_load_full(tile_type *t, \
            const local element_type *ptr, int ld, int offset_r, \
            int offset_c) { \
        ptr += ld * offset_c + offset_r; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr += ld) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                int i = i0 + get_sub_group_local_id(); \
                tile_access(*t, i0, j, sg, br, bc, nbr) = ptr[i]; \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_load(tile_type *t, \
            const global element_type *ptr, int m, int n, int ld, \
            int offset_r, int offset_c) { \
        if (m >= offset_r + br * nbr && n >= offset_c + bc * nbc) { \
            tile_load_full(t, ptr, ld, offset_r, offset_c); \
            return; \
        } \
        ptr += ld * offset_c + offset_r; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr += ld) { \
            if (offset_c + j < n) { \
                _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                    int i = i0 + get_sub_group_local_id(); \
                    if (offset_r + i < m) \
                        tile_access(*t, i0, j, sg, br, bc, nbr) = ptr[i]; \
                } \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_load(tile_type *t, \
            const global element_type *ptr, int m, int n, int offset_r, \
            int offset_c) { \
        tile_load(t, ptr, m, n, m, offset_r, offset_c); \
    } \
    __attribute__((overloadable)) void tile_load_t_full(tile_type *t, \
            const global element_type *ptr, int ld, int offset_r, \
            int offset_c) { \
        ptr += ld * offset_r + offset_c; \
        _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg, ptr += ld*sg) { \
            _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
                tile_access(*t, i0, j, sg, br, bc, nbr) = ptr[get_sub_group_local_id() * ld + j]; \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_load_t(tile_type *t, \
            const global element_type *ptr, int m, int n, int ld, \
            int offset_r, int offset_c) { \
        if (m >= offset_r + br * nbr && n >= offset_c + bc * nbc) { \
            tile_load_t_full(t, ptr, ld, offset_r, offset_c); \
            return; \
        } \
        ptr += ld * offset_r + offset_c; \
        _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg, ptr += ld*sg) { \
            int i = i0 + get_sub_group_local_id(); \
            if (offset_r + i < m) \
                _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
                    if (offset_c + j < n) { \
                        tile_access(*t, i0, j, sg, br, bc, nbr) = ptr[get_sub_group_local_id() * ld + j]; \
                } \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_load_t(tile_type *t, \
            const global element_type *ptr, int m, int n, int offset_r, \
            int offset_c) { \
        tile_load(t, ptr, m, n, n, offset_r, offset_c); \
    } \
    __attribute__((overloadable)) void tile_store_full(tile_type t, \
            local element_type *ptr, int ld, int offset_r, int offset_c) { \
        ptr += ld * offset_c + offset_r; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr += ld) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                int i = i0 + get_sub_group_local_id(); \
                ptr[i] = tile_access(t, i0, j, sg, br, bc, nbr); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_store_full(tile_type t, \
            global element_type *ptr, int ld, int offset_r, int offset_c) { \
        ptr += ld * offset_c + offset_r; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr += ld) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                int i = i0 + get_sub_group_local_id(); \
                ptr[i] = tile_access(t, i0, j, sg, br, bc, nbr); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_store(tile_type t, \
            global element_type *ptr, int m, int n, int ld, int offset_r, \
            int offset_c) { \
        if (m >= offset_r + br * nbr && n >= offset_c + bc * nbc) { \
            tile_store_full(t, ptr, ld, offset_r, offset_c); \
            return; \
        } \
        ptr += ld * offset_c + offset_r; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr += ld) { \
            if (offset_c + j < n) { \
                _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                    int i = i0 + get_sub_group_local_id(); \
                    if (offset_r + i < m) \
                        ptr[i] = tile_access(t, i0, j, sg, br, bc, nbr); \
                } \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_store(tile_type t, \
            global element_type *ptr, int m, int n, int offset_r, \
            int offset_c) { \
        tile_store(t, ptr, m, n, m, offset_r, offset_c); \
    } \
    __attribute__((overloadable)) void tile_store_t_sys_src1(tile_type t, \
            local element_type *ptr, int ld, int offset_r, int offset_c) { \
        offset_c += get_sub_group_local_id(); \
        int offset_r0 = offset_r & (sg - 1); \
        int offset_r1 = offset_r & ~(sg - 1); \
        ptr += offset_r0 + sg * offset_c + ld * offset_r1; \
        _Pragma("unroll") for (int j0 = 0; j0 < br * nbr; \
                               j0 += sg, ptr += sg * sg) { \
            _Pragma("unroll") for (int i = 0; i < bc * nbc; i++) ptr[i] \
                    = tile_access(t, j0, i, sg, br, bc, nbr); \
        } \
    } \
    __attribute__((overloadable)) void tile_store_t_sys_src2(tile_type t, \
            local element_type *ptr, int tile_n, int ld, int offset_r, \
            int offset_c) { \
        const int cp = 32 / sizeof(element_type); \
        offset_c += get_sub_group_local_id(); \
        int offset_r0 = offset_r & (cp - 1); \
        int offset_r1 = offset_r & ~(cp - 1); \
        ptr += offset_r0 + tile_n * offset_r1; \
        _Pragma("unroll") for (int j0 = 0; j0 < br * nbr; \
                               j0 += sg, offset_c += sg) { \
            int offset_c0 = offset_c & (tile_n - 1); \
            int offset_c1 = offset_c & ~(tile_n - 1); \
            local element_type *ptr_j = ptr + cp * offset_c0 + ld * offset_c1; \
            _Pragma("unroll") for (int i = 0; i < bc * nbc; i++) { \
                *ptr_j = tile_access(t, j0, i, sg, br, bc, nbr); \
                ptr_j++; \
                if ((~i & (cp - 1)) == 0) ptr_j += cp * (tile_n - 1); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_atomic_max_full(tile_type t, \
            local element_type *ptr, int ld, int offset_r, int offset_c) { \
        ptr += ld * offset_c + offset_r; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr += ld) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                int i = i0 + get_sub_group_local_id(); \
                (void)local_atomic_max( \
                        ptr + i, tile_access(t, i0, j, sg, br, bc, nbr)); \
            } \
        } \
    }

#define DECLARE_2D_TILE_VREDUCE(tile_type, sg, br, bc, nbr, nbc, rtile_type, \
        rsg, rbr, rbc, rnbr, rnbc) \
    __attribute__((overloadable)) void tile_vreduce_add( \
            tile_type t, rtile_type *tr) { \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                tile_access(*tr, i0, 0, rsg, rbr, rbc, rnbr) \
                        += tile_access(t, i0, j, sg, br, bc, nbr); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_vreduce_max( \
            tile_type t, rtile_type *tr) { \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                tile_access(*tr, i0, 0, rsg, rbr, rbc, rnbr) \
                        = max(tile_access(t, i0, j, sg, br, bc, nbr), \
                                tile_access(*tr, i0, 0, rsg, rbr, rbc, rnbr)); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_vbroadcast_sub( \
            tile_type *t, rtile_type tr) { \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                tile_access(*t, i0, j, sg, br, bc, nbr) \
                        -= tile_access(tr, i0, 0, rsg, rbr, rbc, rnbr); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_vbroadcast_mul( \
            tile_type *t, rtile_type tr) { \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                tile_access(*t, i0, j, sg, br, bc, nbr) \
                        *= tile_access(tr, i0, 0, rsg, rbr, rbc, rnbr); \
            } \
        } \
    }

#define DECLARE_2D_TILE_HREDUCE(tile_type, sg, br, bc, nbr, nbc, rtile_type, \
        rsg, rbr, rbc, rnbr, rnbc) \
    __attribute__((overloadable)) void tile_hbroadcast_add( \
            tile_type *t, rtile_type tr) { \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                tile_access(*t, i0, j, sg, br, bc, nbr) \
                        += xlane_tile_access(tr, j, 0, rsg, rbr, rbc, rnbr); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_hbroadcast_mul( \
            tile_type *t, rtile_type tr) { \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                tile_access(*t, i0, j, sg, br, bc, nbr) \
                        *= xlane_tile_access(tr, j, 0, rsg, rbr, rbc, rnbr); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_hbroadcast_min( \
            tile_type *t, rtile_type tr) { \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                tile_access(*t, i0, j, sg, br, bc, nbr) = min( \
                        tile_access(*t, i0, j, sg, br, bc, nbr), \
                        xlane_tile_access(tr, j, 0, rsg, rbr, rbc, rnbr)); \
            } \
        } \
    }

#define DECLARE_2D_TILE_RSELECT(tile_type0, sg0, br0, bc0, nbr0, nbc0, \
        tile_type1, sg1, br1, bc1, nbr1, nbc1) \
    __attribute__((overloadable)) void tile_rselect( \
            tile_type0 *t0, tile_type1 t1, int idx) { \
        _Pragma("unroll") for (int j = 0; j < bc0 * nbc0; j++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br0 * nbr0; i0 += sg0) { \
                tile_access(*t0, i0, j, sg0, br0, bc0, nbr0) \
                        = tile_access(t1, i0, j, sg1, br1, bc1, nbr1); \
                _Pragma("unroll") for (int z = 1; \
                                       z < (br1 * nbr1 / br0 * nbr0); \
                                       z++) if (z == idx) { \
                    tile_access(*t0, i0, j, sg0, br0, bc0, nbr0) \
                            = tile_access(t1, i0 + z * br0 * nbr0, j, sg1, \
                                    br1, bc1, nbr1); \
                } \
            } \
        } \
    }

#define DECLARE_2D_TILE_COPY_REBLOCK(tile_type0, sg0, br0, bc0, nbr0, nbc0, \
        tile_type1, sg1, br1, bc1, nbr1, nbc1) \
    __attribute__((overloadable)) void tile_copy_reblock( \
            tile_type0 t0, tile_type1 *t1) { \
        _Pragma("unroll") for (int j = 0; j < bc0 * nbc0; j++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br0 * nbr0; i0 += sg0) { \
                tile_access(*t1, i0, j, sg1, br1, bc1, nbr1) \
                        = tile_access(t0, i0, j, sg0, br0, bc0, nbr0); \
            } \
        } \
    }

#define DECLARE_2D_TILE(tile_type, element_type, sg, br, bc, nbr, nbc) \
    typedef element_type __attribute__((ext_vector_type(br * bc / sg))) \
            _e_##tile_type; \
    typedef struct { \
        _e_##tile_type x[nbr * nbc]; \
    } tile_type; \
    DECLARE_2D_TILE_OPS(tile_type, element_type, sg, br, bc, nbr, nbc)

/* Requires bc == 1 currently */
#define DECLARE_2D_TILE_BLOCK_OPS( \
        tile_type, element_type, sg, br, bc, nbr, nbc) \
    __attribute__((overloadable)) void tile_load_block(tile_type *t, \
            const global element_type *ptr, int ld, int offset_r, \
            int offset_c) { \
        ptr += ld * offset_c + offset_r; \
        _Pragma("unroll") for (int jj = 0; jj < nbc; jj++, ptr += ld * bc) { \
            _Pragma("unroll") for (int ii = 0; ii < nbr; ii++)(t) \
                    ->x[ii + nbr * jj] \
                    = block_load(ptr + ii * br, br / SUBGROUP_SIZE); \
        } \
    } \
    __attribute__((overloadable)) void tile_store_block(tile_type t, \
            global element_type *ptr, int ld, int offset_r, int offset_c) { \
        ptr += ld * offset_c + offset_r; \
        _Pragma("unroll") for (int jj = 0; jj < nbc; jj++, ptr += ld * bc) { \
            _Pragma("unroll") for (int ii = 0; ii < nbr; ii++) \
                    block_store(ptr + ii * br, (t).x[ii + nbr * jj]); \
        } \
    } \
    __attribute__((overloadable)) void tile_load_block(tile_type *t, \
            const global element_type *ptr, int n, int ld, int offset_r, \
            int offset_c) { \
        ptr += ld * offset_c + offset_r; \
        n -= offset_c; \
        _Pragma("unroll") for (int jj = 0; jj < nbc; jj++, ptr += ld * bc) { \
            if (jj < n) { \
                _Pragma("unroll") for (int ii = 0; ii < nbr; ii++)(t) \
                        ->x[ii + nbr * jj] \
                        = block_load(ptr + ii * br, br / SUBGROUP_SIZE); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_store_block(tile_type t, \
            global element_type *ptr, int n, int ld, int offset_r, \
            int offset_c) { \
        ptr += ld * offset_c + offset_r; \
        n -= offset_c; \
        _Pragma("unroll") for (int jj = 0; jj < nbc; jj++, ptr += ld * bc) { \
            if (jj < n) { \
                _Pragma("unroll") for (int ii = 0; ii < nbr; ii++) \
                        block_store(ptr + ii * br, (t).x[ii + nbr * jj]); \
            } \
        } \
    }

#define DECLARE_2D_TILE_BLOCK2D_OPS( \
        tile_type, element_type, sg, br, bc, nbr, nbc) \
    __attribute__((overloadable)) void tile_load_block2d(tile_type *t, \
            const global element_type *ptr, int m, int n, int ld, \
            int offset_r, int offset_c) { \
        const int e = sizeof(element_type); \
        _Pragma("unroll") for (int jj = 0; jj < nbc; jj++) { \
            _Pragma("unroll") for (int ii = 0; ii < nbr; ii++)(t) \
                    ->x[ii + nbr * jj] \
                    = block2d_load(ptr, m * e, n, ld * e, offset_r + ii * br, \
                            offset_c + jj * bc, br, bc, sg); \
        } \
    } \
    __attribute__((overloadable)) void tile_load_block2d(tile_type *t, \
            const global element_type *ptr, int m, int n, int offset_r, \
            int offset_c) { \
        tile_load_block2d(t, ptr, m, n, m, offset_r, offset_c); \
    } \
    __attribute__((overloadable)) void tile_store_block2d(tile_type t, \
            global element_type *ptr, int m, int n, int ld, int offset_r, \
            int offset_c) { \
        const int e = sizeof(element_type); \
        _Pragma("unroll") for (int jj = 0; jj < nbc; jj++) { \
            _Pragma("unroll") for (int ii = 0; ii < nbr; ii++) block2d_store( \
                    (t).x[ii + nbr * jj], ptr, m *e, n, ld *e, \
                    offset_r + ii * br, offset_c + jj * bc, br, bc, sg); \
        } \
    } \
    __attribute__((overloadable)) void tile_store_block2d(tile_type t, \
            const global element_type *ptr, int m, int n, int offset_r, \
            int offset_c) { \
        tile_store_block2d(t, ptr, m, n, m, offset_r, offset_c); \
    }

#define DECLARE_2D_TILE_LOAD_PACKED_HALF(tile_type, sg, br, bc, nbr, nbc) \
    __attribute__((overloadable)) void tile_load_packed_half(tile_type *t, \
            const global half *ptr, int m, int n, int ld, int offset_r, \
            int offset_c) { \
        ptr += ld * offset_c + offset_r; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr += ld) { \
            if (offset_c + j < n) { \
                _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                    int i = 2 * (i0 + get_sub_group_local_id()); \
                    half2 loaded = 0; \
                    if (offset_r + i < m) loaded.s0 = ptr[i]; \
                    if (offset_r + i + 1 < m) loaded.s1 = ptr[i + 1]; \
                    tile_access(*t, i0, j, sg, br, bc, nbr) = as_uint(loaded); \
                } \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_load_packed_half(tile_type *t, \
            const global half *ptr, int m, int n, int offset_r, \
            int offset_c) { \
        tile_load_packed_half(t, ptr, m, n, m, offset_r, offset_c); \
    }

#define cooperative_prefetch_2d(ptr, r, c, ld, sg_id, n_sg, sg_size, caching) \
    cooperative_prefetch_2d_internal((const global char *)ptr, \
            (r) * sizeof(*(ptr)), c, (ld) * sizeof(*(ptr)), sg_id, n_sg, \
            sg_size, caching)

#define cooperative_prefetch_2d_rem( \
        ptr, r, c, rmax, cmax, ld, sg_id, n_sg, sg_size, caching) \
    cooperative_prefetch_2d_internal((const global char *)ptr, \
            (r) * sizeof(*(ptr)), c, (rmax) * sizeof(*(ptr)), cmax, \
            (ld) * sizeof(*(ptr)), sg_id, n_sg, sg_size, caching)

/* IGC prefetch intrinsics */
enum LSC_LDCC {
    LSC_LDCC_DEFAULT = 0,
    LSC_LDCC_L1UC_L3UC = 1,
    LSC_LDCC_L1UC_L3C = 2,
    LSC_LDCC_L1C_L3UC = 3,
    LSC_LDCC_L1C_L3C = 4,
    LSC_LDCC_L1S_L3UC = 5,
    LSC_LDCC_L1S_L3C = 6,
    LSC_LDCC_L1IAR_L3C = 7,
};

extern void __builtin_IB_lsc_prefetch_global_uchar(
        const __global uchar *base, int immElemOff, enum LSC_LDCC cacheOpt);

extern void __builtin_IB_lsc_prefetch_global_uint(
        const __global uint *base, int immElemOff, enum LSC_LDCC cacheOpt);

__attribute__((overloadable)) void cooperative_prefetch_2d_internal(
        const global char *ptr, uint rbytes, uint c, uint ld_bytes, uint sg_id,
        uint n_sg, uint sg_size, enum LSC_LDCC caching) {
    const uint cl_per_col = (rbytes + 63) >> 6;
    const uint cl = cl_per_col * c;
    const uint cl_per_sg = (cl + n_sg - 1) / n_sg;
    const uint cl_iters = (cl_per_sg + sg_size - 1) / sg_size;
#pragma unroll
    for (uint ii_cl = 0; ii_cl < cl_iters; ii_cl++) {
        uint i_cl = ii_cl + (sg_id * cl_per_sg) + get_sub_group_local_id();
        uint r_cl = i_cl % cl_per_col;
        uint c_cl = i_cl / cl_per_col;
        if (i_cl < cl) {
            __builtin_IB_lsc_prefetch_global_uint(
                    (const global uint *)(ptr + r_cl * 64 + c_cl * ld_bytes), 0,
                    caching);
        }
    }
}

__attribute__((overloadable)) void cooperative_prefetch_2d_internal(
        const global char *ptr, uint rbytes, uint c, uint rbytes_max,
        uint c_max, uint ld_bytes, uint sg_id, uint n_sg, uint sg_size,
        enum LSC_LDCC caching) {
    const uint cl_per_col = (rbytes_max + 63) >> 6;
    const uint cl = cl_per_col * c_max;
    const uint cl_per_sg = (cl + n_sg - 1) / n_sg;
    const uint cl_iters = (cl_per_sg + sg_size - 1) / sg_size;
    const uint max_off = rbytes - 1 + (c - 1) * ld_bytes;
#pragma unroll
    for (uint ii_cl = 0; ii_cl < cl_iters; ii_cl++) {
        uint i_cl = ii_cl + (sg_id * cl_per_sg) + get_sub_group_local_id();
        uint r_cl = i_cl % cl_per_col;
        uint c_cl = i_cl / cl_per_col;
        uint pf_off = min(r_cl * 64 + c_cl * ld_bytes, max_off);
        if (i_cl < cl) {
            __builtin_IB_lsc_prefetch_global_uchar(
                    (const global uchar *)(ptr + pf_off), 0, caching);
        }
    }
}

#endif
