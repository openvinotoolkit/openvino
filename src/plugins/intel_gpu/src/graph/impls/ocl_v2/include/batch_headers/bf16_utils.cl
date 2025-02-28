#ifdef intel_convert_as_bfloat16_float
#define _convert_as_bfloat16_float(val) intel_convert_as_bfloat16_float(val)
#else
inline float _convert_as_bfloat16_float(ushort source) {
    uint u = 0;
    //sign
    if ( (source>>15) ) { 
        u = 1 << 31;
    }
    //exponent
    u += ( ( (source >> 7) & 0b11111111)) << 23;
    //fraction 
    u += (source & 0b1111111) << 16;
    float* f = &u;
    return *f;
}
#endif

#ifdef intel_convert_bfloat16_as_ushort
#define _convert_bfloat16_as_ushort(val) intel_convert_bfloat16_as_ushort(val)
#else
inline ushort _convert_bfloat16_as_ushort(float source) {
    uint* in = &source;
    ushort u = 0;
    if ( (*in>>31) ) { 
        u = 1 << 15;
    }
    //exponent
    u += ( ( (*in >> 23) & 0b11111111)) << 7;
    //fraction
    u += (*in >> 16) & 0b1111111;
    return u;
}
#endif
