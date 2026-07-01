inline float convert_as_int4_float(uchar source, uint index) {
    float out;
    if (index % 2 == 0) {
       out = source & 0xF;
    } else {
       out = source >> 4;
    }

    if (out > 7.f)
        out -= 16.f;

    return out;
}

inline float convert_as_uint4_float(uchar source, uint index) {
    float out;
    if (index % 2 == 0) {
       out = source & 0xF;
    } else {
       out = source >> 4;
    }

    return out;
}
