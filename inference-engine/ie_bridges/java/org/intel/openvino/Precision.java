package org.intel.openvino;

import java.util.HashMap;
import java.util.Map;

public enum Precision {
    UNSPECIFIED(255),
    MIXED(0),
    FP32(10),
    FP16(11),
    Q78(20),
    I16(30),
    U8(40),
    I8(50),
    U16(60),
    I32(70),
    I64(72),
    BIN(71),
    CUSTOM(80);

    private int value;
    private static Map<Integer, Precision> map = new HashMap<Integer, Precision>();

    static {
        for (Precision precision : Precision.values()) {
            map.put(precision.value, precision);
        }
    }

    private Precision(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }

    static Precision valueOf(int value) {
        return map.get(value);
    }
}
