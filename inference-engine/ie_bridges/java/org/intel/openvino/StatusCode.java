package org.intel.openvino;

import java.util.HashMap;
import java.util.Map;

public enum StatusCode {
    OK(0),
    GENERAL_ERROR(-1),
    NOT_IMPLEMENTED(-2),
    NETWORK_NOT_LOADED(-3),
    PARAMETER_MISMATCH(-4),
    NOT_FOUND(-5),
    OUT_OF_BOUNDS(-6),
    UNEXPECTED(-7),
    REQUEST_BUSY(-8),
    RESULT_NOT_READY(-9),
    NOT_ALLOCATED(-10),
    INFER_NOT_STARTED(-11),
    NETWORK_NOT_READ(-12);

    private int value;

    private static Map<Integer, StatusCode> map = new HashMap<Integer, StatusCode>();

    static {
        for (StatusCode statusCode : StatusCode.values()) {
            map.put(statusCode.value, statusCode);
        }
    }

    private StatusCode(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }

    public static StatusCode valueOf(int value) {
        return map.get(value);
    }
}
