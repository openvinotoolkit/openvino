package org.intel.openvino;

import java.util.HashMap;
import java.util.Map;

public enum Layout {
    ANY(0),
    NCHW(1),
    NHWC(2),
    NCDHW(3),
    NDHWC(4),

    OIHW(64),

    SCALAR(95),

    C(96),

    CHW(128),

    HW(192),
    NC(193),
    CN(194),

    BLOCKED(200);

    private int value;
    private static Map<Integer, Layout> map = new HashMap<Integer, Layout>();

    static {
        for (Layout layout : Layout.values()) {
            map.put(layout.value, layout);
        }
    }

    private Layout(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }

    static Layout valueOf(int value) {
        return map.get(value);
    }
}
