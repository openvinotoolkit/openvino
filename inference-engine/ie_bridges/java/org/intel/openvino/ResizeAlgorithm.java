package org.intel.openvino;

public enum ResizeAlgorithm {
    NO_RESIZE(0),
    RESIZE_BILINEAR(1),
    RESIZE_AREA(2);

    private int value;

    private ResizeAlgorithm(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }
}
