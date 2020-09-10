package org.intel.openvino;

import java.util.HashMap;
import java.util.Map;

public class InferenceEngineProfileInfo {
    public enum LayerStatus {
        NOT_RUN(0),
        OPTIMIZED_OUT(1),
        EXECUTED(2);

        private int value;
        private static Map<Integer, LayerStatus> map = new HashMap<Integer, LayerStatus>();

        static {
            for (LayerStatus layerStatus : LayerStatus.values()) {
                map.put(layerStatus.value, layerStatus);
            }
        }

        LayerStatus(int value) {
            this.value = value;
        }

        int getValue() {
            return value;
        }

        static LayerStatus valueOf(int value) {
            return map.get(value);
        }
    }

    public LayerStatus status;
    public long realTimeUSec;
    public long cpuUSec;
    public String execType;
    public String layerType;
    public int executionIndex;

    public InferenceEngineProfileInfo(
            LayerStatus status,
            long realTimeUSec,
            long cpuUSec,
            String execType,
            String layerType,
            int executionIndex) {
        this.status = status;
        this.realTimeUSec = realTimeUSec;
        this.cpuUSec = cpuUSec;
        this.execType = execType;
        this.layerType = layerType;
        this.executionIndex = executionIndex;
    }
}
