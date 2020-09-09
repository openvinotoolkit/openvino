import org.intel.openvino.*;

import java.util.Vector;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

public class InferRequestsQueue {
    public InferRequestsQueue(ExecutableNetwork net, int nireq) {
        for (int id = 0; id < nireq; id++) {
            requests.add(new InferReqWrap(net, id, this));
            idleIds.add(id);
        }
        resetTimes();
    }

    void resetTimes() {
        startTime = Long.MAX_VALUE;
        endTime = Long.MIN_VALUE;
        latencies.clear();
    }

    double getDurationInMilliseconds() {
        return (double) (endTime - startTime) * 1e-6;
    }

    void putIdleRequest(int id, double latency) {
        latencies.add(latency);
        idleIds.add(id);
        endTime = Math.max(System.nanoTime(), endTime);

        synchronized (foo) {
            foo.notify();
        }
    }

    InferReqWrap getIdleRequest() {
        try {
            InferReqWrap request = requests.get(idleIds.take());
            startTime = Math.min(System.nanoTime(), startTime);
            return request;
        } catch (InterruptedException e) {
            System.out.println(e.getMessage());
        }
        return null;
    }

    void waitAll() {
        synchronized (foo) {
            try {
                while (idleIds.size() != requests.size()) {
                    foo.wait();
                }
            } catch (InterruptedException e) {
                System.out.println("InterruptedException: " + e.getMessage());
            }
        }
    }

    Vector<Double> getLatencies() {
        return latencies;
    }

    Vector<InferReqWrap> requests = new Vector<InferReqWrap>();
    private BlockingQueue<Integer> idleIds = new LinkedBlockingQueue<Integer>();
    private long startTime;
    private long endTime;
    Vector<Double> latencies = new Vector<Double>();

    Object foo = new Object();
}
