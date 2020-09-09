import org.intel.openvino.*;

import java.util.Map;

public class InferReqWrap {

    public InferReqWrap(ExecutableNetwork net, int id, InferRequestsQueue irQueue) {
        request = net.CreateInferRequest();
        this.id = id;
        this.irQueue = irQueue;
        request.SetCompletionCallback(
                new Runnable() {

                    @Override
                    public void run() {
                        endTime = System.nanoTime();
                        irQueue.putIdleRequest(id, getExecutionTimeInMilliseconds());
                    }
                });
    }

    void startAsync() {
        startTime = System.nanoTime();
        request.StartAsync();
    }

    void _wait() {
        request.Wait(WaitMode.RESULT_READY);
    }

    void infer() {
        startTime = System.nanoTime();
        request.Infer();
        endTime = System.nanoTime();
        irQueue.putIdleRequest(id, getExecutionTimeInMilliseconds());
    }

    Map<String, InferenceEngineProfileInfo> getPerformanceCounts() {
        return request.GetPerformanceCounts();
    }

    Blob getBlob(String name) {
        return request.GetBlob(name);
    }

    double getExecutionTimeInMilliseconds() {
        return (double) (endTime - startTime) * 1e-6;
    }

    InferRequest request;
    private InferRequestsQueue irQueue;
    private long startTime;
    private long endTime;
    int id;
}
