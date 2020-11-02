import org.intel.openvino.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Vector;

public class Main {

    static boolean adjustShapesBatch(
            Map<String, int[]> shapes, int batchSize, Map<String, InputInfo> inputInfo) {
        boolean updated = false;

        for (Map.Entry<String, InputInfo> entry : inputInfo.entrySet()) {
            Layout layout = entry.getValue().getTensorDesc().getLayout();
            int batchIndex = -1;
            if ((layout == Layout.NCHW)
                    || (layout == Layout.NCDHW)
                    || (layout == Layout.NHWC)
                    || (layout == Layout.NDHWC)
                    || (layout == Layout.NC)) {
                batchIndex = 0;
            } else if (layout == Layout.CN) {
                batchIndex = 1;
            }
            if ((batchIndex != -1) && (shapes.get(entry.getKey())[batchIndex] != batchSize)) {
                shapes.get(entry.getKey())[batchIndex] = batchSize;
                updated = true;
            }
        }
        return updated;
    }

    static String setThroughputStreams(
            IECore core,
            Map<String, String> device_config,
            String device,
            int nstreams,
            boolean isAsync) {
        String key = device + "_THROUGHPUT_STREAMS";
        if (nstreams > 0) {
            device_config.put(key, Integer.toString(nstreams));
        } else if (!device_config.containsKey(key) && isAsync) {
            System.err.println(
                    "[ WARNING ] -nstreams default value is determined automatically for "
                            + device
                            + " device. Although the automatic selection usually provides a"
                            + " reasonable performance,but it still may be non-optimal for some"
                            + " cases, for more information look at README.");
            device_config.put(key, device + "_THROUGHPUT_AUTO");
        }
        return device_config.get(key);
    }

    static void fillBlobs(Vector<InferReqWrap> requests, Map<String, InputInfo> inputsInfo) {
        for (Map.Entry<String, InputInfo> entry : inputsInfo.entrySet()) {
            String inputName = entry.getKey();
            TensorDesc tDesc = entry.getValue().getTensorDesc();

            System.err.print(
                    "[ INFO ] Network input '"
                            + inputName
                            + "' precision "
                            + tDesc.getPrecision()
                            + ", dimensions ("
                            + tDesc.getLayout()
                            + "): ");

            for (int dim : tDesc.getDims()) System.err.print(dim + " ");
            System.err.println();
        }

        for (int i = 0; i < requests.size(); i++) {
            InferRequest request = requests.get(i).request;
            for (Map.Entry<String, InputInfo> entry : inputsInfo.entrySet()) {
                String inputName = entry.getKey();
                TensorDesc tDesc = entry.getValue().getTensorDesc();
                request.SetBlob(inputName, blobRandomByte(tDesc));
            }
        }
    }

    static Blob blobRandomByte(TensorDesc tDesc) {
        int dims[] = tDesc.getDims();

        int size = 1;
        for (int i = 0; i < dims.length; i++) {
            size *= dims[i];
        }

        byte[] buff = new byte[size];
        Random rand = new Random();
        rand.nextBytes(buff);

        return new Blob(tDesc, buff);
    }

    static double getMedianValue(Vector<Double> vec) {
        Object[] objArr = vec.toArray();
        Double[] arr = Arrays.copyOf(objArr, objArr.length, Double[].class);

        Arrays.sort(arr);

        if (arr.length % 2 == 0)
            return ((double) arr[arr.length / 2] + (double) arr[arr.length / 2 - 1]) / 2;
        else return (double) arr[arr.length / 2];
    }

    static boolean getApiBoolean(String api) throws RuntimeException {
        if (api.equals("sync")) return false;
        else if (api.equals("async")) return true;
        else throw new RuntimeException("Incorrect argument: '-api'");
    }

    static int step = 0;

    static void nextStep(String stepInfo) {
        step += 1;
        System.out.println("[Step " + step + "/11] " + stepInfo);
    }

    static int deviceDefaultDeviceDurationInSeconds(String device) {
        final Map<String, Integer> deviceDefaultDurationInSeconds =
                new HashMap<String, Integer>() {
                    {
                        put("CPU", 60);
                        put("GPU", 60);
                        put("VPU", 60);
                        put("MYRIAD", 60);
                        put("HDDL", 60);
                        put("FPGA", 120);
                        put("UNKNOWN", 120);
                    }
                };

        Integer duration = deviceDefaultDurationInSeconds.get(device);

        if (duration == null) {
            duration = deviceDefaultDurationInSeconds.get("UNKNOWN");
            System.err.println(
                    "[ WARNING ] Default duration "
                            + duration
                            + " seconds for unknown device '"
                            + device
                            + "' is used");
        }
        return duration;
    }

    static long getTotalMsTime(long startTimeMilliSec) {
        return (System.currentTimeMillis() - startTimeMilliSec);
    }

    static long getDurationInMilliseconds(int seconds) {
        return seconds * 1000L;
    }

    public static void main(String[] args) {
        try {
            System.loadLibrary(IECore.NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Failed to load Inference Engine library\n" + e);
            System.exit(1);
        }

        // ----------------- 1. Parsing and validating input arguments -----------------
        nextStep("Parsing and validating input arguments");

        ArgumentParser parser = new ArgumentParser("This is benchmarking application");
        parser.addArgument("-m", "path to model .xml");
        parser.addArgument("-d", "device");
        parser.addArgument("-nireq", "number of infer requests");
        parser.addArgument("-niter", "number of iterations");
        parser.addArgument("-b", "batch size");
        parser.addArgument("-nthreads", "number of threads");
        parser.addArgument("-nstreams", "number of streams");
        parser.addArgument("-t", "time limit");
        parser.addArgument("-api", "sync or async");
        parser.parseArgs(args);

        String xmlPath = parser.get("-m", null);
        String device = parser.get("-d", "CPU");
        int nireq = parser.getInteger("-nireq", 0);
        int niter = parser.getInteger("-niter", 0);
        int batchSize = parser.getInteger("-b", 0);
        int nthreads = parser.getInteger("-nthreads", 0);
        int nstreams = parser.getInteger("-nstreams", 0);
        int timeLimit = parser.getInteger("-t", 0);
        String api = parser.get("-api", "async");
        boolean isAsync;

        try {
            isAsync = getApiBoolean(api);
        } catch (RuntimeException e) {
            System.out.println(e.getMessage());
            return;
        }

        if (xmlPath == null) {
            System.out.println("Error: Missed argument: -m");
            return;
        }

        // ----------------- 2. Loading the Inference Engine --------------------------
        nextStep("Loading the Inference Engine");

        IECore core = new IECore();

        // ----------------- 3. Setting device configuration --------------------------
        nextStep("Setting device configuration");

        Map<String, String> device_config = new HashMap<>();

        if (device.equals("CPU")) { // CPU supports few special performance-oriented keys
            // limit threading for CPU portion of inference
            if (nthreads > 0) device_config.put("CPU_THREADS_NUM", Integer.toString(nthreads));

            if (!device_config.containsKey("CPU_BIND_THREAD")) {
                device_config.put("CPU_BIND_THREAD", "YES");
            }

            // for CPU execution, more throughput-oriented execution via streams
            setThroughputStreams(core, device_config, device, nstreams, isAsync);
        } else if (device.equals("GPU")) {
            // for GPU execution, more throughput-oriented execution via streams
            setThroughputStreams(core, device_config, device, nstreams, isAsync);
        } else if (device.equals("MYRIAD")) {
            device_config.put("LOG_LEVEL", "LOG_WARNING");
        } else if (device.equals("GNA")) {
            device_config.put("GNA_PRECISION", "I16");

            if (nthreads > 0) device_config.put("GNA_LIB_N_THREADS", Integer.toString(nthreads));
        }

        core.SetConfig(device_config, device);

        // ----------- 4. Reading the Intermediate Representation network -------------
        nextStep("Reading the Intermediate Representation network");

        long startTime = System.currentTimeMillis();
        CNNNetwork net = core.ReadNetwork(xmlPath);
        long durationMs = getTotalMsTime(startTime);

        System.err.println("[ INFO ] Read network took " + durationMs + " ms");

        Map<String, InputInfo> inputsInfo = net.getInputsInfo();
        String inputName = new ArrayList<String>(inputsInfo.keySet()).get(0);
        InputInfo inputInfo = inputsInfo.get(inputName);

        // ----- 5. Resizing network to match image sizes and given batch --------------
        nextStep("Resizing network to match image sizes and given batch");

        int inputBatchSize = batchSize;
        batchSize = net.getBatchSize();

        Map<String, int[]> shapes = net.getInputShapes();

        if ((inputBatchSize != 0) && (batchSize != inputBatchSize)) {
            adjustShapesBatch(shapes, batchSize, inputsInfo);

            startTime = System.currentTimeMillis();
            net.reshape(shapes);
            durationMs = getTotalMsTime(startTime);
            batchSize = net.getBatchSize();

            System.err.println("[ INFO ] Reshape network took " + durationMs + " ms");
        }

        System.err.println(
                (inputBatchSize != 0
                                ? "[ INFO ] Network batch size was changed to: "
                                : "[ INFO ] Network batch size: ")
                        + batchSize);

        // ----------------- 6. Configuring input -------------------------------------
        nextStep("Configuring input");

        inputInfo.getPreProcess().setResizeAlgorithm(ResizeAlgorithm.RESIZE_BILINEAR);
        inputInfo.setPrecision(Precision.U8);

        // ----------------- 7. Loading the model to the device -----------------------
        nextStep("Loading the model to the device");

        startTime = System.currentTimeMillis();
        ExecutableNetwork executableNetwork = core.LoadNetwork(net, device);
        durationMs = getTotalMsTime(startTime);

        System.err.println("[ INFO ] Load network took " + durationMs + " ms");

        // ---------------- 8. Setting optimal runtime parameters ---------------------
        nextStep("Setting optimal runtime parameters");

        // Update number of streams
        String nStr = core.GetConfig(device, device + "_THROUGHPUT_STREAMS").asString();
        nstreams = Integer.parseInt(nStr);

        // Number of requests
        if (nireq == 0) {
            if (!isAsync) {
                nireq = 1;
            } else {
                String key = "OPTIMAL_NUMBER_OF_INFER_REQUESTS";
                nireq = executableNetwork.GetMetric(key).asInt();
            }
        }

        if ((niter > 0) && isAsync) {
            int temp = niter;
            niter = ((niter + nireq - 1) / nireq) * nireq;
            if (temp != niter) {
                System.err.println(
                        "[ INFO ] Number of iterations was aligned by request number from "
                                + " to "
                                + niter
                                + " using number of requests "
                                + nireq);
            }
        }

        // Time limit
        int durationSeconds = 0;
        if (timeLimit != 0) {
            // time limit
            durationSeconds = timeLimit;
        } else if (niter == 0) {
            // default time limit
            durationSeconds = deviceDefaultDeviceDurationInSeconds(device);
        }
        durationMs = getDurationInMilliseconds(durationSeconds);

        // ---------- 9. Creating infer requests and filling input blobs ---------------
        nextStep("Creating infer requests and filling input blobs");

        InferRequestsQueue inferRequestsQueue = new InferRequestsQueue(executableNetwork, nireq);
        fillBlobs(inferRequestsQueue.requests, inputsInfo);

        // ---------- 10. Measuring performance ----------------------------------------
        String ss = "Start inference " + api + "ronously";
        if (isAsync) {
            if (!ss.isEmpty()) {
                ss += ", ";
            }
            ss = ss + nireq + " inference requests using " + nstreams + " streams for " + device;
        }
        ss += ", limits: ";
        if (durationSeconds > 0) {
            ss += durationMs + " ms duration";
        }
        if (niter != 0) {
            if (durationSeconds > 0) {
                ss += ", ";
            }
            ss = ss + niter + " iterations";
        }

        nextStep("Measuring performance (" + ss + ")");

        int iteration = 0;
        InferReqWrap inferRequest = null;

        inferRequest = inferRequestsQueue.getIdleRequest();
        if (inferRequest == null) {
            System.out.println("No idle Infer Requests!");
            return;
        }

        if (isAsync) {
            inferRequest.startAsync();
        } else {
            inferRequest.infer();
        }

        inferRequestsQueue.waitAll();
        inferRequestsQueue.resetTimes();

        startTime = System.currentTimeMillis();
        long execTime = getTotalMsTime(startTime);

        while ((niter != 0 && iteration < niter)
                || (durationMs != 0L && execTime < durationMs)
                || (isAsync && iteration % nireq != 0)) {
            inferRequest = inferRequestsQueue.getIdleRequest();

            if (isAsync) {
                // As the inference request is currently idle, the wait() adds no additional
                // overhead (and should return immediately).
                // The primary reason for calling the method is exception checking/re-throwing.
                // Callback, that governs the actual execution can handle errors as well,
                // but as it uses just error codes it has no details like ‘what()’ method of
                // `std::exception`.
                // So, rechecking for any exceptions here.
                inferRequest._wait();
                inferRequest.startAsync();

            } else {
                inferRequest.infer();
            }

            iteration++;
            execTime = getTotalMsTime(startTime);
        }

        inferRequestsQueue.waitAll();

        double latency = getMedianValue(inferRequestsQueue.getLatencies());
        double totalDuration = inferRequestsQueue.getDurationInMilliseconds();
        double fps =
                (!isAsync)
                        ? batchSize * 1000.0 / latency
                        : batchSize * 1000.0 * iteration / totalDuration;

        // ------------ 11. Dumping statistics report ----------------------------------
        nextStep("Dumping statistics report");

        System.out.println("Count:      " + iteration + " iterations");
        System.out.println("Duration:   " + String.format("%.2f", totalDuration) + " ms");
        System.out.println("Latency:    " + String.format("%.2f", latency) + " ms");
        System.out.println("Throughput: " + String.format("%.2f", fps) + " FPS");
    }
}
