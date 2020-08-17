import org.opencv.core.*;
import org.opencv.imgcodecs.*;
import org.opencv.videoio.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;

import java.util.LinkedList;
import java.util.Vector;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.Map;
import java.util.Queue;
import java.util.ArrayList;
import java.util.HashMap;

import org.intel.openvino.*;

/*
This is async face detection java sample.

Upon the start-up the sample application reads command line parameters and loads a network 
and an images to the Inference Engine device. When inference is done, the application 
shows the video with detected objects enclosed in rectangles in new window.

To get the list of command line parameters run the application with `--help` paramether.
*/
public class Main {
 
    public static Blob imageToBlob(Mat image) {
        if (buff == null) 
            buff = new byte[(int) (image.total() * image.channels())];

        image.get(0, 0, buff);

        int[] dimsArr = {1, image.channels(), image.height(), image.width()};
        TensorDesc tDesc = new TensorDesc(Precision.U8, dimsArr, Layout.NHWC);

        return new Blob(tDesc, buff);
    }

    static void processInferRequets(WaitMode wait) {
        int size = 0;
        float[] res = null;

        while (!startedRequestsIds.isEmpty()) {
            int requestId = startedRequestsIds.peek();
            InferRequest inferRequest = inferRequests.get(requestId);
 
            if (inferRequest.Wait(wait) != StatusCode.OK)
                return;

            if (size == 0 && res == null) {
                size = inferRequest.GetBlob(outputName).size();
                res = new float[size];
            }

            inferRequest.GetBlob(outputName).rmap().get(res);
            detectionOutput.add(res);

            resultCounter++;

            asyncInferIsFree.setElementAt(true, requestId);
            startedRequestsIds.remove();
        }
    }

    public static void main(String[] args) {
        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Failed to load OpenCV library\n" + e);
            System.exit(1);
        }
        try {
            System.loadLibrary(IECore.NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Failed to load Inference Engine library\n" + e);
            System.exit(1);
        }

        ArgumentParser parser = new ArgumentParser("This is async face detection sample");
        parser.addArgument("-i", "path to video");
        parser.addArgument("-m", "path to model .xml");
        parser.addArgument("-d", "device");
        parser.addArgument("-nireq", "number of infer requests");
        parser.parseArgs(args);

        String imgsPath = parser.get("-i", null);
        String xmlPath = parser.get("-m", null);
        String device = parser.get("-d", "CPU");
        int inferRequestsSize = parser.getInteger("-nireq", 2);

        if(imgsPath == null ) {
            System.out.println("Error: Missed argument: -i");
            return;
        }
        if(xmlPath == null) {
            System.out.println("Error: Missed argument: -m");
            return;
        }

        int warmupNum = inferRequestsSize * 2;

        BlockingQueue<Mat> framesQueue = new LinkedBlockingQueue<Mat>();

        Thread captureThread = new Thread(new Runnable() {
            @Override
            public void run() {
                VideoCapture cam = new VideoCapture();
                cam.open(imgsPath);
                Mat frame = new Mat();

                while (cam.read(frame)) {
                        framesCounter++;
                        framesQueue.add(frame.clone());           
                }
            }
        });

        Thread inferThread = new Thread(new Runnable() {
        
            @Override
            public void run() {
                try {
                    IECore core = new IECore();
                    CNNNetwork net = core.ReadNetwork(xmlPath);
    
                    Map<String, InputInfo> inputsInfo = net.getInputsInfo();
                    String inputName = new ArrayList<String>(inputsInfo.keySet()).get(0);
                    InputInfo inputInfo = inputsInfo.get(inputName);
    
                    inputInfo.getPreProcess().setResizeAlgorithm(ResizeAlgorithm.RESIZE_BILINEAR);
                    inputInfo.setLayout(Layout.NHWC);
                    inputInfo.setPrecision(Precision.U8);

                    outputName = new ArrayList<String>(net.getOutputsInfo().keySet()).get(0);
    
                    ExecutableNetwork executableNetwork = core.LoadNetwork(net, device);
                    
                    asyncInferIsFree = new Vector<Boolean>(inferRequestsSize);

                    for (int i = 0; i < inferRequestsSize; i++) {
                        inferRequests.add(executableNetwork.CreateInferRequest());
                        asyncInferIsFree.add(true);
                    }
    
                    boolean isRunning = true;

                    while (captureThread.isAlive() || !framesQueue.isEmpty()) {
                        processInferRequets(WaitMode.STATUS_ONLY);

                        for (int i = 0; i < inferRequestsSize; i++) {
                            if (!asyncInferIsFree.get(i))
                                continue;

                            Mat frame = framesQueue.poll(0, TimeUnit.SECONDS);

                            if (frame == null)
                                break;

                            InferRequest request = inferRequests.get(i);
        
                            asyncInferIsFree.setElementAt(false, i);
                            processedFramesQueue.add(frame);  // predictionsQueue is used in rendering

                            Blob imgBlob = imageToBlob(frame);
                            request.SetBlob(inputName, imgBlob); 

                            startedRequestsIds.add(i);
                            request.StartAsync();
                        }
                    }
                    processInferRequets(WaitMode.RESULT_READY);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                    
                    for (Thread t : Thread.getAllStackTraces().keySet())
                        if (t.getState()==Thread.State.RUNNABLE) 
                            t.interrupt(); 
                }
            }
        });

        captureThread.start();
        inferThread.start();

        TickMeter tm = new TickMeter();       
        try {
            while (inferThread.isAlive() || !detectionOutput.isEmpty()) {

                float[] detection = detectionOutput.poll(waitingTime, TimeUnit.SECONDS);  
                if (detection == null)
                  continue;
                                
                Mat img = processedFramesQueue.poll(waitingTime, TimeUnit.SECONDS);   
                int maxProposalCount = detection.length / 7;

                for (int curProposal = 0; curProposal < maxProposalCount; curProposal++) {
                    int imageId = (int) detection[curProposal * 7];
                    if (imageId < 0)
                        break;
        
                    float confidence = detection[curProposal * 7 + 2];

                    // Drawing only objects with >70% probability
                    if (confidence < CONFIDENCE_THRESHOLD)
                        continue;
                    
                    int label = (int) (detection[curProposal * 7 + 1]);
                    int xmin = (int) (detection[curProposal * 7 + 3] * img.cols());
                    int ymin = (int) (detection[curProposal * 7 + 4] * img.rows());
                    int xmax = (int) (detection[curProposal * 7 + 5] * img.cols());
                    int ymax = (int) (detection[curProposal * 7 + 6] * img.rows());
                    
                    // Draw rectangle around detected object.
                    Imgproc.rectangle(img, new Point(xmin, ymin), new Point(xmax, ymax), new Scalar(0, 255, 0), 2); 
                }

                if (resultCounter == warmupNum) {    
                    tm.start();
                } else if (resultCounter > warmupNum) {
                    tm.stop();
                    double worksFps = ((double)(resultCounter - warmupNum)) / tm.getTimeSec();
                    double readFps = ((double)(framesCounter - warmupNum)) / tm.getTimeSec();        
                    tm.start();

                    Imgproc.putText(img, "Reading fps: " + String.format("%.3f", readFps), new Point(10, 50), 0 , 0.7, new Scalar(0, 255, 0), 1);
                    Imgproc.putText(img, "Inference fps: " + String.format("%.3f", worksFps), new Point(10, 80), 0 , 0.7, new Scalar(0, 255, 0), 1);
                }
                
                HighGui.imshow("Detection", img);
            }
           
            captureThread.join();
            inferThread.join();

            HighGui.waitKey(0);
            HighGui.destroyAllWindows();

        } catch (InterruptedException e) {
            e.printStackTrace();
            for (Thread t : Thread.getAllStackTraces().keySet())
                if (t.getState()==Thread.State.RUNNABLE) 
                    t.interrupt(); 
        }
    }

    static final float CONFIDENCE_THRESHOLD = 0.7f;
    static int waitingTime = 1;

    static BlockingQueue<Mat> processedFramesQueue = new LinkedBlockingQueue<Mat>();
    static BlockingQueue<float[]> detectionOutput = new LinkedBlockingQueue<float[]>();

    static String outputName;
    static Queue<Integer> startedRequestsIds = new LinkedList<Integer>();
    static Vector<InferRequest> inferRequests = new Vector<InferRequest>();
    static Vector<Boolean> asyncInferIsFree;

    static byte[] buff = null;

    static int framesCounter = 0;
    static int resultCounter = 0;
}
