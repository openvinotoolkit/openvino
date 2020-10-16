import org.intel.openvino.*;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Map;

/*
This is face detection java sample.

Upon the start-up the sample application reads command line parameters and loads a network
and an image to the Inference Engine device. When inference is done, the application will show
the image with detected objects enclosed in rectangles in new window.It also outputs the
confidence value and the coordinates of the rectangle to the standard output stream.

To get the list of command line parameters run the application with `--help` paramether.
*/
public class Main {
    public static void main(String[] args) {
        final double THRESHOLD = 0.7;
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

        ArgumentParser parser = new ArgumentParser("This is face detection sample");
        parser.addArgument("-i", "path to image");
        parser.addArgument("-m", "path to model .xml");
        parser.parseArgs(args);

        String imgPath = parser.get("-i", null);
        String xmlPath = parser.get("-m", null);

        if (imgPath == null) {
            System.out.println("Error: Missed argument: -i");
            return;
        }
        if (xmlPath == null) {
            System.out.println("Error: Missed argument: -m");
            return;
        }

        Mat image = Imgcodecs.imread(imgPath);

        int[] dimsArr = {1, image.channels(), image.height(), image.width()};
        TensorDesc tDesc = new TensorDesc(Precision.U8, dimsArr, Layout.NHWC);

        // The source image is also used at the end of the program to display the detection results,
        // therefore the Mat object won't be destroyed by Garbage Collector while the network is
        // running.
        Blob imgBlob = new Blob(tDesc, image.dataAddr());

        IECore core = new IECore();

        CNNNetwork net = core.ReadNetwork(xmlPath);

        Map<String, InputInfo> inputsInfo = net.getInputsInfo();
        String inputName = new ArrayList<String>(inputsInfo.keySet()).get(0);
        InputInfo inputInfo = inputsInfo.get(inputName);

        inputInfo.getPreProcess().setResizeAlgorithm(ResizeAlgorithm.RESIZE_BILINEAR);
        inputInfo.setLayout(Layout.NHWC);
        inputInfo.setPrecision(Precision.U8);

        String outputName = new ArrayList<String>(net.getOutputsInfo().keySet()).get(0);

        ExecutableNetwork executableNetwork = core.LoadNetwork(net, "CPU");
        InferRequest inferRequest = executableNetwork.CreateInferRequest();

        inferRequest.SetBlob(inputName, imgBlob);
        inferRequest.Infer();

        Blob output = inferRequest.GetBlob(outputName);
        int dims[] = output.getTensorDesc().getDims();
        int maxProposalCount = dims[2];

        float detection[] = new float[output.size()];
        output.rmap().get(detection);

        for (int curProposal = 0; curProposal < maxProposalCount; curProposal++) {
            int image_id = (int) detection[curProposal * 7];
            if (image_id < 0) break;

            float confidence = detection[curProposal * 7 + 2];

            // Drawing only objects with >70% probability
            if (confidence < THRESHOLD) continue;

            int label = (int) (detection[curProposal * 7 + 1]);
            int xmin = (int) (detection[curProposal * 7 + 3] * image.cols());
            int ymin = (int) (detection[curProposal * 7 + 4] * image.rows());
            int xmax = (int) (detection[curProposal * 7 + 5] * image.cols());
            int ymax = (int) (detection[curProposal * 7 + 6] * image.rows());

            String result = "[" + curProposal + "," + label + "] element, prob = " + confidence;
            result += "    (" + xmin + "," + ymin + ")-(" + xmax + "," + ymax + ")";

            System.out.println(result);
            System.out.println(" - WILL BE PRINTED!");

            // Draw rectangle around detected object.
            Imgproc.rectangle(
                    image, new Point(xmin, ymin), new Point(xmax, ymax), new Scalar(0, 255, 0));
        }

        HighGui.namedWindow("Detection", HighGui.WINDOW_AUTOSIZE);
        HighGui.imshow("Detection", image);
        HighGui.waitKey(0);
        HighGui.destroyAllWindows();
    }
}
