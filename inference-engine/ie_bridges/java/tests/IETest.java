import junit.framework.TestCase;

import java.nio.file.Paths;
import java.lang.Class;
import java.util.List;

import org.intel.openvino.*;

public class IETest extends TestCase {
    String modelXml;
    String modelBin;
    String device;

    public IETest(){
        try {
            System.loadLibrary(IECore.NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Failed to load Inference Engine library\n" + e);
            System.exit(1);
        }

        modelXml = Paths.get(System.getenv("MODELS_PATH"), "models", "test_model", "test_model_fp32.xml").toString();
        modelBin = Paths.get(System.getenv("MODELS_PATH"), "models", "test_model", "test_model_fp32.bin").toString();
        device = "CPU";
    }
}
