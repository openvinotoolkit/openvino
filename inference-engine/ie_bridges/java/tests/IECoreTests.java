import static org.junit.Assert.*;
import org.junit.Test;

import org.intel.openvino.*;

import java.util.Map;
import java.util.HashMap;

public class IECoreTests extends IETest {
    IECore core = new IECore();
    
    @Test
    public void testReadNetwork() {
        CNNNetwork net = core.ReadNetwork(modelXml, modelBin);
        assertEquals("Network name", "test_model", net.getName());
    }

    @Test
    public void testReadNetworkXmlOnly() {
        CNNNetwork net = core.ReadNetwork(modelXml);
        assertEquals("Batch size", 1, net.getBatchSize());
    }

    @Test
    public void testReadNetworkIncorrectXmlPath() {
        String exceptionMessage = "";
        try {
            CNNNetwork net = core.ReadNetwork("model.xml", modelBin);
        } catch (Exception e) {
            exceptionMessage = e.getMessage();
        }
        assertTrue(exceptionMessage.contains("Model file model.xml cannot be opened!"));
    }

    @Test
    public void testReadNetworkIncorrectBinPath() {
        String exceptionMessage = "";
        try {
            CNNNetwork net = core.ReadNetwork(modelXml, "model.bin");
        } catch (Exception e) {
            exceptionMessage = e.getMessage();
        }
        assertTrue(exceptionMessage.contains("Weights file model.bin cannot be opened!"));
    }

    @Test
    public void testLoadNetwork() {
        CNNNetwork net = core.ReadNetwork(modelXml, modelBin);
        ExecutableNetwork executableNetwork = core.LoadNetwork(net, device);

        assertTrue(executableNetwork instanceof ExecutableNetwork);
    }

    @Test
    public void testLoadNetworDeviceConfig() {
        CNNNetwork net = core.ReadNetwork(modelXml, modelBin);

        Map<String, String> testMap = new HashMap<String, String>();

        //When specifying key values as raw strings, omit the KEY_ prefix
        testMap.put("CPU_BIND_THREAD", "YES");
        testMap.put("CPU_THREADS_NUM", "1");

        ExecutableNetwork executableNetwork = core.LoadNetwork(net, device, testMap);

        assertTrue(executableNetwork instanceof ExecutableNetwork);
    }

    @Test
    public void testLoadNetworkWrongDevice() {
        String exceptionMessage = "";
        CNNNetwork net = core.ReadNetwork(modelXml, modelBin);
        try {
            core.LoadNetwork(net, "DEVISE");
        } catch (Exception e) {
            exceptionMessage = e.getMessage();
        }
        assertTrue(exceptionMessage.contains("Device with \"DEVISE\" name is not registered in the InferenceEngine"));
    }
}
