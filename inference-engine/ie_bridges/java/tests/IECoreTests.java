import org.intel.openvino.*;

import java.util.Map;
import java.util.HashMap;

public class IECoreTests extends IETest {
    IECore core;
    String exceptionMessage;
    
    @Override
    protected void setUp() {
        core = new IECore();
        exceptionMessage = "";
    }

    public void testInitIECore(){
        assertTrue(core instanceof IECore);
    }

    public void testReadNetwork(){
        CNNNetwork net = core.ReadNetwork(modelXml, modelBin);
        assertEquals("Network name", "test_model", net.getName());
    }

    public void testReadNetworkXmlOnly(){
        CNNNetwork net = core.ReadNetwork(modelXml);
        assertEquals("Batch size", 1, net.getBatchSize());
    }

    public void testReadNetworkIncorrectXmlPath(){
        try{
            CNNNetwork net = core.ReadNetwork("model.xml", modelBin);
        } catch (Exception e){
            exceptionMessage = e.getMessage();
        }
        assertTrue(exceptionMessage.contains("Model file model.xml cannot be opened!"));
    }

    public void testReadNetworkIncorrectBinPath(){
        try{
            CNNNetwork net = core.ReadNetwork(modelXml, "model.bin");
        } catch (Exception e){
            exceptionMessage = e.getMessage();
        }
        assertTrue(exceptionMessage.contains("Weights file model.bin cannot be opened!"));
    }

    public void testLoadNetwork(){
        CNNNetwork net = core.ReadNetwork(modelXml, modelBin);
        ExecutableNetwork executableNetwork = core.LoadNetwork(net, device);

        assertTrue(executableNetwork instanceof ExecutableNetwork);
    }

    public void testLoadNetworDeviceConfig(){
        CNNNetwork net = core.ReadNetwork(modelXml, modelBin);

        Map<String, String> testMap = new HashMap<String, String>();

        //When specifying key values as raw strings, omit the KEY_ prefix
        testMap.put("CPU_BIND_THREAD", "YES");
        testMap.put("CPU_THREADS_NUM", "1");

        ExecutableNetwork executableNetwork = core.LoadNetwork(net, device, testMap);

        assertTrue(executableNetwork instanceof ExecutableNetwork);
    }

    public void testLoadNetworkWrongDevice(){
        CNNNetwork net = core.ReadNetwork(modelXml, modelBin);
        try{
            core.LoadNetwork(net, "DEVISE");
        } catch (Exception e){
            exceptionMessage = e.getMessage();
        }
        assertTrue(exceptionMessage.contains("Device with \"DEVISE\" name is not registered in the InferenceEngine"));
    }
}
