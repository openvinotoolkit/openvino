import static org.junit.Assert.*;
import org.junit.Test;
import org.junit.Before;

import java.util.Map;
import java.util.Vector;
import java.util.ArrayList;

import org.intel.openvino.*;
import org.intel.openvino.InferenceEngineProfileInfo.LayerStatus;

public class InferRequestTests extends IETest {
    IECore core;
    CNNNetwork net;
    ExecutableNetwork executableNetwork;
    InferRequest inferRequest;
    boolean completionCallback;

    @Before
    public void setUp() {
        core = new IECore();
        net = core.ReadNetwork(modelXml);
        executableNetwork = core.LoadNetwork(net, device);
        inferRequest = executableNetwork.CreateInferRequest();
        completionCallback = false;
    }

    @Test
    public void testGetPerformanceCounts() {
        inferRequest.Infer();

        Vector<String> layer_name = new Vector<>();
        layer_name.add("19/Fused_Add_");
        layer_name.add("21");
        layer_name.add("22");
        layer_name.add("23");
        layer_name.add("24/Fused_Add_");
        layer_name.add("26");
        layer_name.add("27");
        layer_name.add("29");
        layer_name.add("fc_out");
        layer_name.add("out_fc_out");

        Vector<String> exec_type = new Vector<>();
        exec_type.add("Convolution");
        exec_type.add("ReLU");
        exec_type.add("Pooling");
        exec_type.add("Convolution");
        exec_type.add("Convolution");
        exec_type.add("ReLU");
        exec_type.add("Pooling");
        exec_type.add("FullyConnected");
        exec_type.add("SoftMax");
        exec_type.add("Output");

        Map<String, InferenceEngineProfileInfo> res = inferRequest.GetPerformanceCounts();

        assertEquals("Map size", layer_name.size(), res.size());
        ArrayList<String> resKeySet = new ArrayList<String>(res.keySet());

        for (int i = 0; i < res.size(); i++) {
            String key  = resKeySet.get(i);
            InferenceEngineProfileInfo resVal = res.get(key);

            assertEquals(key + " execType", key, layer_name.elementAt(i));
            assertEquals(key + " executionIndex", i, resVal.executionIndex);
            assertTrue(resVal.status == InferenceEngineProfileInfo.LayerStatus.EXECUTED
                        || resVal.status == InferenceEngineProfileInfo.LayerStatus.NOT_RUN);
        }
    }

    @Test
    public void testStartAsync() {
        inferRequest.StartAsync();
        StatusCode statusCode = inferRequest.Wait(WaitMode.RESULT_READY);

        assertEquals("StartAsync", StatusCode.OK, statusCode);
    }

    @Test
    public void testSetCompletionCallback() {
        inferRequest.SetCompletionCallback(new Runnable() {

            @Override
            public void run() {
                completionCallback = true;
            }
        });

        for(int i = 0; i < 5; i++) {
            inferRequest.Wait(WaitMode.RESULT_READY); 
            inferRequest.StartAsync();
        } 
        
        inferRequest.Wait(WaitMode.RESULT_READY); 
        inferRequest.StartAsync();
        StatusCode statusCode = inferRequest.Wait(WaitMode.RESULT_READY);

        assertEquals("SetCompletionCallback", true, completionCallback);
    }
}
