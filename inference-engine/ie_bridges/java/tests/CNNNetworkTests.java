import static org.junit.Assert.*;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import org.intel.openvino.*;

public class CNNNetworkTests extends IETest {
    IECore core = new IECore();

    @Test
    public void testInputName() {
        CNNNetwork net = core.ReadNetwork(modelXml);
        Map<String, InputInfo> inputsInfo = net.getInputsInfo();
        String inputName = new ArrayList<String>(inputsInfo.keySet()).get(0);

        assertEquals("Input name", "data", inputName);
    }

    @Test
    public void testReshape() {
        CNNNetwork net = core.ReadNetwork(modelXml);

        Map<String, int[]> input = new HashMap<>();
        int[] val = {1, 3, 34, 34};
        input.put("data", val);

        net.reshape(input);
        Map<String, int[]> res = net.getInputShapes();

        assertArrayEquals(input.get("data"), res.get("data"));
    }

    @Test
    public void testAddOutput() {
        CNNNetwork net = core.ReadNetwork(modelXml);
        Map<String, Data> output = net.getOutputsInfo();
        
        assertEquals("Input size", 1, output.size());
        
        net.addOutput("19/WithoutBiases");
        output = net.getOutputsInfo();

        assertEquals("Input size", 2, output.size());
    }

}
