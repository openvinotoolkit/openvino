import static org.junit.Assert.*;

import org.intel.openvino.*;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Map;

public class InputInfoTests extends IETest {
    IECore core = new IECore();

    @Test
    public void testSetLayout() {
        CNNNetwork net = core.ReadNetwork(modelXml);
        Map<String, InputInfo> inputsInfo = net.getInputsInfo();

        String inputName = new ArrayList<String>(inputsInfo.keySet()).get(0);
        InputInfo inputInfo = inputsInfo.get(inputName);
        assertTrue(inputInfo.getLayout() != Layout.NHWC);

        inputInfo.setLayout(Layout.NHWC);
        assertEquals("setLayout", Layout.NHWC, inputInfo.getLayout());
    }

    @Test
    public void testSetPrecision() {
        CNNNetwork net = core.ReadNetwork(modelXml);
        Map<String, InputInfo> inputsInfo = net.getInputsInfo();

        String inputName = new ArrayList<String>(inputsInfo.keySet()).get(0);
        InputInfo inputInfo = inputsInfo.get(inputName);
        inputInfo.setPrecision(Precision.U8);

        assertEquals("setPrecision", Precision.U8, inputInfo.getPrecision());
    }
}
