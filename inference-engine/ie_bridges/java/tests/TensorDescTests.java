import org.intel.openvino.*;

public class TensorDescTests extends IETest {
    int[] dimsArr = {1, 3, 200, 200};
    TensorDesc tDesc;

    @Override
    protected void setUp() {
        tDesc = new TensorDesc(Precision.U8, dimsArr, Layout.NHWC);
    }

    public void testSetPrecision() {
        tDesc.setPrecision(Precision.U16);
        
        assertEquals("setPrecision", Precision.U16, tDesc.getPrecision());
    }

    public void testSetDims(){
        int[] dims = {0, 1, 22, 333};
        tDesc.setDims(dims);

        int[] res = tDesc.getDims();

        assertEquals("setDims array size", dims.length, res.length);
        for (int i = 0; i < dims.length; i++){
            assertEquals("setDims", dims[i], res[i]);
        }
    }
}
