package org.intel.openvino;

public class BlockingDesc extends IEWrapper {

    public BlockingDesc(){
        super(GetBlockingDesc());
    }

    public BlockingDesc(int[] dims, Layout layout){
        super(GetBlockingDesc1(dims, layout.getValue()));
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native long GetBlockingDesc();

    private static native long GetBlockingDesc1(int[] dims, int layout);
}
