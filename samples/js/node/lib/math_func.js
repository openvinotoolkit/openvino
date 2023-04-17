module.exports = {
    argMax: function(arr) {
        if (arr.length === 0) {
            return -1;
        }

        var max = arr[0];
        var maxIndex = 0;

        for (var i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                maxIndex = i;
                max = arr[i];
            }
        }

        return maxIndex;
    },

    prepare_resnet_tensor: function(arr) {
        /*
        The images have to normalized using
        mean_values: [123.675,116.28,103.53] and
        scale_values: [58.395,57.12,57.375]
        of the channels
        For more information refer to: 
        https://docs.openvino.ai/latest/ovms_demo_using_onnx_model.html
        */
        // [0.485, 0.456, 0.406]
        // [0.229, 0.224, 0.225]
        if (arr.length === 0 && arr.length % 3 === 0) {
            return -1;
            console.log("fail");
        }

        // for (var i = 0; i< arr.length; i+=3){
        //     arr[i] = (arr[i]/255 - 0.485) / 0.229;
        //     arr[i+1] = (arr[i]/255 - 0.456) / 0.224;
        //     arr[i+2] = (arr[i]/255 - 0.406) / 0.225;
        // }
        for (var i = 0; i< arr.length; i+=3){
            arr[i] = (arr[i] - 103.53) / 57.375;
            arr[i+1] = (arr[i] - 116.28) / 57.12;
            arr[i+2] = (arr[i] - 123.675) / 58.395;
        }
    },

    nhwc2ncwh: function(view){
        const new_arr = new Array();
        for(var j=0; j<3; j++){
            for(var i=j; i<view.length; i+=3){
                new_arr.push(view[i]);
            }
        }
        view.set(new_arr)
    }

}
