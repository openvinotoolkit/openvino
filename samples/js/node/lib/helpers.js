module.exports = { argMax, prepareResnetTensor, nhwc2ncwh };

function argMax(arr) {
  if (arr.length === 0) return -1;

  let max = arr[0];
  let maxIndex = 0;

  for (let i = 1; i < arr.length; i++)
    if (arr[i] > max) {
      maxIndex = i;
      max = arr[i];
    }

  return maxIndex;
}

function prepareResnetTensor(arr) {
  /*
        The images have to be normalized using
        mean_values: [123.675,116.28,103.53]
        and scale_values: [58.395,57.12,57.375]
        of the channels
        For more information refer to:
        https://docs.openvino.ai/latest/ovms_demo_using_onnx_model.html
    */
  if (arr.length === 0 && arr.length % 3 === 0) return -1;

  for (let i = 0; i< arr.length; i+=3) {
    arr[i] = (arr[i] - 103.53) / 57.375;
    arr[i+1] = (arr[i] - 116.28) / 57.12;
    arr[i+2] = (arr[i] - 123.675) / 58.395;
  }
}

function nhwc2ncwh(view) {
  const newArray = [];
  for (let j=0; j<3; j++)
    for (let i=j; i<view.length; i+=3)
      newArray.push(view[i]);

  view.set(newArray);
}
