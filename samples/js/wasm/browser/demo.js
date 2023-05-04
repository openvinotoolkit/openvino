const MODELS_PATH = './assets/models/';

const statusElement = document.getElementById('status');

const events = {
  onLibInitializing: setStatus('OpenVINO initializing...'),
  onModelLoaging: setStatus('OpenVINO successfully initialized. Model loading...'),
  onInferenceRunning: setStatus('Inference is in the progress, please wait...'),
  onFinish: outputTensor => {
    console.log(outputTensor);

    setStatus('Open browser\'s console to see result')();
  },
};

const inferenceParametersMobilenetV3 = {
  modelPath: `${MODELS_PATH}v3-small_224_1.0_float.xml`,
  imgPath: './assets/images/coco224x224.jpg',
  shape: [1, 224, 224, 3],
  layout: 'NHWC',
};

const inferenceParametersFaceDetection = {
  modelPath: `${MODELS_PATH}face-detection-0200/face-detection-0200.xml`,
  imgPath: './assets/images/peopleAndCake256x256.jpg',
  shape: [1, 3, 256, 256],
  layout: 'NCHW',
};

makeInference(openvino, inferenceParametersMobilenetV3, events);

function setStatus(txt) {
  return () => statusElement.innerText = txt;
}
