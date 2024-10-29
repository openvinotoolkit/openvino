const { downloadFile } = require('./helpers.js');

const host = 'https://storage.openvinotoolkit.org';

const models = [
  // hello classification
  '/repositories/openvino_notebooks/models/mobelinet-v3-tf/FP32/v3-small_224_1.0_float.xml',
  '/repositories/openvino_notebooks/models/mobelinet-v3-tf/FP32/v3-small_224_1.0_float.bin',

  // hello reshape ssd
  '/repositories/open_model_zoo/2022.3/models_bin/1/road-segmentation-adas-0001/FP32/road-segmentation-adas-0001.xml',
  '/repositories/open_model_zoo/2022.3/models_bin/1/road-segmentation-adas-0001/FP32/road-segmentation-adas-0001.bin',

  // hello detection, optical character recognition
  '/repositories/open_model_zoo/2022.3/models_bin/1/horizontal-text-detection-0001/FP32/horizontal-text-detection-0001.xml',
  '/repositories/open_model_zoo/2022.3/models_bin/1/horizontal-text-detection-0001/FP32/horizontal-text-detection-0001.bin',

  '/repositories/open_model_zoo/public/text-recognition-resnet-fc/text-recognition-resnet-fc.xml',
  '/repositories/open_model_zoo/public/text-recognition-resnet-fc/text-recognition-resnet-fc.bin',

  // vision background removal
  '/repositories/open_model_zoo/public/vision-background-removal/unet_ir_model.xml',
  '/repositories/open_model_zoo/public/vision-background-removal/unet_ir_model.bin',

  // pose estimation
  '/repositories/open_model_zoo/2022.1/models_bin/3/human-pose-estimation-0001/FP16-INT8/human-pose-estimation-0001.xml',
  '/repositories/open_model_zoo/2022.1/models_bin/3/human-pose-estimation-0001/FP16-INT8/human-pose-estimation-0001.bin',
];
const modelsDir = __dirname + '/../assets/models';


const images = [
  // hello classification
  '/repositories/openvino_notebooks/data/data/image/coco.jpg',

  // hello reshape ssd
  '/repositories/openvino_notebooks/data/data/image/empty_road_mapillary.jpg',

  // hello detection, optical character recognition, pose estimation
  '/repositories/openvino_notebooks/data/data/image/intel_rnb.jpg',

  // vision background removal
  '/repositories/openvino_notebooks/data/data/image/coco_hollywood.jpg',
  '/repositories/openvino_notebooks/data/data/image/wall.jpg',
];
const imagesDir = __dirname + '/../assets/images';

const datasets = [
  // hello classification
  '/repositories/openvino_notebooks/data/data/datasets/imagenet/imagenet_class_index.json',
];
const datasetsDir = __dirname + '/../assets/datasets';

try {
  main();
} catch(error) {
  console.error('Error Occurred', error);
}

async function main() {
  await downloadAssets(models, modelsDir);
  await downloadAssets(images, imagesDir);
  await downloadAssets(datasets, datasetsDir);
}

async function downloadAssets(links, destinationDir) {
  for (const link of links) {
    const url = host + link;
    const filename = link.split('/').pop();

    await downloadFile(url, filename, destinationDir);
    console.log(`Downloaded: ${filename} \n`);
  }
}
