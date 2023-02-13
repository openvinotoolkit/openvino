import Shape from './shape.mjs';
import { jsTypeByPrecisionMap, heapLabelByTypeMap } from './types.mjs';

export default class Tensor {
  #precision;
  #data;
  #shape;

  constructor(precision, data, shapeData) {
    this.#precision = precision;
    this.#data = new jsTypeByPrecisionMap[this.#precision](data);
    
    if (shapeData instanceof Shape) this.#shape = shapeData;
    else this.#shape = new Shape(...shapeData);
  }

  get precision() {
    return this.#precision;
  }

  get data() {
    return this.#data;
  }

  get shape() {
    return this.#shape;
  }

  static parse(ov, originalTensor) {
    // FIXME:
    // const precison = originalTensor.getPrecision();
    const precison = 'float32';
    const shape = Shape.parse(ov, originalTensor.getShape());
    const dataType = jsTypeByPrecisionMap[precison];
    const heapTypeLabel = heapLabelByTypeMap[dataType.name];
    const originalDataPointer = originalTensor.getData();

    const elementsCount = shape.data.reduce((acc, val) => acc*val);
    const data = new dataType(elementsCount);

    for (let i = 0; i < elementsCount; i++) {
      const element = ov[heapTypeLabel][originalDataPointer/dataType.BYTES_PER_ELEMENT + i];
      
      data[i] = element;
    }

    return new Tensor(precison, data, shape);
  }
};

// const inputData = (new Array(224*224*3)).fill(1);

// const tensor = new Tensor('float16', inputData, [1, 224, 224, 3]);

// tensor.shape;
// tensor.data;
// tensor.precision;

// tensor.convert(); // => { obj, free() }
// Tensor.parse(ov, originalTensor);

