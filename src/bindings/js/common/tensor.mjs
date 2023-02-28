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
    const precision = 'float32';
    const shape = Shape.parse(ov, originalTensor.getShape());
    const dataType = jsTypeByPrecisionMap[precision];
    const heapTypeLabel = heapLabelByTypeMap[dataType.name];
    const originalDataPointer = originalTensor.getData();

    const elementsCount = shape.data.reduce((acc, val) => acc*val);
    const data = new dataType(elementsCount);

    for (let i = 0; i < elementsCount; i++) {
      const element = ov[heapTypeLabel][originalDataPointer/dataType.BYTES_PER_ELEMENT + i];
      
      data[i] = element;
    }

    return new Tensor(precision, data, shape);
  }

  static convert(ov, tensor) {
    const { precision } = tensor;
    const dataType = jsTypeByPrecisionMap[precision];
    const originalShape = tensor.shape.convert(ov);

    const originalData = new dataType(tensor.data);
    const elementSizeInBytes = originalData.BYTES_PER_ELEMENT;
    const heapSpace = ov._malloc(originalData.length*elementSizeInBytes);
    const offset = Math.sqrt(elementSizeInBytes);
    const waPrecision = heapLabelByTypeMap[dataType.name];

    ov[waPrecision].set(originalData, heapSpace>>offset); 

    return { 
      obj: new ov.Tensor(precision, heapSpace, originalShape.obj),
      free: () => {
        originalShape.free();
        ov._free(heapSpace);
      }
     };
  }

  convert(ov) {
    return Tensor.convert(ov, this);
  }
};

// const inputData = (new Array(224*224*3)).fill(1);

// const tensor = new Tensor('float16', inputData, [1, 224, 224, 3]);

// tensor.shape;
// tensor.data;
// tensor.precision;

// tensor.convert(); // => { obj, free() }
// Tensor.parse(ov, originalTensor);

