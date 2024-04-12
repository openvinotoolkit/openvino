const { addon: ov } = require('./dist/index.js');

const device = 'CPU';
const tokenizerModelPath = '/home/nvishnya/Code/openvino_tokenizers/output_dir/openvino_tokenizer.xml';
const detokenizerModelPath = '/home/nvishnya/Code/openvino_tokenizers/output_dir/openvino_detokenizer.xml';
const modelPath = '/home/nvishnya/Code/TinyLlama-1.1B-Chat-v1.0/';

main('Hey, how are you?');

async function main(promt) {
  try {
    const core = new ov.Core();

    const val = 121;

    core.test(new Uint8Array(), val);

    // core.test();

    // core.addExtension('libopenvino_tokenizers.so');

    // const tokenizerModel = core.readModelSync(tokenizerModelPath);
    // const tokenizerCModel = await core.compileModel(tokenizerModel, device);
    // const tokenizer = tokenizerCModel.createInferRequest();

    // const [inputIds, attentionMask] = await tokenize(tokenizer, promt);

    // const detokenizerModel = await core.readModel(detokenizerModelPath);
    // const detokenizerCModel = await core.compileModel(detokenizerModel, device);
    // const detokenizer = tokenizerCModel.createInferRequest();

    // const lmModel = await core.readModel(modelPath);

  } catch(e) {
    console.log(e);
  }
}

async function tokenize(tokenizer, promt) {
  const batchSize = 1;
  const tensor = new ov.Tensor(ov.element.string, [batchSize], promt);

  console.log(tensor.getData())

  tokenizer.setInputTensor(tensor);
  await tokenizer.infer();

  return [];

  // return [
  //   tokenizer.getTensor('input_ids'),
  //   tokenizer.getTensor('attention_mask')
  // ];
}

async function detokenize(detokenizer, tokens) {
  const batchSize = 1;
  const tensor = new ov.Tensor(ov.element.i64, [batchSize, tokens.getSize()], tokens.getData());

  detokenizer.setInputTensor(tensor);
  await detokenizer.infer();

  return detokenizer.getOutputTensor().getData()[0];
}
