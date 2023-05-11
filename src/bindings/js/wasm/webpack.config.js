const path = require('path');
const CopyPlugin = require('copy-webpack-plugin');

module.exports = getConfigs();

function getConfigs(mode = 'development') {
  const commonConfig = {
    mode,
    entry: {
      'openvino-wasm': './lib/index.ts',
    },
    module: {
      rules: [
        {
          test: /\.ts?$/,
          use: 'ts-loader',
          exclude: /node_modules/,
        },
      ],
    },
    resolve: {
      extensions: ['.ts', '.js'],
      fallback: {
        'crypto': false,
        'fs': false,
        'path': false,
      }
    },
    plugins: [
      new CopyPlugin({
        patterns: [
          { from: './bin/openvino_wasm.wasm', to: 'openvino_wasm.wasm' },
        ],
      }),
    ],
  };

  const webConfig = Object.assign({}, commonConfig, {
    output: {
      filename: getConfigFilename('web'),
      path: path.resolve(__dirname, 'dist'),
      clean: true,
      library: {
        name: 'openvinojs',
        type: 'umd',
        export: 'default',
      },
      libraryTarget: 'window',
    },
  });
  const nodeConfig = Object.assign({ target: 'node' }, commonConfig, {
    output: {
      filename: getConfigFilename('node'),
      path: path.resolve(__dirname, 'dist'),
      library: {
        type: 'commonjs',
        export: 'default',
      },
      publicPath: '',
      globalObject: 'this',
      clean: true,
    },
  });

  return [webConfig, nodeConfig];

  function getConfigFilename(label) {
    return `[name].${label}.bundle.js`;
  }
}
