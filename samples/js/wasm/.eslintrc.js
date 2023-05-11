module.exports = {
  parserOptions: {
    ecmaVersion: 'latest'
  },
  env: {
    browser: true,
    node: true,
    es6: true,
  },
  extends: [
    'eslint:recommended',
    '../../../src/bindings/js/.eslintrc-common.js',
  ],
  globals: {
    openvinojs: true,
    makeInference: true,
  },
  ignorePatterns: ['node_modules/', '.eslintrc.js'],
  root: true,
};
