module.exports = {
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    './.eslintrc-global.js',
  ],
  ignorePatterns: ['node_modules/', 'types/', 'dist/', 'bin/', '.eslintrc.js'],
  root: true,
};
