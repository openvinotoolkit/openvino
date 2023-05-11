module.exports = {
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    './.eslintrc-common.js',
  ],
  ignorePatterns: ['**/*.js', 'node_modules/', 'types/', 'dist/', 'bin/'],
  root: true,
};
