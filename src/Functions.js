const _ = require('lodash');

module.exports = {
  sigmoid: (x) => (1 / (1 + Math.exp(-x))),
  derivative: (x) => (x * (1 - x)),
  normalizeByConstant: (arr, c) => arr.map((x) => x / c),
  normalizeByMax: (arr) => {
    const max = _.max(arr);
    return arr.map((x) => x / max);
  },
};
