module.exports = {
  sigmoid: (x) => (1 / (1 + Math.exp(-x))),
  derivative: (x) => (x * (1 - x)),
};
