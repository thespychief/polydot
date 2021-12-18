const Matrix = require('./Matrix');

class Layer {
  constructor(prevLayerSize, layerSize) {
    this.weights = Matrix.create(layerSize, prevLayerSize);
    this.bias = Matrix.create(layerSize, 1);
  }

  getWeights() {
    return this.weights;
  }

  getBias() {
    return this.bias;
  }

  updateWeights(weights) {
    this.weights = weights;
  }

  updateBias(bias) {
    this.bias = bias;
  }

  addToWeights(deltaWeight) {
    this.weights = Matrix.add(this.weights, deltaWeight);
  }

  addToBias(bias) {
    this.bias = Matrix.add(this.bias, bias);
  }
}

module.exports = Layer;
