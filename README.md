
Polydot
========

Polydot is a pure Javascript neural network library.

## Overview

### Installation

```cmd
npm install polydot --save
```

### Usage

```javascript
const Polydot = require('polydot');
```

### Input Data Format
```javascript
0 <= x <= 1

[
  {
    "input": [x, x, x, x, x, x, ...],
    "output": [x, x, x, x, ...]
  },
  ...
]
```

### Examples

```javascript
// Create a new network
const network = new Polydot.Network({
  structure: [784, 100, 10],
});

// Train the network for one epoch
network.train(trainingData);

// Predict a single point
network.predict(input);

// Evaluate network accuracy
network.evaluate(testData);

// Save the network
const modelToSave = network.save();
fs.writeFileSync('model.json', JSON.stringify(modelToSave));

// Load a network
const modelToLoad = JSON.parse(fs.readFileSync('model.json'));
const networkFromModel = new Polydot.Network(modelToLoad);
```

### Performance

##### MNIST

```
Dataset: MNIST Handwritten Digits
Device: 2014 Macbook Pro, 2.5 GHz Quad-Core Intel Core i7
Training Set Size: 60000, One Epoch
Test Set Size: 10000
---------------------------------------------------------
Training Time: 2:57.259 (m:ss.mmm)
Prediction Time: 0.789 ms
Accuracy: 9441 / 10000
```
