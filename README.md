# GOML - Machine Learning in Go

<p align="center">
  <img src="goml.png" alt="GOML - Simple Machine Learning in Go" width="600">
</p>

GOML is a simple machine learning package in Go that supports training and inference with multiple input/output parameters using generic types. It's designed to be easy to use while providing flexible model serialization and configuration options.

## Features

- Support for multiple input and output parameters
- Support for various data types (numeric, string)
- Serialization of models and weights to JSON for persistence
- Flexible training configuration
- Support for linear and logistic regression models
- Ability to save and restore trained models

## Installation

```bash
go get github.com/audi70r/goml
```

## Usage

### Basic Example

```go
package main

import (
	"fmt"
	
	"github.com/audi70r/goml/pkg/goml"
)

func main() {
	// Create a new ML engine with a linear model
	engine := goml.New()
	model := goml.NewLinearModel()
	engine.WithModel(model.JSON())
	
	// Configure training
	config := &goml.Config{
		LearningRate: 0.01,
		Epochs:       1000,
		BatchSize:    32,
		Regularize:   0.0001,
		Tolerance:    0.00001,
	}
	engine.WithConfig(config)
	
	// Prepare training data
	inputs := []map[string]interface{}{
		{"feature1": 1.0, "feature2": 2.0},
		{"feature1": 2.0, "feature2": 3.0},
		{"feature1": 3.0, "feature2": 4.0},
	}
	
	outputs := []map[string]interface{}{
		{"target1": 5.0, "target2": 10.0},
		{"target1": 7.0, "target2": 14.0},
		{"target1": 9.0, "target2": 18.0},
	}
	
	// Train the model
	err := engine.Train(inputs, outputs)
	if err != nil {
		fmt.Printf("Training error: %v\n", err)
		return
	}
	
	// Make a prediction
	newInput := map[string]interface{}{
		"feature1": 4.0,
		"feature2": 5.0,
	}
	
	prediction, err := engine.Predict(newInput)
	if err != nil {
		fmt.Printf("Prediction error: %v\n", err)
		return
	}
	
	fmt.Println("Prediction:", prediction)
	
	// Save the model and weights
	modelJSON, _ := engine.GetModel()
	weightsJSON, _ := engine.GetWeights()
	
	// Later, restore the model
	newEngine := goml.New()
	newEngine.WithModel(*modelJSON)
	newEngine.WithWeights(*weightsJSON)
}
```

### Logistic Regression Example

```go
// Create a logistic regression model
logisticModel := goml.NewLogisticModel()
logisticEngine := goml.New()
logisticEngine.WithModel(logisticModel.JSON())
logisticEngine.WithConfig(&goml.Config{
	LearningRate: 0.01,
	Epochs:       1000,
})

// Train for binary classification
inputs := []map[string]interface{}{
	{"feature1": 1.0, "feature2": 0.5},
	{"feature1": 0.5, "feature2": 0.2},
	{"feature1": 2.0, "feature2": 1.5},
}
outputs := []map[string]interface{}{
	{"class": 1},
	{"class": 0},
	{"class": 1},
}

logisticEngine.Train(inputs, outputs)

// Get prediction (will be between 0 and 1)
prediction, _ := logisticEngine.Predict(map[string]interface{}{
	"feature1": 1.8,
	"feature2": 1.2,
})
```

## API Reference

### Core Types

- `Engine`: Main entry point for all ML operations
- `Model`: Represents the model structure (linear, logistic)
- `Weights`: Stores the trained parameters
- `Config`: Defines training parameters

### Key Methods

- `New() *Engine`: Create a new engine
- `WithModel(modelJson string) (*Model, error)`: Load a model from JSON
- `WithWeights(weightsJson string) (*Weights, error)`: Load weights from JSON
- `WithConfig(*Config) *Engine`: Set training configuration
- `Train(inputs []map[string]interface{}, outputs []map[string]interface{}) error`: Train the model
- `Predict(input map[string]interface{}) (map[string]interface{}, error)`: Perform inference
- `GetModel() (*string, error)`: Serialize model to JSON
- `GetWeights() (*string, error)`: Serialize weights to JSON

## License

MIT