package goml

import (
	"encoding/json"
	"fmt"
)

// Engine encapsulates the entire ML system: model, weights, config, etc.
type Engine struct {
	model   *Model
	weights *Weights
	config  *Config
}

// New creates a new engine with default configuration
func New() *Engine {
	return &Engine{
		config: DefaultConfig(),
	}
}

// NewAuto creates a new engine with automatic model selection
// It examines the first output sample to determine the appropriate model type
func NewAuto(outputSample map[string]interface{}) *Engine {
	return &Engine{
		model:  NewAutoModel(outputSample),
		config: DefaultConfig(),
	}
}

// TrainAuto creates a new engine and trains it with automatic model selection
// This is a convenience function that handles the entire process
func TrainAuto(inputs []map[string]interface{}, outputs []map[string]interface{}) (*Engine, error) {
	if len(outputs) == 0 {
		return nil, fmt.Errorf("no output data provided")
	}

	// Create engine with auto-detected model type
	engine := NewAuto(outputs[0])

	// Train with provided data
	err := engine.Train(inputs, outputs)
	if err != nil {
		return nil, fmt.Errorf("training error: %w", err)
	}

	return engine, nil
}

// WithModel loads a model from JSON
func (e *Engine) WithModel(modelJSON string) (*Model, error) {
	var model Model
	err := json.Unmarshal([]byte(modelJSON), &model)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal model: %w", err)
	}
	e.model = &model
	return &model, nil
}

// WithWeights loads weights from JSON
func (e *Engine) WithWeights(weightsJSON string) (*Weights, error) {
	var weights Weights
	err := json.Unmarshal([]byte(weightsJSON), &weights)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal weights: %w", err)
	}
	e.weights = &weights
	return &weights, nil
}

// WithConfig sets the training configuration
func (e *Engine) WithConfig(config *Config) *Engine {
	e.config = config
	return e
}

// Train trains the model with given input and output parameters
func (e *Engine) Train(inputs []map[string]interface{}, outputs []map[string]interface{}) error {
	if e.model == nil {
		return fmt.Errorf("model not initialized")
	}

	if len(inputs) != len(outputs) {
		return fmt.Errorf("number of input samples (%d) must match number of output samples (%d)", len(inputs), len(outputs))
	}

	if len(inputs) == 0 {
		return fmt.Errorf("no training data provided")
	}

	// Initialize weights if needed
	if e.weights == nil {
		e.weights = &Weights{
			Values: make(map[string]interface{}),
		}
	}

	// Delegate training to the model implementation
	return e.model.Train(inputs, outputs, e.weights, e.config)
}

// Predict performs inference on the trained model
func (e *Engine) Predict(input map[string]interface{}) (map[string]interface{}, error) {
	if e.model == nil {
		return nil, fmt.Errorf("model not initialized")
	}

	if e.weights == nil {
		return nil, fmt.Errorf("weights not initialized, model not trained")
	}

	// Delegate prediction to the model implementation
	return e.model.Predict(input, e.weights)
}

// GetModel serializes the current model to JSON
func (e *Engine) GetModel() (*string, error) {
	if e.model == nil {
		return nil, fmt.Errorf("model not initialized")
	}

	modelJSON := e.model.JSON()
	return &modelJSON, nil
}

// GetWeights serializes the current weights to JSON
func (e *Engine) GetWeights() (*string, error) {
	if e.weights == nil {
		return nil, fmt.Errorf("weights not initialized")
	}

	weightsJSON := e.weights.JSON()
	return &weightsJSON, nil
}
