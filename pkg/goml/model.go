package goml

import (
	"encoding/json"
)

// Model represents the model structure (linear, logistic, etc.)
type Model struct {
	Type       string                 `json:"type"`
	Parameters map[string]interface{} `json:"parameters"`
	Features   map[string]interface{} `json:"features,omitempty"` // Feature metadata (e.g., mean, min, max)
	Targets    map[string]interface{} `json:"targets,omitempty"`  // Target metadata
}

// Train defines how the model is trained on data
func (m *Model) Train(inputs []map[string]interface{}, outputs []map[string]interface{}, weights *Weights, config *Config) error {
	// Different implementations based on model type
	switch m.Type {
	case "linear":
		return trainLinearModel(inputs, outputs, weights, config)
	case "logistic":
		return trainLogisticModel(inputs, outputs, weights, config)
	default:
		return ErrUnsupportedModelType
	}
}

// Predict performs inference using the trained model
func (m *Model) Predict(input map[string]interface{}, weights *Weights) (map[string]interface{}, error) {
	// Different implementations based on model type
	switch m.Type {
	case "linear":
		return predictLinearModel(input, weights)
	case "logistic":
		return predictLogisticModel(input, weights)
	default:
		return nil, ErrUnsupportedModelType
	}
}

// JSON serializes the model to JSON
func (m *Model) JSON() string {
	bytes, err := json.Marshal(m)
	if err != nil {
		return "{}"
	}
	return string(bytes)
}

// NewLinearModel creates a new linear regression model
func NewLinearModel() *Model {
	return &Model{
		Type: "linear",
		Parameters: map[string]interface{}{
			"bias": true,
		},
	}
}

// NewLogisticModel creates a new logistic regression model
func NewLogisticModel() *Model {
	return &Model{
		Type: "logistic",
		Parameters: map[string]interface{}{
			"bias": true,
		},
	}
}
