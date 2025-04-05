package goml

import (
	"encoding/json"
)

// Model represents the model structure (linear, logistic, etc.)
type Model struct {
	Type              string                    `json:"type"`
	Parameters        map[string]interface{}    `json:"parameters"`
	Features          map[string]interface{}    `json:"features,omitempty"`          // Feature metadata (e.g., type, mean, min, max)
	Targets           map[string]interface{}    `json:"targets,omitempty"`           // Target metadata (e.g., type)
	Categories        map[string]map[string]int `json:"categories,omitempty"`        // Maps output names to category->index mappings
	FeatureCategories map[string]map[string]int `json:"feature_categories,omitempty"` // Maps categorical feature names to value->index mappings
}

// Train defines how the model is trained on data
func (m *Model) Train(inputs []map[string]interface{}, outputs []map[string]interface{}, weights *Weights, config *Config) error {
	// Different implementations based on model type
	switch m.Type {
	case "linear":
		return trainLinearModel(inputs, outputs, weights, config)
	case "logistic":
		return trainLogisticModel(inputs, outputs, weights, config)
	case "categorical":
		return trainCategoricalModel(inputs, outputs, weights, config, m)
	case "mixed":
		return trainMixedModel(inputs, outputs, weights, config, m)
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
	case "categorical":
		return predictCategoricalModel(input, weights, m)
	case "mixed":
		return predictMixedModel(input, weights, m)
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

// NewCategoricalModel creates a new categorical classification model
func NewCategoricalModel() *Model {
	return &Model{
		Type: "categorical",
		Parameters: map[string]interface{}{
			"bias": true,
		},
		Categories: make(map[string]map[string]int),
	}
}

// NewAutoModel automatically creates the right model type based on outputs
// It analyzes the provided sample of output data to determine model type
func NewAutoModel(outputSample map[string]interface{}) *Model {
	// Check if we have mixed output types
	hasString := false
	hasNumeric := false
	hasBoolean := false
	
	// Analyze all output types
	for _, val := range outputSample {
		switch v := val.(type) {
		case string:
			hasString = true
		case bool:
			hasBoolean = true
		case int, int32, int64, float32, float64:
			hasNumeric = true
		default:
			// Try to see if it's a numeric type that wasn't caught
			if IsSupportedNumericType(v) {
				hasNumeric = true
			}
		}
	}
	
	// If we have mixed types (string and numeric/boolean), use mixed model
	if (hasString && hasNumeric) || (hasString && hasBoolean) || (hasNumeric && hasBoolean) {
		return NewMixedModel()
	}
	
	// If all outputs are strings, use categorical
	if hasString {
		return NewCategoricalModel()
	}
	
	// Check if all values are binary (0/1) or boolean
	isLogistic := hasBoolean // Already logistic if we have boolean outputs
	
	// If we don't already know it's logistic, check numeric values
	if !isLogistic && hasNumeric {
		isLogistic = true // Assume it's logistic until proven otherwise
		
		for _, val := range outputSample {
			switch v := val.(type) {
			case int:
				if v != 0 && v != 1 {
					isLogistic = false
				}
			case float64:
				if v != 0.0 && v != 1.0 {
					isLogistic = false
				}
			default:
				// Other types already handled above
			}
		}
	}
	
	// Create and return the appropriate model
	if isLogistic {
		return NewLogisticModel()
	} else {
		// Default to linear for all other cases
		return NewLinearModel()
	}
}