package goml

import (
	"encoding/json"
)

// Weights stores the learned weights for the model
type Weights struct {
	Values map[string]interface{} `json:"values"`
}

// JSON serializes the weights to JSON
func (w *Weights) JSON() string {
	bytes, err := json.Marshal(w)
	if err != nil {
		return "{}"
	}
	return string(bytes)
}

// Get retrieves a weight value by key
func (w *Weights) Get(key string) (interface{}, bool) {
	val, exists := w.Values[key]
	return val, exists
}

// Set updates a weight value
func (w *Weights) Set(key string, value interface{}) {
	w.Values[key] = value
}

// GetFloat retrieves a weight as a float64
func (w *Weights) GetFloat(key string) (float64, bool) {
	val, exists := w.Values[key]
	if !exists {
		return 0, false
	}

	switch v := val.(type) {
	case float64:
		return v, true
	case int:
		return float64(v), true
	case int64:
		return float64(v), true
	default:
		return 0, false
	}
}
