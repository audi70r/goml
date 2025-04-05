package goml

import (
	"testing"
)

// TestEmptyTrainingData tests behavior with empty training data
func TestEmptyTrainingData(t *testing.T) {
	engine := New()
	model := NewLinearModel()
	engine.WithModel(model.JSON())

	// Empty inputs and outputs
	inputs := []map[string]interface{}{}
	outputs := []map[string]interface{}{}

	err := engine.Train(inputs, outputs)
	if err == nil {
		t.Error("Expected error for empty training data, got nil")
	}
}

// TestMismatchedTrainingData tests behavior with mismatched input/output counts
func TestMismatchedTrainingData(t *testing.T) {
	engine := New()
	model := NewLinearModel()
	engine.WithModel(model.JSON())

	// Mismatched counts
	inputs := []map[string]interface{}{
		{"x": 1},
		{"x": 2},
		{"x": 3},
	}
	outputs := []map[string]interface{}{
		{"y": 1},
		{"y": 2},
	}

	err := engine.Train(inputs, outputs)
	if err == nil {
		t.Error("Expected error for mismatched input/output counts, got nil")
	}
}

// TestMissingModel tests behavior when model is missing
func TestMissingModel(t *testing.T) {
	engine := New()
	// No model set

	inputs := []map[string]interface{}{
		{"x": 1},
		{"x": 2},
	}
	outputs := []map[string]interface{}{
		{"y": 1},
		{"y": 2},
	}

	err := engine.Train(inputs, outputs)
	if err == nil {
		t.Error("Expected error for missing model, got nil")
	}

	_, err = engine.Predict(map[string]interface{}{"x": 3})
	if err == nil {
		t.Error("Expected error for prediction with missing model, got nil")
	}
}

// TestUninitializedWeights tests prediction behavior with uninitialized weights
func TestUninitializedWeights(t *testing.T) {
	engine := New()
	model := NewLinearModel()
	engine.WithModel(model.JSON())
	// No training, so weights are uninitialized

	_, err := engine.Predict(map[string]interface{}{"x": 1})
	if err == nil {
		t.Error("Expected error for prediction with uninitialized weights, got nil")
	}
}

// TestConfigDefaults tests default configuration
func TestConfigDefaults(t *testing.T) {
	config := DefaultConfig()

	if config.LearningRate <= 0 {
		t.Error("Expected positive learning rate in default config")
	}

	if config.Epochs <= 0 {
		t.Error("Expected positive epochs in default config")
	}

	if config.BatchSize <= 0 {
		t.Error("Expected positive batch size in default config")
	}
}

// TestTrainMultipleTimes tests training a model multiple times
func TestTrainMultipleTimes(t *testing.T) {
	engine := New()
	model := NewLinearModel()
	engine.WithModel(model.JSON())

	// First training
	inputs1 := []map[string]interface{}{
		{"x": 1.0},
		{"x": 2.0},
	}
	outputs1 := []map[string]interface{}{
		{"y": 2.0},
		{"y": 4.0},
	}

	err := engine.Train(inputs1, outputs1)
	if err != nil {
		t.Fatalf("First training failed: %v", err)
	}

	// Get weights after first training
	weights1, _ := engine.GetWeights()

	// Second training with different data
	inputs2 := []map[string]interface{}{
		{"x": 1.0},
		{"x": 2.0},
	}
	outputs2 := []map[string]interface{}{
		{"y": 3.0},
		{"y": 6.0},
	}

	err = engine.Train(inputs2, outputs2)
	if err != nil {
		t.Fatalf("Second training failed: %v", err)
	}

	// Get weights after second training
	weights2, _ := engine.GetWeights()

	// Weights should be different after retraining
	if *weights1 == *weights2 {
		t.Error("Expected weights to change after retraining")
	}
}

// TestInvalidModelJSON tests loading invalid model JSON
func TestInvalidModelJSON(t *testing.T) {
	engine := New()

	// Invalid JSON
	invalidJSON := "{not valid json}"

	_, err := engine.WithModel(invalidJSON)
	if err == nil {
		t.Error("Expected error for invalid model JSON, got nil")
	}
}

// TestInvalidWeightsJSON tests loading invalid weights JSON
func TestInvalidWeightsJSON(t *testing.T) {
	engine := New()

	// Invalid JSON
	invalidJSON := "{not valid json}"

	_, err := engine.WithWeights(invalidJSON)
	if err == nil {
		t.Error("Expected error for invalid weights JSON, got nil")
	}
}

// TestModelType tests behavior with unsupported model type
func TestUnsupportedModelType(t *testing.T) {
	// Create a model with unsupported type
	model := &Model{
		Type: "unsupported",
		Parameters: map[string]interface{}{
			"bias": true,
		},
	}

	engine := New()
	engine.model = model

	inputs := []map[string]interface{}{
		{"x": 1},
		{"x": 2},
	}
	outputs := []map[string]interface{}{
		{"y": 1},
		{"y": 2},
	}

	// Training should fail
	err := engine.Train(inputs, outputs)
	if err == nil || err != ErrUnsupportedModelType {
		t.Errorf("Expected unsupported model type error, got: %v", err)
	}

	// Prediction should fail
	_, err = engine.Predict(map[string]interface{}{"x": 3})
	if err == nil || err != ErrUnsupportedModelType {
		t.Errorf("Expected unsupported model type error, got: %v", err)
	}
}

// TestMixedTypes tests mixed input types with all models
func TestMixedTypes(t *testing.T) {
	// Create inputs with mixed types
	inputs := []map[string]interface{}{
		{"numeric": 1.0, "boolean": true, "string": "red", "integer": 10},
		{"numeric": 2.0, "boolean": false, "string": "blue", "integer": 20},
		{"numeric": 3.0, "boolean": true, "string": "green", "integer": 30},
	}

	// Test with linear model
	linearModel := NewLinearModel()
	linearEngine := New()
	_, _ = linearEngine.WithModel(linearModel.JSON())

	linearOutputs := []map[string]interface{}{
		{"value": 10.0},
		{"value": 20.0},
		{"value": 30.0},
	}

	err := linearEngine.Train(inputs, linearOutputs)
	if err != nil {
		t.Errorf("Linear model failed with mixed input types: %v", err)
	}

	// Test with logistic model
	logisticModel := NewLogisticModel()
	logisticEngine := New()
	_, _ = logisticEngine.WithModel(logisticModel.JSON())

	logisticOutputs := []map[string]interface{}{
		{"value": true},
		{"value": false},
		{"value": true},
	}

	err = logisticEngine.Train(inputs, logisticOutputs)
	if err != nil {
		t.Errorf("Logistic model failed with mixed input types: %v", err)
	}

	// Test with categorical model
	categoricalModel := NewCategoricalModel()
	categoricalEngine := New()
	_, _ = categoricalEngine.WithModel(categoricalModel.JSON())

	categoricalOutputs := []map[string]interface{}{
		{"value": "small"},
		{"value": "medium"},
		{"value": "large"},
	}

	err = categoricalEngine.Train(inputs, categoricalOutputs)
	if err != nil {
		t.Errorf("Categorical model failed with mixed input types: %v", err)
	}
}

// TestGetModelWithNilModel tests GetModel with nil model
func TestGetModelWithNilModel(t *testing.T) {
	engine := New()
	// No model set

	_, err := engine.GetModel()
	if err == nil {
		t.Error("Expected error for GetModel with nil model, got nil")
	}
}

// TestGetWeightsWithNilWeights tests GetWeights with nil weights
func TestGetWeightsWithNilWeights(t *testing.T) {
	engine := New()
	// No weights set

	_, err := engine.GetWeights()
	if err == nil {
		t.Error("Expected error for GetWeights with nil weights, got nil")
	}
}

// TestEngineWithDefaultConfig tests training with default config
func TestEngineWithDefaultConfig(t *testing.T) {
	// Create engine without explicitly setting config
	engine := New()
	model := NewLinearModel()
	engine.WithModel(model.JSON())

	// The engine should have default config
	if engine.config == nil {
		t.Error("Engine should initialize with default config")
	}

	inputs := []map[string]interface{}{
		{"x": 1.0},
		{"x": 2.0},
	}
	outputs := []map[string]interface{}{
		{"y": 2.0},
		{"y": 4.0},
	}

	// Training should succeed with default config
	err := engine.Train(inputs, outputs)
	if err != nil {
		t.Errorf("Expected training to succeed with default config, got: %v", err)
	}
}
