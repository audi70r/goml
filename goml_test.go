package main

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/audi70r/goml/pkg/goml"
)

// TestLinearModelIntegration tests linear model integration with numeric values
func TestLinearModelIntegration(t *testing.T) {
	// Create a new ML engine
	engine := goml.New()

	// Create a linear regression model
	model := goml.NewLinearModel()

	// Set the model to the engine
	engine.WithModel(model.JSON())

	// Custom training configuration
	config := &goml.Config{
		LearningRate: 0.0001, // Reduced learning rate for stability
		Epochs:       500,
		BatchSize:    4,
		Regularize:   0.0001, // Reduced regularization
		Tolerance:    0.01,   // Increased tolerance
	}

	// Apply config to engine
	engine.WithConfig(config)

	// Training data with multiple output parameters
	inputs := []map[string]interface{}{
		{"size": 1000, "bedrooms": 2, "location": "suburban"},
		{"size": 1500, "bedrooms": 3, "location": "suburban"},
		{"size": 800, "bedrooms": 1, "location": "urban"},
		{"size": 2000, "bedrooms": 4, "location": "rural"},
	}

	outputs := []map[string]interface{}{
		{"price": 200000, "rental": 1500},
		{"price": 300000, "rental": 2100},
		{"price": 220000, "rental": 1700},
		{"price": 350000, "rental": 2400},
	}

	// Train the model
	err := engine.Train(inputs, outputs)
	if err != nil {
		t.Fatalf("Training error: %v", err)
	}

	// Make a prediction
	newHouse := map[string]interface{}{
		"size":     1300,
		"bedrooms": 2,
		"location": "suburban",
	}

	prediction, err := engine.Predict(newHouse)
	if err != nil {
		t.Fatalf("Prediction error: %v", err)
	}

	// Verify prediction format
	if _, ok := prediction["price"].(float64); !ok {
		t.Error("Expected price to be float64")
	}

	if _, ok := prediction["rental"].(float64); !ok {
		t.Error("Expected rental to be float64")
	}

	// Get trained weights
	weightsJSON, err := engine.GetWeights()
	if err != nil {
		t.Fatalf("Error getting weights: %v", err)
	}

	// Save and reload the model
	savedModelJSON, _ := engine.GetModel()
	savedWeightsJSON := *weightsJSON

	// Create a new engine and load the saved model/weights
	newEngine := goml.New()
	newEngine.WithModel(*savedModelJSON)
	newEngine.WithWeights(savedWeightsJSON)

	// Make another prediction using the loaded model
	newPrediction, _ := newEngine.Predict(newHouse)

	// Verify original and reloaded models predict similar values
	// Due to floating point precision issues, compare with a tolerance
	priceDiff := math.Abs(prediction["price"].(float64) - newPrediction["price"].(float64))
	rentalDiff := math.Abs(prediction["rental"].(float64) - newPrediction["rental"].(float64))

	if priceDiff > 0.01 || rentalDiff > 0.01 {
		t.Errorf("Reloaded model made different predictions: original=%v, new=%v",
			prediction, newPrediction)
	}
}

// TestMixedInputTypesIntegration tests the model with various input types
func TestMixedInputTypesIntegration(t *testing.T) {
	// Training data with mixed input types
	inputs := []map[string]interface{}{
		{"numeric": 10.5, "integer": 1, "boolean": true, "category": "red"},
		{"numeric": 20.5, "integer": 2, "boolean": false, "category": "blue"},
		{"numeric": 30.5, "integer": 3, "boolean": true, "category": "green"},
		{"numeric": 40.5, "integer": 4, "boolean": false, "category": "yellow"},
		{"numeric": 50.5, "integer": 5, "boolean": true, "category": "orange"},
	}

	outputs := []map[string]interface{}{
		{"value": 100},
		{"value": 200},
		{"value": 300},
		{"value": 400},
		{"value": 500},
	}

	// Create linear model
	engine := goml.New()
	model := goml.NewLinearModel()
	engine.WithModel(model.JSON())

	// Train with mixed inputs
	err := engine.Train(inputs, outputs)
	if err != nil {
		t.Fatalf("Training with mixed inputs failed: %v", err)
	}

	// Predict with mixed inputs
	prediction, err := engine.Predict(map[string]interface{}{
		"numeric":  35.5,
		"integer":  3,
		"boolean":  true,
		"category": "blue",
	})

	if err != nil {
		t.Fatalf("Prediction with mixed inputs failed: %v", err)
	}

	// Value should exist and be numeric
	if _, ok := prediction["value"].(float64); !ok {
		t.Error("Expected numeric value in prediction")
	}
}

// TestAllModelTypesIntegration tests all model types in one test
func TestAllModelTypesIntegration(t *testing.T) {
	// Common inputs
	inputs := []map[string]interface{}{
		{"f1": 1.0, "f2": 2.0, "f3": true, "f4": "red"},
		{"f1": 2.0, "f2": 3.0, "f3": false, "f4": "blue"},
		{"f1": 3.0, "f2": 4.0, "f3": true, "f4": "green"},
		{"f1": 4.0, "f2": 5.0, "f3": false, "f4": "yellow"},
	}

	// Different output types for different models
	linearOutputs := []map[string]interface{}{
		{"value": 10.0},
		{"value": 20.0},
		{"value": 30.0},
		{"value": 40.0},
	}

	logisticOutputs := []map[string]interface{}{
		{"passed": true},
		{"passed": false},
		{"passed": true},
		{"passed": false},
	}

	categoricalOutputs := []map[string]interface{}{
		{"color": "primary"},
		{"color": "secondary"},
		{"color": "primary"},
		{"color": "secondary"},
	}

	// Test linear model
	linearEngine := goml.New()
	_, _ = linearEngine.WithModel(goml.NewLinearModel().JSON())
	err := linearEngine.Train(inputs, linearOutputs)
	if err != nil {
		t.Fatalf("Linear model training failed: %v", err)
	}

	// Test logistic model
	logisticEngine := goml.New()
	_, _ = logisticEngine.WithModel(goml.NewLogisticModel().JSON())
	err = logisticEngine.Train(inputs, logisticOutputs)
	if err != nil {
		t.Fatalf("Logistic model training failed: %v", err)
	}

	// Test categorical model
	categoricalEngine := goml.New()
	_, _ = categoricalEngine.WithModel(goml.NewCategoricalModel().JSON())
	err = categoricalEngine.Train(inputs, categoricalOutputs)
	if err != nil {
		t.Fatalf("Categorical model training failed: %v", err)
	}

	// Make prediction with all models
	testInput := map[string]interface{}{
		"f1": 2.5,
		"f2": 3.5,
		"f3": true,
		"f4": "blue",
	}

	linearPred, _ := linearEngine.Predict(testInput)
	logisticPred, _ := logisticEngine.Predict(testInput)
	categoricalPred, _ := categoricalEngine.Predict(testInput)

	// Verify prediction types
	if _, ok := linearPred["value"].(float64); !ok {
		t.Error("Linear model should predict float64")
	}

	if _, ok := logisticPred["passed"].(bool); !ok {
		// If not boolean directly, it should be a probability
		if prob, ok := logisticPred["passed"].(float64); !ok || prob < 0 || prob > 1 {
			t.Error("Logistic model should predict boolean or probability")
		}
	}

	if _, ok := categoricalPred["color"].(string); !ok {
		t.Error("Categorical model should predict string")
	}
}

// TestAutoDetectionIntegration tests the automatic model selection functionality
func TestAutoDetectionIntegration(t *testing.T) {
	// Common inputs for all tests
	inputs := []map[string]interface{}{
		{"feature1": 1.0, "feature2": 2.0},
		{"feature1": 2.0, "feature2": 3.0},
		{"feature1": 3.0, "feature2": 4.0},
	}

	// Test auto-detection with numeric output
	numericOutputs := []map[string]interface{}{
		{"result": 5.0},
		{"result": 8.0},
		{"result": 11.0},
	}

	numericEngine, err := goml.TrainAuto(inputs, numericOutputs)
	if err != nil {
		t.Fatalf("Auto training with numeric outputs failed: %v", err)
	}

	numModelJSON, _ := numericEngine.GetModel()
	var numModel goml.Model
	json.Unmarshal([]byte(*numModelJSON), &numModel)
	if numModel.Type != "linear" {
		t.Errorf("Auto-detection should have selected linear model for numeric output, got %s", numModel.Type)
	}

	// Test auto-detection with boolean output
	boolOutputs := []map[string]interface{}{
		{"result": true},
		{"result": false},
		{"result": true},
	}

	boolEngine, err := goml.TrainAuto(inputs, boolOutputs)
	if err != nil {
		t.Fatalf("Auto training with boolean outputs failed: %v", err)
	}

	boolModelJSON, _ := boolEngine.GetModel()
	var boolModel goml.Model
	json.Unmarshal([]byte(*boolModelJSON), &boolModel)
	if boolModel.Type != "logistic" {
		t.Errorf("Auto-detection should have selected logistic model for boolean output, got %s", boolModel.Type)
	}

	// Test auto-detection with string output
	stringOutputs := []map[string]interface{}{
		{"result": "small"},
		{"result": "medium"},
		{"result": "large"},
	}

	stringEngine, err := goml.TrainAuto(inputs, stringOutputs)
	if err != nil {
		t.Fatalf("Auto training with string outputs failed: %v", err)
	}

	stringModelJSON, _ := stringEngine.GetModel()
	var stringModel goml.Model
	json.Unmarshal([]byte(*stringModelJSON), &stringModel)
	if stringModel.Type != "categorical" {
		t.Errorf("Auto-detection should have selected categorical model for string output, got %s", stringModel.Type)
	}

	// Verify all models can predict
	testInput := map[string]interface{}{"feature1": 4.0, "feature2": 5.0}

	_, err1 := numericEngine.Predict(testInput)
	_, err2 := boolEngine.Predict(testInput)
	_, err3 := stringEngine.Predict(testInput)

	if err1 != nil || err2 != nil || err3 != nil {
		t.Errorf("Prediction errors: %v, %v, %v", err1, err2, err3)
	}
}

// TestMultipleOutputsIntegration tests handling various combinations of output types
func TestMultipleOutputsIntegration(t *testing.T) {
	// Common inputs
	inputs := []map[string]interface{}{
		{"feature1": 1.0, "feature2": 2.0},
		{"feature1": 2.0, "feature2": 3.0},
		{"feature1": 3.0, "feature2": 4.0},
		{"feature1": 4.0, "feature2": 5.0},
	}

	// Complex test with multiple outputs of different types
	complexOutputs := []map[string]interface{}{
		{"price": 100.0, "category": "small", "approved": true},
		{"price": 200.0, "category": "medium", "approved": false},
		{"price": 300.0, "category": "large", "approved": true},
		{"price": 400.0, "category": "extra", "approved": false},
	}

	// Create separate models for each output type
	numericEngine := goml.New()
	_, _ = numericEngine.WithModel(goml.NewLinearModel().JSON())

	categoricalEngine := goml.New()
	_, _ = categoricalEngine.WithModel(goml.NewCategoricalModel().JSON())

	logisticEngine := goml.New()
	_, _ = logisticEngine.WithModel(goml.NewLogisticModel().JSON())

	// Prepare separate outputs
	numericOutputs := make([]map[string]interface{}, len(complexOutputs))
	categoricalOutputs := make([]map[string]interface{}, len(complexOutputs))
	logisticOutputs := make([]map[string]interface{}, len(complexOutputs))

	for i, out := range complexOutputs {
		numericOutputs[i] = map[string]interface{}{"price": out["price"]}
		categoricalOutputs[i] = map[string]interface{}{"category": out["category"]}
		logisticOutputs[i] = map[string]interface{}{"approved": out["approved"]}
	}

	// Train all models
	err1 := numericEngine.Train(inputs, numericOutputs)
	err2 := categoricalEngine.Train(inputs, categoricalOutputs)
	err3 := logisticEngine.Train(inputs, logisticOutputs)

	if err1 != nil || err2 != nil || err3 != nil {
		t.Fatalf("Training errors: %v, %v, %v", err1, err2, err3)
	}

	// Make predictions with all models
	testInput := map[string]interface{}{"feature1": 2.5, "feature2": 3.5}

	numPred, _ := numericEngine.Predict(testInput)
	catPred, _ := categoricalEngine.Predict(testInput)
	logPred, _ := logisticEngine.Predict(testInput)

	// Combine predictions
	result := map[string]interface{}{}
	for k, v := range numPred {
		result[k] = v
	}
	for k, v := range catPred {
		result[k] = v
	}
	for k, v := range logPred {
		result[k] = v
	}

	// Check all output types are present
	if _, ok := result["price"].(float64); !ok {
		t.Error("Missing numeric output")
	}

	if _, ok := result["category"].(string); !ok {
		t.Error("Missing categorical output")
	}

	// Check either boolean or probability
	_, okBool := result["approved"].(bool)
	_, okFloat := result["approved"].(float64)
	if !okBool && !okFloat {
		t.Error("Missing boolean/probability output")
	}

	// Verify result by printing
	fmt.Printf("Combined prediction result: %+v\n", result)
}

// Helper function to check if a string contains a substring
func contains(s, substr string) bool {
	return strings.Contains(s, substr)
}
