package goml

import (
	"encoding/json"
	"testing"
)

// TestLinearModelNumericIO tests linear model with numeric inputs and outputs
func TestLinearModelNumericIO(t *testing.T) {
	// Create a new ML engine with linear model
	engine := New()
	model := NewLinearModel()
	engine.WithModel(model.JSON())

	// Training data with numeric inputs and outputs
	inputs := []map[string]interface{}{
		{"x1": 1.0, "x2": 2.0},
		{"x1": 2.0, "x2": 3.0},
		{"x1": 3.0, "x2": 4.0},
		{"x1": 4.0, "x2": 5.0},
	}

	outputs := []map[string]interface{}{
		{"y1": 5.0, "y2": 10.0},
		{"y1": 8.0, "y2": 16.0},
		{"y1": 11.0, "y2": 22.0},
		{"y1": 14.0, "y2": 28.0},
	}

	// Train
	err := engine.Train(inputs, outputs)
	if err != nil {
		t.Fatalf("Training error: %v", err)
	}

	// Test prediction
	prediction, err := engine.Predict(map[string]interface{}{"x1": 5.0, "x2": 6.0})
	if err != nil {
		t.Fatalf("Prediction error: %v", err)
	}

	// Verify prediction fields
	if _, ok := prediction["y1"]; !ok {
		t.Errorf("Missing y1 in prediction")
	}
	if _, ok := prediction["y2"]; !ok {
		t.Errorf("Missing y2 in prediction")
	}

	// Verify values are reasonable
	y1, _ := prediction["y1"].(float64)
	y2, _ := prediction["y2"].(float64)

	// Given linearity, y1 should be around 17 and y2 around 34
	if y1 < 15 || y1 > 19 {
		t.Errorf("y1 prediction outside expected range: %f", y1)
	}
	if y2 < 30 || y2 > 38 {
		t.Errorf("y2 prediction outside expected range: %f", y2)
	}
}

// TestLinearModelMixedIO tests linear model with mixed input types (numeric, string, bool) and numeric outputs
func TestLinearModelMixedIO(t *testing.T) {
	// Create a new ML engine with linear model
	engine := New()
	model := NewLinearModel()
	engine.WithModel(model.JSON())

	// Training data with mixed input types
	inputs := []map[string]interface{}{
		{"numeric": 1.0, "category": "red", "flag": true},
		{"numeric": 2.0, "category": "blue", "flag": false},
		{"numeric": 3.0, "category": "red", "flag": true},
		{"numeric": 4.0, "category": "blue", "flag": false},
	}

	outputs := []map[string]interface{}{
		{"price": 100.0},
		{"price": 150.0},
		{"price": 200.0},
		{"price": 250.0},
	}

	// Train
	err := engine.Train(inputs, outputs)
	if err != nil {
		t.Fatalf("Training error: %v", err)
	}

	// Test prediction with mixed inputs
	prediction, err := engine.Predict(map[string]interface{}{
		"numeric":  2.5,
		"category": "red",
		"flag":     true,
	})
	if err != nil {
		t.Fatalf("Prediction error: %v", err)
	}

	// Verify prediction
	price, ok := prediction["price"].(float64)
	if !ok {
		t.Errorf("Missing or invalid price in prediction")
	}

	// Price should be between 150 and 200 given inputs
	if price < 140 || price > 210 {
		t.Errorf("Price prediction outside expected range: %f", price)
	}
}

// TestLogisticModelWithBooleanOutput tests logistic regression with boolean outputs
func TestLogisticModelWithBooleanOutput(t *testing.T) {
	// Create auto-detecting engine
	boolOutputs := map[string]interface{}{
		"passed": true,
	}
	engine := NewAuto(boolOutputs)

	// Verify it picked logistic regression
	modelJSON, _ := engine.GetModel()
	var model Model
	json.Unmarshal([]byte(*modelJSON), &model)
	if model.Type != "logistic" {
		t.Errorf("Expected logistic model for boolean output, got %s", model.Type)
	}

	// Training data
	inputs := []map[string]interface{}{
		{"hours": 1, "previous": 60},
		{"hours": 2, "previous": 65},
		{"hours": 3, "previous": 70},
		{"hours": 4, "previous": 75},
		{"hours": 5, "previous": 80},
		{"hours": 6, "previous": 85},
	}

	outputs := []map[string]interface{}{
		{"passed": false},
		{"passed": false},
		{"passed": false},
		{"passed": true},
		{"passed": true},
		{"passed": true},
	}

	// Train
	err := engine.Train(inputs, outputs)
	if err != nil {
		t.Fatalf("Training error with boolean output: %v", err)
	}

	// Predict for someone who studied 5.5 hours
	prediction, err := engine.Predict(map[string]interface{}{
		"hours":    5.5,
		"previous": 82,
	})
	if err != nil {
		t.Fatalf("Prediction error: %v", err)
	}

	// Check if result is boolean
	passed, ok := prediction["passed"].(bool)
	if !ok {
		// If not boolean, it might be a probability
		if passedVal, ok := prediction["passed"].(float64); ok {
			passed = passedVal >= 0.5
		} else {
			t.Fatalf("Expected boolean or float64 for 'passed', got %T", prediction["passed"])
		}
	}

	// Given the training data, this student should pass
	if !passed {
		t.Errorf("Expected student with 5.5 hours to pass, but got %v", passed)
	}
}

// TestCategoricalModelWithStringOutput tests categorical model with string outputs
func TestCategoricalModelWithStringOutput(t *testing.T) {
	// Create auto-detecting engine
	stringOutputs := map[string]interface{}{
		"color": "red",
	}
	engine := NewAuto(stringOutputs)

	// Verify it picked categorical
	modelJSON, _ := engine.GetModel()
	var model Model
	json.Unmarshal([]byte(*modelJSON), &model)
	if model.Type != "categorical" {
		t.Errorf("Expected categorical model for string output, got %s", model.Type)
	}

	// Training data
	inputs := []map[string]interface{}{
		{"size": 10, "weight": 100, "premium": true},
		{"size": 20, "weight": 200, "premium": false},
		{"size": 30, "weight": 300, "premium": true},
		{"size": 40, "weight": 400, "premium": false},
		{"size": 50, "weight": 500, "premium": true},
	}

	outputs := []map[string]interface{}{
		{"color": "red"},
		{"color": "blue"},
		{"color": "green"},
		{"color": "blue"},
		{"color": "green"},
	}

	// Train
	err := engine.Train(inputs, outputs)
	if err != nil {
		t.Fatalf("Training error with string output: %v", err)
	}

	// Predictions
	p1, _ := engine.Predict(map[string]interface{}{"size": 15, "weight": 150, "premium": true})
	p2, _ := engine.Predict(map[string]interface{}{"size": 35, "weight": 350, "premium": false})

	// Check if results are strings
	_, ok1 := p1["color"].(string)
	_, ok2 := p2["color"].(string)

	if !ok1 || !ok2 {
		t.Fatalf("Expected string outputs, got %T and %T", p1["color"], p2["color"])
	}

	// Should get probabilities too
	if _, ok := p1["color_probs"].(map[string]float64); !ok {
		t.Errorf("Missing color_probs in prediction")
	}
}

// TestMultipleOutputTypes tests models with multiple output fields of different types
func TestMultipleOutputTypes(t *testing.T) {
	// Training data
	inputs := []map[string]interface{}{
		{"feature1": 1.0, "feature2": "A", "feature3": true},
		{"feature1": 2.0, "feature2": "B", "feature3": false},
		{"feature1": 3.0, "feature2": "C", "feature3": true},
		{"feature1": 4.0, "feature2": "A", "feature3": false},
		{"feature1": 5.0, "feature2": "B", "feature3": true},
		{"feature1": 6.0, "feature2": "C", "feature3": false},
	}

	// Mixed outputs
	mixedOutputs := []map[string]interface{}{
		{"numeric": 10.0, "category": "small", "passed": true},
		{"numeric": 20.0, "category": "medium", "passed": false},
		{"numeric": 30.0, "category": "large", "passed": true},
		{"numeric": 40.0, "category": "small", "passed": false},
		{"numeric": 50.0, "category": "medium", "passed": true},
		{"numeric": 60.0, "category": "large", "passed": false},
	}

	// Train separate models
	engineNumeric := New()
	_, _ = engineNumeric.WithModel(NewLinearModel().JSON())

	engineCategory := New()
	_, _ = engineCategory.WithModel(NewCategoricalModel().JSON())

	engineBinary := New()
	_, _ = engineBinary.WithModel(NewLogisticModel().JSON())

	// Prepare outputs for each model
	numericOutputs := make([]map[string]interface{}, len(mixedOutputs))
	categoryOutputs := make([]map[string]interface{}, len(mixedOutputs))
	binaryOutputs := make([]map[string]interface{}, len(mixedOutputs))

	for i, out := range mixedOutputs {
		numericOutputs[i] = map[string]interface{}{"numeric": out["numeric"]}
		categoryOutputs[i] = map[string]interface{}{"category": out["category"]}
		binaryOutputs[i] = map[string]interface{}{"passed": out["passed"]}
	}

	// Train all models
	err1 := engineNumeric.Train(inputs, numericOutputs)
	err2 := engineCategory.Train(inputs, categoryOutputs)
	err3 := engineBinary.Train(inputs, binaryOutputs)

	if err1 != nil || err2 != nil || err3 != nil {
		t.Fatalf("Training errors: %v, %v, %v", err1, err2, err3)
	}

	// Test prediction
	testInput := map[string]interface{}{
		"feature1": 3.5,
		"feature2": "B",
		"feature3": true,
	}

	numPred, _ := engineNumeric.Predict(testInput)
	catPred, _ := engineCategory.Predict(testInput)
	binPred, _ := engineBinary.Predict(testInput)

	// Combine predictions
	combinedPred := make(map[string]interface{})
	for k, v := range numPred {
		combinedPred[k] = v
	}
	for k, v := range catPred {
		combinedPred[k] = v
	}
	for k, v := range binPred {
		combinedPred[k] = v
	}

	// Verify all outputs are present
	if _, ok := combinedPred["numeric"].(float64); !ok {
		t.Errorf("Missing numeric output")
	}
	if _, ok := combinedPred["category"].(string); !ok {
		t.Errorf("Missing category output")
	}
	if _, ok := combinedPred["passed"].(bool); !ok && combinedPred["passed"] != nil {
		// Check if it's a probability if not boolean
		if _, ok := combinedPred["passed"].(float64); !ok {
			t.Errorf("Missing passed output")
		}
	}
}

// TestAutoDetection tests the automatic model detection feature
func TestAutoDetection(t *testing.T) {
	testCases := []struct {
		name          string
		outputSample  map[string]interface{}
		expectedType  string
		trainingData  []map[string]interface{}
		trainingLabel []map[string]interface{}
	}{
		{
			name:         "Numeric Outputs",
			outputSample: map[string]interface{}{"price": 100.0, "quantity": 5.0},
			expectedType: "linear",
			trainingData: []map[string]interface{}{
				{"f1": 1.0, "f2": 2.0},
				{"f1": 2.0, "f2": 3.0},
			},
			trainingLabel: []map[string]interface{}{
				{"price": 100.0, "quantity": 5.0},
				{"price": 200.0, "quantity": 10.0},
			},
		},
		{
			name:         "String Outputs",
			outputSample: map[string]interface{}{"color": "red", "size": "large"},
			expectedType: "categorical",
			trainingData: []map[string]interface{}{
				{"f1": 1.0, "f2": 2.0},
				{"f1": 2.0, "f2": 3.0},
			},
			trainingLabel: []map[string]interface{}{
				{"color": "red", "size": "large"},
				{"color": "blue", "size": "small"},
			},
		},
		{
			name:         "Binary Int Outputs",
			outputSample: map[string]interface{}{"passed": 1, "approved": 0},
			expectedType: "logistic",
			trainingData: []map[string]interface{}{
				{"f1": 1.0, "f2": 2.0},
				{"f1": 2.0, "f2": 3.0},
			},
			trainingLabel: []map[string]interface{}{
				{"passed": 1, "approved": 0},
				{"passed": 0, "approved": 1},
			},
		},
		{
			name:         "Boolean Outputs",
			outputSample: map[string]interface{}{"passed": true, "approved": false},
			expectedType: "logistic",
			trainingData: []map[string]interface{}{
				{"f1": 1.0, "f2": 2.0},
				{"f1": 2.0, "f2": 3.0},
			},
			trainingLabel: []map[string]interface{}{
				{"passed": true, "approved": false},
				{"passed": false, "approved": true},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Test auto detection
			engine := NewAuto(tc.outputSample)

			// Verify model type
			modelJSON, _ := engine.GetModel()
			var model Model
			json.Unmarshal([]byte(*modelJSON), &model)
			if model.Type != tc.expectedType {
				t.Errorf("Expected %s model, got %s", tc.expectedType, model.Type)
			}

			// Verify it can be trained
			err := engine.Train(tc.trainingData, tc.trainingLabel)
			if err != nil {
				t.Errorf("Training error: %v", err)
			}

			// Verify prediction works
			_, err = engine.Predict(tc.trainingData[0])
			if err != nil {
				t.Errorf("Prediction error: %v", err)
			}
		})
	}
}

// TestSerialization tests model and weights serialization/deserialization
func TestSerialization(t *testing.T) {
	// Create and train a model
	engine := New()
	model := NewLinearModel()
	engine.WithModel(model.JSON())

	inputs := []map[string]interface{}{
		{"x": 1.0, "y": 2.0},
		{"x": 2.0, "y": 3.0},
		{"x": 3.0, "y": 4.0},
	}

	outputs := []map[string]interface{}{
		{"z": 5.0},
		{"z": 8.0},
		{"z": 11.0},
	}

	err := engine.Train(inputs, outputs)
	if err != nil {
		t.Fatalf("Training error: %v", err)
	}

	// Get model and weights JSON
	modelJSON, _ := engine.GetModel()
	weightsJSON, _ := engine.GetWeights()

	// Create a new engine and load the serialized model/weights
	newEngine := New()
	_, err1 := newEngine.WithModel(*modelJSON)
	_, err2 := newEngine.WithWeights(*weightsJSON)

	if err1 != nil || err2 != nil {
		t.Fatalf("Deserialization errors: %v, %v", err1, err2)
	}

	// Verify both make the same predictions
	testInput := map[string]interface{}{"x": 4.0, "y": 5.0}

	pred1, _ := engine.Predict(testInput)
	pred2, _ := newEngine.Predict(testInput)

	// Check predictions are the same
	z1 := pred1["z"].(float64)
	z2 := pred2["z"].(float64)

	if z1 != z2 {
		t.Errorf("Predictions differ after serialization: %f vs %f", z1, z2)
	}
}

// TestTrainAutoConvenience tests the TrainAuto convenience method
func TestTrainAutoConvenience(t *testing.T) {
	// Prepare data
	inputs := []map[string]interface{}{
		{"f1": 1.0, "f2": true, "f3": "red"},
		{"f1": 2.0, "f2": false, "f3": "blue"},
		{"f1": 3.0, "f2": true, "f3": "green"},
	}

	// Test cases for different output types
	testCases := []struct {
		name         string
		outputs      []map[string]interface{}
		expectedType string
	}{
		{
			name: "Numeric Outputs",
			outputs: []map[string]interface{}{
				{"value": 10.0},
				{"value": 20.0},
				{"value": 30.0},
			},
			expectedType: "linear",
		},
		{
			name: "String Outputs",
			outputs: []map[string]interface{}{
				{"category": "small"},
				{"category": "medium"},
				{"category": "large"},
			},
			expectedType: "categorical",
		},
		{
			name: "Boolean Outputs",
			outputs: []map[string]interface{}{
				{"passed": true},
				{"passed": false},
				{"passed": true},
			},
			expectedType: "logistic",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Use the TrainAuto convenience method
			engine, err := TrainAuto(inputs, tc.outputs)
			if err != nil {
				t.Fatalf("TrainAuto error: %v", err)
			}

			// Verify model type
			modelJSON, _ := engine.GetModel()
			var model Model
			json.Unmarshal([]byte(*modelJSON), &model)
			if model.Type != tc.expectedType {
				t.Errorf("Expected %s model, got %s", tc.expectedType, model.Type)
			}

			// Verify it makes predictions
			_, err = engine.Predict(inputs[0])
			if err != nil {
				t.Errorf("Prediction error: %v", err)
			}
		})
	}
}
