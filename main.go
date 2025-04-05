package main

import (
	"encoding/json"
	"fmt"

	"github.com/audi70r/goml/pkg/goml"
)

func main() {
	// Create a new ML engine
	engine := goml.New()

	// Create a linear regression model
	model := goml.NewLinearModel()

	// Serialize model to JSON
	modelJSON := model.JSON()
	fmt.Println("Model JSON:")
	fmt.Println(modelJSON)

	// Set the model to the engine
	engine.WithModel(modelJSON)

	// Custom training configuration
	config := &goml.Config{
		LearningRate: 0.000000001, // Very small learning rate for large values
		Epochs:       10000,
		BatchSize:    6, // use all examples in each batch
		Regularize:   0.00000001,
		Tolerance:    0.001,
	}

	// Apply config to engine
	engine.WithConfig(config)

	// Training data with multiple output parameters
	inputs := []map[string]interface{}{
		{"size": 1000, "bedrooms": 2, "location": "suburban"},
		{"size": 1500, "bedrooms": 3, "location": "suburban"},
		{"size": 800, "bedrooms": 1, "location": "urban"},
		{"size": 2000, "bedrooms": 4, "location": "rural"},
		{"size": 1200, "bedrooms": 2, "location": "urban"},
		{"size": 1700, "bedrooms": 3, "location": "rural"},
	}

	outputs := []map[string]interface{}{
		{"price": 200000, "rental": 1500},
		{"price": 300000, "rental": 2100},
		{"price": 220000, "rental": 1700},
		{"price": 350000, "rental": 2400},
		{"price": 240000, "rental": 1900},
		{"price": 320000, "rental": 2300},
	}

	// Train the model
	err := engine.Train(inputs, outputs)
	if err != nil {
		fmt.Printf("Training error: %v\n", err)
		return
	}

	// Get trained weights
	weightsJSON, err := engine.GetWeights()
	if err != nil {
		fmt.Printf("Error getting weights: %v\n", err)
		return
	}

	fmt.Println("\nTrained weights:")
	fmt.Println(*weightsJSON)

	// Make a prediction
	newHouse := map[string]interface{}{
		"size":     1300,
		"bedrooms": 2,
		"location": "suburban",
	}

	prediction, err := engine.Predict(newHouse)
	if err != nil {
		fmt.Printf("Prediction error: %v\n", err)
		return
	}

	fmt.Println("\nPrediction for new house (price and rental):")
	predJSON, _ := json.MarshalIndent(prediction, "", "  ")
	fmt.Println(string(predJSON))

	// Demonstrating prediction with multiple outputs
	if price, ok := prediction["price"].(float64); ok {
		fmt.Printf("Predicted price: $%.2f\n", price)
	}
	if rental, ok := prediction["rental"].(float64); ok {
		fmt.Printf("Predicted monthly rental: $%.2f\n", rental)
	}

	// Save and reload the model
	fmt.Println("\nSaving and reloading model...")

	savedModelJSON, _ := engine.GetModel()
	savedWeightsJSON := *weightsJSON

	// Create a new engine and load the saved model/weights
	newEngine := goml.New()
	newEngine.WithModel(*savedModelJSON)
	newEngine.WithWeights(savedWeightsJSON)

	// Make another prediction using the loaded model
	newPrediction, _ := newEngine.Predict(newHouse)

	fmt.Println("\nPrediction using reloaded model:")
	newPredJSON, _ := json.MarshalIndent(newPrediction, "", "  ")
	fmt.Println(string(newPredJSON))

	// Demonstrating multiple outputs from reloaded model
	if price, ok := newPrediction["price"].(float64); ok {
		fmt.Printf("Predicted price (reloaded model): $%.2f\n", price)
	}
	if rental, ok := newPrediction["rental"].(float64); ok {
		fmt.Printf("Predicted monthly rental (reloaded model): $%.2f\n", rental)
	}

	// Logistic regression example
	fmt.Println("\n\nLogistic Regression Example")
	fmt.Println("---------------------------")

	// Create a logistic regression model
	logisticModel := goml.NewLogisticModel()
	logisticEngine := goml.New()
	logisticEngine.WithModel(logisticModel.JSON())
	logisticEngine.WithConfig(&goml.Config{
		LearningRate: 0.01,
		Epochs:       1000,
		BatchSize:    6, // use all examples
		Regularize:   0.001,
		Tolerance:    0.0001,
	})

	// Example: Predict if a student will pass an exam based on study hours and previous grades
	logisticInputs := []map[string]interface{}{
		{"study_hours": 8, "prev_grade": 85},
		{"study_hours": 3, "prev_grade": 60},
		{"study_hours": 5, "prev_grade": 70},
		{"study_hours": 10, "prev_grade": 90},
		{"study_hours": 2, "prev_grade": 50},
		{"study_hours": 7, "prev_grade": 80},
	}

	logisticOutputs := []map[string]interface{}{
		{"pass": 1}, // will pass
		{"pass": 0}, // will fail
		{"pass": 0}, // will fail
		{"pass": 1}, // will pass
		{"pass": 0}, // will fail
		{"pass": 1}, // will pass
	}

	// Train the logistic model
	err = logisticEngine.Train(logisticInputs, logisticOutputs)
	if err != nil {
		fmt.Printf("Logistic training error: %v\n", err)
		return
	}

	// Get trained logistic weights
	logisticWeights, _ := logisticEngine.GetWeights()
	fmt.Println("\nTrained logistic weights:")
	fmt.Println(*logisticWeights)

	// Predict with the logistic model
	newStudent := map[string]interface{}{
		"study_hours": 6,
		"prev_grade":  75,
	}

	logisticPrediction, _ := logisticEngine.Predict(newStudent)

	fmt.Println("\nLogistic prediction for new student:")
	logisticPredJSON, _ := json.MarshalIndent(logisticPrediction, "", "  ")
	fmt.Println(string(logisticPredJSON))

	// Categorical model example with string outputs
	fmt.Println("\n\nCategorical Model Example")
	fmt.Println("---------------------------")

	// Create a categorical model
	categoricalModel := goml.NewCategoricalModel()
	categoricalEngine := goml.New()
	categoricalEngine.WithModel(categoricalModel.JSON())
	categoricalEngine.WithConfig(&goml.Config{
		LearningRate: 0.01,
		Epochs:       1000,
		BatchSize:    5,
		Regularize:   0.001,
		Tolerance:    0.0001,
	})

	// Example: Predict car type based on features
	categoricalInputs := []map[string]interface{}{
		{"engine_size": 2.0, "doors": 4, "weight": 1500},
		{"engine_size": 3.0, "doors": 2, "weight": 1800},
		{"engine_size": 1.6, "doors": 4, "weight": 1200},
		{"engine_size": 5.0, "doors": 2, "weight": 2200},
		{"engine_size": 1.2, "doors": 4, "weight": 1100},
	}

	// String output values
	categoricalOutputs := []map[string]interface{}{
		{"car_type": "sedan"},
		{"car_type": "sports"},
		{"car_type": "compact"},
		{"car_type": "luxury"},
		{"car_type": "economy"},
	}

	// Train the categorical model
	err = categoricalEngine.Train(categoricalInputs, categoricalOutputs)
	if err != nil {
		fmt.Printf("Categorical training error: %v\n", err)
		return
	}

	// Get trained categorical weights
	categoricalWeights, _ := categoricalEngine.GetWeights()
	fmt.Println("\nTrained categorical weights:")
	fmt.Println(*categoricalWeights)

	// Make prediction with string output
	newCar := map[string]interface{}{
		"engine_size": 2.5,
		"doors":       4,
		"weight":      1700,
	}

	categoricalPrediction, _ := categoricalEngine.Predict(newCar)

	fmt.Println("\nCategorical prediction for new car:")
	catPredJSON, _ := json.MarshalIndent(categoricalPrediction, "", "  ")
	fmt.Println(string(catPredJSON))

	// Now demonstrate multiple categorical outputs
	fmt.Println("\n\nMultiple Categorical Outputs Example")
	fmt.Println("----------------------------------")

	// Create model for multiple categorical outputs
	multiCatModel := goml.NewCategoricalModel()
	multiCatEngine := goml.New()
	multiCatEngine.WithModel(multiCatModel.JSON())
	multiCatEngine.WithConfig(&goml.Config{
		LearningRate: 0.01,
		Epochs:       2000,
		BatchSize:    6,
	})

	// Example: Predict both car type and color based on features
	multiCatInputs := []map[string]interface{}{
		{"engine_size": 2.0, "doors": 4, "weight": 1500, "year": 2020},
		{"engine_size": 3.0, "doors": 2, "weight": 1800, "year": 2022},
		{"engine_size": 1.6, "doors": 4, "weight": 1200, "year": 2019},
		{"engine_size": 5.0, "doors": 2, "weight": 2200, "year": 2023},
		{"engine_size": 1.2, "doors": 4, "weight": 1100, "year": 2018},
		{"engine_size": 2.5, "doors": 4, "weight": 1700, "year": 2021},
	}

	// Multiple string output values
	multiCatOutputs := []map[string]interface{}{
		{"car_type": "sedan", "color": "blue"},
		{"car_type": "sports", "color": "red"},
		{"car_type": "compact", "color": "silver"},
		{"car_type": "luxury", "color": "black"},
		{"car_type": "economy", "color": "white"},
		{"car_type": "sedan", "color": "green"},
	}

	// Train the model with multiple categorical outputs
	err = multiCatEngine.Train(multiCatInputs, multiCatOutputs)
	if err != nil {
		fmt.Printf("Multi-categorical training error: %v\n", err)
		return
	}

	// Make prediction with multiple string outputs
	newCarMulti := map[string]interface{}{
		"engine_size": 2.2,
		"doors":       4,
		"weight":      1600,
		"year":        2021,
	}

	multiCatPrediction, _ := multiCatEngine.Predict(newCarMulti)

	fmt.Println("\nMulti-categorical prediction (car type and color):")
	multiCatPredJSON, _ := json.MarshalIndent(multiCatPrediction, "", "  ")
	fmt.Println(string(multiCatPredJSON))

	fmt.Println("\n\nFull Type Support")
	fmt.Println("------------------")
	fmt.Println("The GOML package supports:")
	fmt.Println("1. All input types: int, float64, string, bool")
	fmt.Println("2. All output types: int, float64, string, bool")
	fmt.Println("3. Multiple outputs of different types in a single model or separate models")

	fmt.Println("\nFor mixed input/output types, use the automatic model selection:")
	fmt.Println("- goml.NewAuto(outputSample) - create engine with automatic model selection")
	fmt.Println("- goml.TrainAuto(inputs, outputs) - detect, create, and train in one step")
	fmt.Println("\nDemonstrating auto-detection:")

	// Example of auto-detection with numeric outputs
	numericOutputs := map[string]interface{}{
		"price":    100.0,
		"quantity": 5,
	}

	autoNumericEngine := goml.NewAuto(numericOutputs)
	autoModelJSON, _ := autoNumericEngine.GetModel()
	fmt.Println("Auto-detected for numeric outputs:", *autoModelJSON)

	// Example of auto-detection with string outputs
	stringOutputs := map[string]interface{}{
		"category": "electronics",
		"color":    "red",
	}

	autoStringEngine := goml.NewAuto(stringOutputs)
	autoStringModelJSON, _ := autoStringEngine.GetModel()
	fmt.Println("Auto-detected for string outputs:", *autoStringModelJSON)

	// Example of auto-detection with binary outputs (integers)
	binaryOutputs := map[string]interface{}{
		"pass":      1,
		"qualified": 0,
	}

	autoBinaryEngine := goml.NewAuto(binaryOutputs)
	autoBinaryModelJSON, _ := autoBinaryEngine.GetModel()
	fmt.Println("Auto-detected for binary outputs (int):", *autoBinaryModelJSON)

	// Example of auto-detection with boolean outputs
	boolOutputs := map[string]interface{}{
		"is_premium": true,
		"in_stock":   false,
	}

	autoBoolEngine := goml.NewAuto(boolOutputs)
	autoBoolModelJSON, _ := autoBoolEngine.GetModel()
	fmt.Println("Auto-detected for boolean outputs:", *autoBoolModelJSON)

	// Demonstrate training with boolean inputs and outputs
	fmt.Println("\nTraining with boolean inputs/outputs:")

	// Example: Predict product availability based on features
	boolTrainInputs := []map[string]interface{}{
		{"is_popular": true, "is_seasonal": false, "has_discount": true},
		{"is_popular": false, "is_seasonal": true, "has_discount": false},
		{"is_popular": true, "is_seasonal": true, "has_discount": false},
		{"is_popular": false, "is_seasonal": false, "has_discount": true},
	}

	boolTrainOutputs := []map[string]interface{}{
		{"in_stock": true},
		{"in_stock": false},
		{"in_stock": true},
		{"in_stock": false},
	}

	// Train with boolean inputs/outputs
	booleanEngine, err := goml.TrainAuto(boolTrainInputs, boolTrainOutputs)
	if err != nil {
		fmt.Printf("Boolean training error: %v\n", err)
	} else {
		// Make prediction with boolean input
		boolPrediction, _ := booleanEngine.Predict(map[string]interface{}{
			"is_popular":   true,
			"is_seasonal":  false,
			"has_discount": false,
		})

		fmt.Println("Boolean prediction:", boolPrediction)
	}

	// Demonstrating mixed output types with auto-detection
	fmt.Println("\n\nMixed Output Types Example")
	fmt.Println("---------------------------")

	// Example with fully mixed output types
	mixedInputs := []map[string]interface{}{
		{"metric1": 1.0, "category": "red", "flag": true},
		{"metric1": 2.0, "category": "blue", "flag": false},
		{"metric1": 3.0, "category": "green", "flag": true},
		{"metric1": 4.0, "category": "yellow", "flag": false},
	}

	// Outputs with mixed types (string, numeric, and boolean)
	mixedOutputs := []map[string]interface{}{
		{"string_output": "small", "numeric_output": 10.5, "boolean_output": true},
		{"string_output": "medium", "numeric_output": 20.5, "boolean_output": false},
		{"string_output": "large", "numeric_output": 30.5, "boolean_output": true},
		{"string_output": "extra", "numeric_output": 40.5, "boolean_output": false},
	}

	// Auto-detect model type for mixed outputs
	mixedEngine, err := goml.TrainAuto(mixedInputs, mixedOutputs)
	if err != nil {
		fmt.Printf("Mixed output training error: %v\n", err)
	} else {
		// Check the model type that was auto-detected
		modelJSON, _ := mixedEngine.GetModel()
		fmt.Println("Auto-detected model type for mixed outputs:", *modelJSON)

		// Make a prediction with mixed outputs
		mixedPrediction, _ := mixedEngine.Predict(map[string]interface{}{
			"metric1":  2.5,
			"category": "red",
			"flag":     true,
		})

		fmt.Println("\nMixed output prediction:")
		mixedPredJSON, _ := json.MarshalIndent(mixedPrediction, "", "  ")
		fmt.Println(string(mixedPredJSON))

		// Explain the result
		fmt.Println("\nThe mixed model handles all output types simultaneously:")
		fmt.Println("- String output handled by categorical model")
		fmt.Println("- Numeric output handled by linear model")
		fmt.Println("- Boolean output handled by logistic model")
	}
}
