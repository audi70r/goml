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
		{"pass": 1},  // will pass
		{"pass": 0},  // will fail
		{"pass": 0},  // will fail
		{"pass": 1},  // will pass
		{"pass": 0},  // will fail
		{"pass": 1},  // will pass
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
		"prev_grade": 75,
	}
	
	logisticPrediction, _ := logisticEngine.Predict(newStudent)
	
	fmt.Println("\nLogistic prediction for new student:")
	logisticPredJSON, _ := json.MarshalIndent(logisticPrediction, "", "  ")
	fmt.Println(string(logisticPredJSON))
}