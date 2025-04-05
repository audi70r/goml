package goml

import (
	"fmt"
)

// NewMixedModel creates a new model that can handle mixed types (numeric, boolean, and categorical)
// This model internally uses separate models for different output types
func NewMixedModel() *Model {
	return &Model{
		Type: "mixed",
		Parameters: map[string]interface{}{
			"bias": true,
		},
		Categories:        make(map[string]map[string]int),
		FeatureCategories: make(map[string]map[string]int),
		Features:          make(map[string]interface{}),
		Targets:           make(map[string]interface{}),
	}
}

// trainMixedModel implements training for a model that can handle both numeric and categorical outputs
func trainMixedModel(inputs []map[string]interface{}, outputs []map[string]interface{}, weights *Weights, config *Config, model *Model) error {
	// First, analyze the outputs to separate by type
	numericOutputs := make([]map[string]interface{}, len(outputs))
	categoricalOutputs := make([]map[string]interface{}, len(outputs))
	booleanOutputs := make([]map[string]interface{}, len(outputs))

	for i := range numericOutputs {
		numericOutputs[i] = make(map[string]interface{})
		categoricalOutputs[i] = make(map[string]interface{})
		booleanOutputs[i] = make(map[string]interface{})
	}

	if len(outputs) == 0 {
		return ErrInvalidOutput
	}

	// Determine which fields are which type
	for key, val := range outputs[0] {
		isNumeric := false
		isBoolean := false
		isCategorical := false

		// Check value type
		switch v := val.(type) {
		case int, int64, int32, float64, float32:
			// Check if it might be binary (0/1)
			switch num := v.(type) {
			case int:
				if num == 0 || num == 1 {
					isBoolean = true
				} else {
					isNumeric = true
				}
			case float64:
				if num == 0.0 || num == 1.0 {
					isBoolean = true
				} else {
					isNumeric = true
				}
			default:
				isNumeric = true
			}
			if isNumeric {
				model.Targets[key] = "numeric"
			} else {
				model.Targets[key] = "boolean"
			}
		case bool:
			isBoolean = true
			model.Targets[key] = "boolean"
		case string:
			isCategorical = true
			model.Targets[key] = "categorical"
		default:
			// Skip unknown types
			continue
		}

		// Copy values to appropriate output maps
		for i, output := range outputs {
			if v, ok := output[key]; ok {
				if isNumeric {
					numericOutputs[i][key] = v
				} else if isBoolean {
					booleanOutputs[i][key] = v
				} else if isCategorical {
					categoricalOutputs[i][key] = v
				}
			}
		}
	}

	// Count different target types
	numericTargetCount := 0
	categoricalTargetCount := 0
	booleanTargetCount := 0

	for _, output := range numericOutputs {
		if len(output) > 0 {
			numericTargetCount = len(output)
			break
		}
	}

	for _, output := range categoricalOutputs {
		if len(output) > 0 {
			categoricalTargetCount = len(output)
			break
		}
	}

	for _, output := range booleanOutputs {
		if len(output) > 0 {
			booleanTargetCount = len(output)
			break
		}
	}

	fmt.Printf("Numeric targets found: %d\n", numericTargetCount)
	fmt.Printf("Categorical targets found: %d\n", categoricalTargetCount)
	fmt.Printf("Boolean targets found: %d\n", booleanTargetCount)

	// Train for numeric outputs if they exist
	if numericTargetCount > 0 {
		fmt.Println("Training numeric model...")
		err := trainLinearModel(inputs, numericOutputs, weights, config)
		if err != nil {
			return fmt.Errorf("error training numeric targets: %w", err)
		}
	}

	// Train for categorical outputs if they exist
	if categoricalTargetCount > 0 {
		fmt.Println("Training categorical model...")
		err := trainCategoricalModel(inputs, categoricalOutputs, weights, config, model)
		if err != nil {
			return fmt.Errorf("error training categorical targets: %w", err)
		}
	}

	// Train for boolean outputs if they exist
	if booleanTargetCount > 0 {
		fmt.Println("Training boolean model...")
		err := trainLogisticModel(inputs, booleanOutputs, weights, config)
		if err != nil {
			return fmt.Errorf("error training boolean targets: %w", err)
		}
	}

	return nil
}

// predictMixedModel performs prediction with a mixed model
func predictMixedModel(input map[string]interface{}, weights *Weights, model *Model) (map[string]interface{}, error) {
	result := make(map[string]interface{})

	// We'll predict with all model types and combine the results based on target type
	
	// First, do linear predictions for numeric outputs
	linearPred, err := predictLinearModel(input, weights)
	if err != nil {
		return nil, fmt.Errorf("error in linear prediction: %w", err)
	}

	// Add numeric predictions to result based on target type
	for k, v := range linearPred {
		if targetType, ok := model.Targets[k]; ok && targetType == "numeric" {
			result[k] = v
		}
	}

	// Do categorical predictions
	catPred, err := predictCategoricalModel(input, weights, model)
	if err != nil {
		return nil, fmt.Errorf("error in categorical prediction: %w", err)
	}

	// Add categorical predictions to result based on target type
	for k, v := range catPred {
		if targetType, ok := model.Targets[k]; ok && targetType == "categorical" {
			result[k] = v
			
			// Also add probability distributions if available
			probKey := k + "_probs"
			if probs, ok := catPred[probKey]; ok {
				result[probKey] = probs
			}
		}
	}

	// Do logistic predictions for boolean outputs
	logPred, err := predictLogisticModel(input, weights)
	if err != nil {
		return nil, fmt.Errorf("error in logistic prediction: %w", err)
	}

	// Add boolean predictions to result based on target type
	for k, v := range logPred {
		if targetType, ok := model.Targets[k]; ok && targetType == "boolean" {
			// Convert probability to boolean if needed
			if prob, ok := v.(float64); ok {
				if prob >= 0.5 {
					result[k] = true
				} else {
					result[k] = false
				}
			} else {
				result[k] = v
			}
		}
	}

	return result, nil
}