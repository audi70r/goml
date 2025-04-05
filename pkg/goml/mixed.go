package goml

import (
	"fmt"
)

// NewMixedModel creates a new model that can handle mixed types (numeric and categorical)
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
	// First, analyze the outputs to separate numeric and categorical targets
	numericOutputs := make([]map[string]interface{}, len(outputs))
	categoricalOutputs := make([]map[string]interface{}, len(outputs))

	for i := range numericOutputs {
		numericOutputs[i] = make(map[string]interface{})
		categoricalOutputs[i] = make(map[string]interface{})
	}

	if len(outputs) == 0 {
		return ErrInvalidOutput
	}

	// Determine which fields are numeric and which are categorical
	for key, val := range outputs[0] {
		isNumeric := false

		// Check if the value is numeric
		switch val.(type) {
		case int, int64, int32, float64, float32:
			isNumeric = true
			model.Targets[key] = "numeric"
		case string:
			isNumeric = false
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
				} else {
					categoricalOutputs[i][key] = v
				}
			}
		}
	}

	// Count numeric and categorical targets
	numericTargetCount := 0
	categoricalTargetCount := 0

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

	fmt.Printf("Numeric targets found: %d\n", numericTargetCount)
	fmt.Printf("Categorical targets found: %d\n", categoricalTargetCount)

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

	return nil
}

// predictMixedModel performs prediction with a mixed model
func predictMixedModel(input map[string]interface{}, weights *Weights, model *Model) (map[string]interface{}, error) {
	result := make(map[string]interface{})

	// We'll predict with both linear and categorical models and combine the results
	// First, do linear predictions for numeric outputs
	linearPred, err := predictLinearModel(input, weights)
	if err != nil {
		return nil, fmt.Errorf("error in linear prediction: %w", err)
	}

	// Then do categorical predictions
	catPred, err := predictCategoricalModel(input, weights, model)
	if err != nil {
		return nil, fmt.Errorf("error in categorical prediction: %w", err)
	}

	// Combine predictions
	for k, v := range linearPred {
		result[k] = v
	}

	for k, v := range catPred {
		result[k] = v
	}

	return result, nil
}
