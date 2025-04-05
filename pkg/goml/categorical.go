package goml

import (
	"fmt"
	"math"
	"strings"
)

// trainCategoricalModel implements categorical classification training
func trainCategoricalModel(inputs []map[string]interface{}, outputs []map[string]interface{}, weights *Weights, config *Config, model *Model) error {
	// Get feature names from the first input
	if len(inputs) == 0 {
		return ErrInvalidInput
	}

	// Extract feature names from first input
	features := make([]string, 0, len(inputs[0]))
	for key := range inputs[0] {
		features = append(features, key)
	}

	// Extract target variable names from first output
	if len(outputs) == 0 {
		return ErrInvalidOutput
	}

	targets := make([]string, 0, len(outputs[0]))
	for key := range outputs[0] {
		targets = append(targets, key)
	}

	// Initialize or clear the categories map if needed
	if model.Categories == nil {
		model.Categories = make(map[string]map[string]int)
	}

	// Identify all possible values for each output target
	for _, target := range targets {
		// Initialize category map for this target if it doesn't exist
		if _, exists := model.Categories[target]; !exists {
			model.Categories[target] = make(map[string]int)
		}

		// Count occurrences of each category
		categoryCount := make(map[string]int)
		for _, out := range outputs {
			if val, ok := out[target]; ok {
				category := fmt.Sprintf("%v", val)
				categoryCount[category]++
			}
		}

		// Assign indices to categories
		idx := 0
		for category := range categoryCount {
			if _, exists := model.Categories[target][category]; !exists {
				model.Categories[target][category] = idx
				idx++
			}
		}
	}

	// For each target (output variable), we train a separate set of weights
	for _, target := range targets {
		categories := model.Categories[target]
		numCategories := len(categories)

		if numCategories <= 1 {
			// Trivial case, only one category
			continue
		}

		// For each category, we create a set of weights
		for category, _ := range categories {
			// For each feature, we need a weight
			for _, feature := range features {
				weightKey := fmt.Sprintf("%s->%s:%s", feature, target, category)
				if _, exists := weights.Get(weightKey); !exists {
					weights.Set(weightKey, 0.0)
				}
			}

			// Add bias term
			biasKey := fmt.Sprintf("bias->%s:%s", target, category)
			if _, exists := weights.Get(biasKey); !exists {
				weights.Set(biasKey, 0.0)
			}
		}

		// We use a softmax approach for multi-class classification
		// Similar to logistic regression but with multiple outputs
		for epoch := 0; epoch < config.Epochs; epoch++ {
			// Use stochastic gradient descent
			for i := range inputs {
				// First calculate scores for each category
				categoryScores := make(map[string]float64)

				for category := range categories {
					score := 0.0

					// Compute weighted sum for this category
					for _, feature := range features {
						weightKey := fmt.Sprintf("%s->%s:%s", feature, target, category)
						featureWeight, _ := weights.GetFloat(weightKey)

						featureVal, ok := inputs[i][feature]
						if !ok {
							continue
						}

						// Convert feature value
						var featureValFloat float64
						switch v := featureVal.(type) {
						case float64:
							featureValFloat = v
						case int:
							featureValFloat = float64(v)
						case string:
							// One-hot encoding for string features
							if v == feature {
								featureValFloat = 1.0
							} else {
								featureValFloat = 0.0
							}
						default:
							continue
						}

						score += featureWeight * featureValFloat
					}

					// Add bias
					biasKey := fmt.Sprintf("bias->%s:%s", target, category)
					bias, _ := weights.GetFloat(biasKey)
					score += bias

					categoryScores[category] = score
				}

				// Apply softmax to get probabilities
				probabilities := softmax(categoryScores)

				// Get actual output category
				actualValue, ok := outputs[i][target]
				if !ok {
					continue
				}

				actualCategory := fmt.Sprintf("%v", actualValue)

				// Update weights using the difference between predicted and actual
				for category := range categories {
					// Target probability (1 for the true category, 0 for others)
					targetProbability := 0.0
					if category == actualCategory {
						targetProbability = 1.0
					}

					// Calculate gradient
					gradient := probabilities[category] - targetProbability

					// Update weights for this category
					for _, feature := range features {
						weightKey := fmt.Sprintf("%s->%s:%s", feature, target, category)
						currentWeight, _ := weights.GetFloat(weightKey)

						featureVal, ok := inputs[i][feature]
						if !ok {
							continue
						}

						// Convert feature value
						var featureValFloat float64
						switch v := featureVal.(type) {
						case float64:
							featureValFloat = v
						case int:
							featureValFloat = float64(v)
						case string:
							if v == feature {
								featureValFloat = 1.0
							} else {
								featureValFloat = 0.0
							}
						default:
							continue
						}

						// Apply regularization
						regularizationTerm := config.Regularize * currentWeight

						// Update weight
						newWeight := currentWeight - config.LearningRate*(gradient*featureValFloat+regularizationTerm)
						weights.Set(weightKey, newWeight)
					}

					// Update bias term (no regularization for bias)
					biasKey := fmt.Sprintf("bias->%s:%s", target, category)
					currentBias, _ := weights.GetFloat(biasKey)
					newBias := currentBias - config.LearningRate*gradient
					weights.Set(biasKey, newBias)
				}
			}
		}
	}

	return nil
}

// predictCategoricalModel implements categorical classification prediction
func predictCategoricalModel(input map[string]interface{}, weights *Weights, model *Model) (map[string]interface{}, error) {
	result := make(map[string]interface{})

	// For each target variable, predict the category
	for target, categories := range model.Categories {
		if len(categories) <= 0 {
			continue
		}

		// Calculate scores for each category
		categoryScores := make(map[string]float64)

		for category := range categories {
			score := 0.0

			// Compute weighted sum for this category
			for feature, featureValRaw := range input {
				weightKey := fmt.Sprintf("%s->%s:%s", feature, target, category)
				weight, exists := weights.GetFloat(weightKey)
				if !exists {
					continue
				}

				// Convert feature value to float64
				var featureVal float64
				switch v := featureValRaw.(type) {
				case float64:
					featureVal = v
				case int:
					featureVal = float64(v)
				case string:
					// One-hot encoding for string features
					if v == feature {
						featureVal = 1.0
					} else {
						featureVal = 0.0
					}
				default:
					continue
				}

				score += weight * featureVal
			}

			// Add bias term
			biasKey := fmt.Sprintf("bias->%s:%s", target, category)
			if bias, exists := weights.GetFloat(biasKey); exists {
				score += bias
			}

			categoryScores[category] = score
		}

		// Apply softmax to get probabilities
		probabilities := softmax(categoryScores)

		// Find the category with the highest probability
		var bestCategory string
		var bestProb float64

		for category, prob := range probabilities {
			if prob > bestProb {
				bestProb = prob
				bestCategory = category
			}
		}

		// Store both the predicted category and the probabilities
		if bestCategory != "" {
			// Convert back to the original type if possible
			// Try conversion to number if the category looks like a number
			if isNumeric(bestCategory) {
				if strings.Contains(bestCategory, ".") {
					// Try as float64
					if val, err := stringToFloat64(bestCategory); err == nil {
						result[target] = val
					} else {
						result[target] = bestCategory
					}
				} else {
					// Try as int
					if val, err := stringToInt(bestCategory); err == nil {
						result[target] = val
					} else {
						result[target] = bestCategory
					}
				}
			} else {
				// Use as string
				result[target] = bestCategory
			}

			// Store probabilities in a nested map
			probsMap := make(map[string]float64)
			for cat, prob := range probabilities {
				probsMap[cat] = prob
			}
			result[target+"_probs"] = probsMap
		}
	}

	return result, nil
}

// softmax computes the softmax of a set of scores
func softmax(scores map[string]float64) map[string]float64 {
	// Find the maximum score to avoid overflow
	var maxScore float64 = -math.MaxFloat64
	for _, score := range scores {
		if score > maxScore {
			maxScore = score
		}
	}

	// Compute exp(score - maxScore) for each category
	expScores := make(map[string]float64)
	var sumExp float64

	for category, score := range scores {
		expScore := math.Exp(score - maxScore)
		expScores[category] = expScore
		sumExp += expScore
	}

	// Normalize to get probabilities
	probabilities := make(map[string]float64)
	for category, expScore := range expScores {
		probabilities[category] = expScore / sumExp
	}

	return probabilities
}

// Helper functions for type conversion
func isNumeric(s string) bool {
	// Check if the string represents a number
	_, err1 := stringToFloat64(s)
	_, err2 := stringToInt(s)
	return err1 == nil || err2 == nil
}

func stringToFloat64(s string) (float64, error) {
	var v float64
	_, err := fmt.Sscanf(s, "%f", &v)
	return v, err
}

func stringToInt(s string) (int, error) {
	var v int
	_, err := fmt.Sscanf(s, "%d", &v)
	return v, err
}
