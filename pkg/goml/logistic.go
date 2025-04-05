package goml

import (
	"fmt"
	"math"
)

// Sigmoid function for logistic regression
func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

// trainLogisticModel implements logistic regression training
func trainLogisticModel(inputs []map[string]interface{}, outputs []map[string]interface{}, weights *Weights, config *Config) error {
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

	// Initialize weights if they don't exist
	for _, feature := range features {
		for _, target := range targets {
			weightKey := fmt.Sprintf("%s->%s", feature, target)
			if _, exists := weights.Get(weightKey); !exists {
				weights.Set(weightKey, 0.0)
			}
		}
	}

	// Add bias term if needed
	for _, target := range targets {
		biasKey := fmt.Sprintf("bias->%s", target)
		if _, exists := weights.Get(biasKey); !exists {
			weights.Set(biasKey, 0.0)
		}
	}

	// Gradient descent for the specified number of epochs
	for epoch := 0; epoch < config.Epochs; epoch++ {
		// Calculate log loss for convergence check
		prevLoss := calculateLogLoss(inputs, outputs, weights, features, targets)

		// Update weights using batched gradient descent
		for batchStart := 0; batchStart < len(inputs); batchStart += config.BatchSize {
			batchEnd := batchStart + config.BatchSize
			if batchEnd > len(inputs) {
				batchEnd = len(inputs)
			}

			// Process each target variable
			for _, target := range targets {
				// Process each feature
				for _, feature := range features {
					weightKey := fmt.Sprintf("%s->%s", feature, target)
					gradient := 0.0

					// Calculate gradient for this batch
					for i := batchStart; i < batchEnd; i++ {
						// Get input feature value
						featureValRaw, ok := inputs[i][feature]
						if !ok {
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
							// For string features, use one-hot encoding (1.0 if matches)
							if v == feature {
								featureVal = 1.0
							} else {
								featureVal = 0.0
							}
						default:
							continue
						}

						// Calculate the z value (linear combination)
						z := 0.0
						for _, f := range features {
							fKey := fmt.Sprintf("%s->%s", f, target)
							w, exists := weights.GetFloat(fKey)
							if !exists {
								continue
							}

							fVal, ok := inputs[i][f]
							if !ok {
								continue
							}

							// Convert feature value
							var fValFloat float64
							switch v := fVal.(type) {
							case float64:
								fValFloat = v
							case int:
								fValFloat = float64(v)
							case string:
								if v == f {
									fValFloat = 1.0
								} else {
									fValFloat = 0.0
								}
							default:
								continue
							}

							z += w * fValFloat
						}

						// Add bias term
						biasKey := fmt.Sprintf("bias->%s", target)
						if bias, exists := weights.GetFloat(biasKey); exists {
							z += bias
						}

						// Apply sigmoid function
						predicted := sigmoid(z)

						// Get actual target value
						actualRaw, ok := outputs[i][target]
						if !ok {
							continue
						}

						// Convert target value to float64
						var actual float64
						switch v := actualRaw.(type) {
						case float64:
							actual = v
						case int:
							actual = float64(v)
						default:
							continue
						}

						// Gradient for logistic regression: (predicted - actual) * featureValue
						error := predicted - actual
						gradient += error * featureVal
					}

					// Average the gradient over the batch
					gradient /= float64(batchEnd - batchStart)

					// Update weight with learning rate and regularization
					currentWeight, _ := weights.GetFloat(weightKey)
					regularizationTerm := config.Regularize * currentWeight
					newWeight := currentWeight - config.LearningRate*(gradient+regularizationTerm)
					weights.Set(weightKey, newWeight)
				}

				// Update bias term (no regularization for bias)
				biasKey := fmt.Sprintf("bias->%s", target)
				biasGradient := 0.0

				// Calculate bias gradient
				for i := batchStart; i < batchEnd; i++ {
					// Calculate z value for this sample
					z := 0.0
					for _, f := range features {
						weightKey := fmt.Sprintf("%s->%s", f, target)
						w, exists := weights.GetFloat(weightKey)
						if !exists {
							continue
						}

						fVal, ok := inputs[i][f]
						if !ok {
							continue
						}

						// Convert feature value
						var fValFloat float64
						switch v := fVal.(type) {
						case float64:
							fValFloat = v
						case int:
							fValFloat = float64(v)
						case string:
							if v == f {
								fValFloat = 1.0
							} else {
								fValFloat = 0.0
							}
						default:
							continue
						}

						z += w * fValFloat
					}

					// Add bias
					bias, _ := weights.GetFloat(biasKey)
					z += bias

					// Apply sigmoid
					predicted := sigmoid(z)

					// Get actual target value
					actualRaw, ok := outputs[i][target]
					if !ok {
						continue
					}

					// Convert target to float64
					var actual float64
					switch v := actualRaw.(type) {
					case float64:
						actual = v
					case int:
						actual = float64(v)
					default:
						continue
					}

					// Update bias gradient with error (predicted - actual)
					biasGradient += predicted - actual
				}

				// Average the gradient and update bias
				biasGradient /= float64(batchEnd - batchStart)
				currentBias, _ := weights.GetFloat(biasKey)
				newBias := currentBias - config.LearningRate*biasGradient
				weights.Set(biasKey, newBias)
			}
		}

		// Check for convergence
		currentLoss := calculateLogLoss(inputs, outputs, weights, features, targets)
		if math.Abs(prevLoss-currentLoss) < config.Tolerance {
			break
		}
	}

	return nil
}

// predictLogisticModel implements logistic regression prediction
func predictLogisticModel(input map[string]interface{}, weights *Weights) (map[string]interface{}, error) {
	result := make(map[string]interface{})

	// Find all the target variables from weight keys
	targets := make(map[string]bool)
	for key := range weights.Values {
		parts := splitWeightKey(key)
		if len(parts) == 2 {
			targets[parts[1]] = true
		}
	}

	// Calculate prediction for each target
	for target := range targets {
		// Calculate z value (linear combination)
		z := 0.0

		// Add contribution from each feature
		for feature, featureValRaw := range input {
			weightKey := fmt.Sprintf("%s->%s", feature, target)
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

			z += weight * featureVal
		}

		// Add bias term
		biasKey := fmt.Sprintf("bias->%s", target)
		if bias, exists := weights.GetFloat(biasKey); exists {
			z += bias
		}

		// Apply sigmoid function
		prediction := sigmoid(z)

		// Store prediction in result
		result[target] = prediction
	}

	return result, nil
}

// Helper function to calculate log loss for logistic regression
func calculateLogLoss(inputs []map[string]interface{}, outputs []map[string]interface{}, weights *Weights, features []string, targets []string) float64 {
	totalLoss := 0.0
	sampleCount := 0

	for i := range inputs {
		for _, target := range targets {
			// Calculate z value for this sample
			z := 0.0

			// Add contribution from each feature
			for _, feature := range features {
				weightKey := fmt.Sprintf("%s->%s", feature, target)
				weight, exists := weights.GetFloat(weightKey)
				if !exists {
					continue
				}

				featureValRaw, ok := inputs[i][feature]
				if !ok {
					continue
				}

				// Convert feature value
				var featureVal float64
				switch v := featureValRaw.(type) {
				case float64:
					featureVal = v
				case int:
					featureVal = float64(v)
				case string:
					if v == feature {
						featureVal = 1.0
					} else {
						featureVal = 0.0
					}
				default:
					continue
				}

				z += weight * featureVal
			}

			// Add bias term
			biasKey := fmt.Sprintf("bias->%s", target)
			if bias, exists := weights.GetFloat(biasKey); exists {
				z += bias
			}

			// Apply sigmoid to get prediction
			prediction := sigmoid(z)

			// Get actual value
			actualRaw, ok := outputs[i][target]
			if !ok {
				continue
			}

			// Convert actual to float64
			var actual float64
			switch v := actualRaw.(type) {
			case float64:
				actual = v
			case int:
				actual = float64(v)
			default:
				continue
			}

			// Calculate log loss: -[y*log(p) + (1-y)*log(1-p)]
			// Use clipping to avoid log(0)
			clippedPred := math.Max(math.Min(prediction, 0.9999), 0.0001)
			loss := -(actual*math.Log(clippedPred) + (1-actual)*math.Log(1-clippedPred))
			totalLoss += loss
			sampleCount++
		}
	}

	if sampleCount == 0 {
		return 0.0
	}

	return totalLoss / float64(sampleCount)
}
