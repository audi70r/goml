package goml

import (
	"fmt"
	"math"
)

// trainLinearModel implements linear regression training
func trainLinearModel(inputs []map[string]interface{}, outputs []map[string]interface{}, weights *Weights, config *Config) error {
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

	// Debug: Print features and targets
	fmt.Printf("Training with features: %v\n", features)
	fmt.Printf("Training with targets: %v\n", targets)

	// Normalization helpers - calculate means for normalization
	featureMeans := make(map[string]float64)
	targetMeans := make(map[string]float64)

	// Calculate feature means for numeric features
	for _, feature := range features {
		sum := 0.0
		count := 0
		for i := range inputs {
			if val, ok := inputs[i][feature]; ok {
				switch v := val.(type) {
				case float64:
					sum += v
					count++
				case int:
					sum += float64(v)
					count++
				}
			}
		}
		if count > 0 {
			featureMeans[feature] = sum / float64(count)
			fmt.Printf("Feature %s mean: %f\n", feature, featureMeans[feature])
		}
	}

	// Calculate target means
	for _, target := range targets {
		sum := 0.0
		count := 0
		for i := range outputs {
			if val, ok := outputs[i][target]; ok {
				switch v := val.(type) {
				case float64:
					sum += v
					count++
				case int:
					sum += float64(v)
					count++
				}
			}
		}
		if count > 0 {
			targetMeans[target] = sum / float64(count)
			fmt.Printf("Target %s mean: %f\n", target, targetMeans[target])
		}
	}

	// Gradient descent for the specified number of epochs
	for epoch := 0; epoch < config.Epochs; epoch++ {
		// Calculate MSE for convergence check
		prevMSE := calculateMSE(inputs, outputs, weights, features, targets)

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

						// Convert feature value to float64 and normalize
						var featureVal float64
						switch v := featureValRaw.(type) {
						case float64:
							// Normalize by dividing by mean if it's non-zero
							if mean, ok := featureMeans[feature]; ok && mean != 0 {
								featureVal = v / mean
							} else {
								featureVal = v
							}
						case int:
							// Normalize by dividing by mean if it's non-zero
							if mean, ok := featureMeans[feature]; ok && mean != 0 {
								featureVal = float64(v) / mean
							} else {
								featureVal = float64(v)
							}
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

						// Calculate the prediction for this sample
						predicted := 0.0
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

							predicted += w * fValFloat
						}

						// Add bias term
						biasKey := fmt.Sprintf("bias->%s", target)
						if bias, exists := weights.GetFloat(biasKey); exists {
							predicted += bias
						}

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

						// Update gradient: (predicted - actual) * featureValue
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
					// Calculate prediction for this sample
					predicted := 0.0
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

						predicted += w * fValFloat
					}

					// Add bias
					bias, _ := weights.GetFloat(biasKey)
					predicted += bias

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
		currentMSE := calculateMSE(inputs, outputs, weights, features, targets)
		if math.Abs(prevMSE-currentMSE) < config.Tolerance {
			break
		}

		// Print MSE every 1000 epochs
		if epoch%1000 == 0 {
			fmt.Printf("Epoch %d, MSE: %f\n", epoch, currentMSE)
		}
	}

	// Print final weights
	fmt.Println("Final weights:")
	for key, val := range weights.Values {
		fmt.Printf("%s: %v\n", key, val)
	}

	return nil
}

// predictLinearModel implements linear regression prediction
func predictLinearModel(input map[string]interface{}, weights *Weights) (map[string]interface{}, error) {
	result := make(map[string]interface{})

	// Find all the target variables from weight keys
	targets := make(map[string]bool)
	features := make(map[string]bool)

	// Extract features and targets from weights
	for key := range weights.Values {
		parts := splitWeightKey(key)
		if len(parts) == 2 && parts[0] != "bias" {
			features[parts[0]] = true
			targets[parts[1]] = true
		} else if len(parts) == 2 {
			targets[parts[1]] = true
		}
	}

	// Calculate prediction for each target
	for target := range targets {
		prediction := 0.0

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

			prediction += weight * featureVal
		}

		// Add bias term
		biasKey := fmt.Sprintf("bias->%s", target)
		if bias, exists := weights.GetFloat(biasKey); exists {
			prediction += bias
		}

		// Store prediction in result
		result[target] = prediction
	}

	return result, nil
}

// Helper function to calculate mean squared error
func calculateMSE(inputs []map[string]interface{}, outputs []map[string]interface{}, weights *Weights, features []string, targets []string) float64 {
	totalMSE := 0.0
	sampleCount := 0

	for i := range inputs {
		for _, target := range targets {
			// Calculate prediction for this sample
			prediction := 0.0

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

				prediction += weight * featureVal
			}

			// Add bias term
			biasKey := fmt.Sprintf("bias->%s", target)
			if bias, exists := weights.GetFloat(biasKey); exists {
				prediction += bias
			}

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

			// Square error
			error := prediction - actual
			totalMSE += error * error
			sampleCount++
		}
	}

	if sampleCount == 0 {
		return 0.0
	}

	return totalMSE / float64(sampleCount)
}

// Helper to split a weight key into feature and target
func splitWeightKey(key string) []string {
	parts := make([]string, 2)
	for i, c := range key {
		if c == '-' && i+1 < len(key) && key[i+1] == '>' {
			parts[0] = key[:i]
			parts[1] = key[i+2:]
			break
		}
	}
	return parts
}
