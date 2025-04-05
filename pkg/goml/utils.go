package goml

// ConvertToFloat64 converts different types to float64 for model training and prediction
// Handles numeric types, bool, and strings in a consistent way
func ConvertToFloat64(val interface{}, oneHotKey string) (float64, bool) {
	switch v := val.(type) {
	case float64:
		return v, true
	case float32:
		return float64(v), true
	case int:
		return float64(v), true
	case int64:
		return float64(v), true
	case int32:
		return float64(v), true
	case bool:
		// Convert boolean to numeric: true=1.0, false=0.0
		if v {
			return 1.0, true
		}
		return 0.0, true
	case string:
		// For string features, use one-hot encoding (1.0 if matches)
		if v == oneHotKey {
			return 1.0, true
		}
		return 0.0, true
	default:
		return 0.0, false
	}
}

// IsSupportedNumericType checks if the value is a numeric type (int, float)
// Used for normalizing features and calculating means
func IsSupportedNumericType(val interface{}) bool {
	switch val.(type) {
	case int, int32, int64, float32, float64:
		return true
	default:
		return false
	}
}

// IsSupportedBooleanType checks if the value is a boolean type
func IsSupportedBooleanType(val interface{}) bool {
	_, ok := val.(bool)
	return ok
}

// ConvertToBool attempts to convert a value to boolean
// This is useful when we want to use a model to predict boolean outputs
func ConvertToBool(val interface{}) (bool, bool) {
	switch v := val.(type) {
	case bool:
		return v, true
	case int:
		return v != 0, true
	case float64:
		return v != 0, true
	case string:
		return v == "true" || v == "yes" || v == "1", true
	default:
		return false, false
	}
}
