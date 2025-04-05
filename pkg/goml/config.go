package goml

// Config includes training configuration parameters
type Config struct {
	LearningRate float64 `json:"learning_rate"`
	Epochs       int     `json:"epochs"`
	BatchSize    int     `json:"batch_size"`
	Regularize   float64 `json:"regularize"` // L2 regularization parameter
	Tolerance    float64 `json:"tolerance"`  // Convergence tolerance
}

// DefaultConfig returns default training configuration
func DefaultConfig() *Config {
	return &Config{
		LearningRate: 0.01,
		Epochs:       100,
		BatchSize:    32,
		Regularize:   0.0001,
		Tolerance:    0.0001,
	}
}
