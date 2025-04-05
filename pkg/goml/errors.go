package goml

import "errors"

// Common errors
var (
	ErrUnsupportedModelType = errors.New("unsupported model type")
	ErrInvalidInput         = errors.New("invalid input data")
	ErrInvalidOutput        = errors.New("invalid output data")
	ErrModelNotTrained      = errors.New("model not trained")
)
