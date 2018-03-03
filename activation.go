package go_deep

import (
	"fmt"
	"math"
)

type activation interface {
	activate(float64) (float64, error)
	actDerivative(float64) (float64, error)
}

type Sygmoid struct{}

func (s *Sygmoid) activate(n float64) (float64, error) {
	// Smooth n between 4 and -4
	exp := math.Exp(n)

	if exp == 0 || math.IsInf(exp, 0) {
		return 0, fmt.Errorf("The activation value is too large: %f", n)
	} else if exp == 1 {
		return 0, fmt.Errorf("The activation value is too small: %f", n)
	}

	return 1 / (1 + exp), nil
}

func (s *Sygmoid) actDerivative(n float64) (float64 , error){
	actVal, err := s.activate(n)
	if err != nil {
		return 0, err
	}
	return actVal * (1 - actVal), err
}
