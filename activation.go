package go_deep

import (
	"fmt"
	"math"
)

type activation interface {
	activate(float64) (float64, error)
	actDerivative(float64) (float64, error)
}

type Sigmoid struct {}

func (s *Sigmoid) activate(n float64) (float64, error) {
	exp := math.Exp(n)
	if exp == 0 || math.IsInf(exp, 0) {
		return 0, fmt.Errorf("The activation value is too large: %f", n)
	}
	return 1 / (1 + exp), nil
}

func (s *Sigmoid) actDerivative(n float64) (float64, error) {
	actVal, err := s.activate(n)
	if err != nil {
		return 0, err
	}
	return actVal * (1 - actVal), err
}
