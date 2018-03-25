package goDeep

import (
	"fmt"
	"math"
)

type activation interface {
	activate(float64) (float64, error)
	actDerivative(float64) (float64, error)
}

/*
Sigmoid activation function defines the output of a node given an input or set of inputs.

A sigmoid function is a mathematical function having a characteristic "S"-shaped curve or sigmoid curve.
Often, sigmoid function refers to the special case of the logistic function shown in the first figure and defined by the formula:

     1
 --------
 1 + e^-x

*/
type Sigmoid struct{}

func (s *Sigmoid) activate(x float64) (float64, error) {
	exp := math.Exp(-x)
	if exp == 0 || math.IsInf(exp, 0) {
		return 0, fmt.Errorf("The activation value is too large: %f", x)
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
