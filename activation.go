package go_deep

import (
	"fmt"
	"math"
)

type activation interface {
	activate(float64) (float64, error)
	actDerivative(float64) (float64, error)
}

type Sigmoid struct {
	incRange  *[2]float64
	goalRange *[2]float64
}

func (s *Sigmoid) correctFunc(x float64) (float64, error) {
	// Smooth n between 4 and -4
	if s.incRange == nil || s.goalRange == nil {
		return 0, fmt.Errorf("Uninitialized scaling ranges")
	}
	pt1 := (x - s.incRange[0]) / (s.incRange[1] - s.incRange[0])
	return s.goalRange[0]*(1-pt1) + s.goalRange[1]*pt1, nil
}

func (s *Sigmoid) activate(n float64) (float64, error) {
	x, err := s.correctFunc(n)
	if err != nil {
		return 0, err
	}

	exp := math.Exp(x)
	if exp == 0 || math.IsInf(exp, 0) {
		return 0, fmt.Errorf("The activation value is too large: %f", n)
	} else if exp == 1 {
		return 0, fmt.Errorf("The activation value is too small: %f", n)
	} else if math.IsNaN(exp) {
		return 0, fmt.Errorf(
			"Invalid scaling ranges. \nIncoming data range: from %#v\nGoal range: %#v\n",
			s.incRange, s.goalRange,
		)
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

func NewSigmoid(inputSize float64, dataValueRange, weightsRange [2]float64, bias float64) activation {
	/*
	Sigmoid operates in range between -4 and 4. Significantly larger values or values
	too close to zero could cause overflow or underflow. Incoming values will be 
	linearly scaled to the range. For this purpose sigmoid must know value range for
	incoming data. To avoid manual range computation incoming data and synapses weights
	range are required at initialization of sigmoid.
	If no bias is present in a layer then put 0 as a bias
	*/
	s := new(Sigmoid)
	s.incRange = &[2]float64{
		inputSize*dataValueRange[0]*weightsRange[0] + bias,
		inputSize*dataValueRange[1]*weightsRange[1] + bias,
	}
	s.goalRange = &[2]float64{-4, 4}
	return s
}
