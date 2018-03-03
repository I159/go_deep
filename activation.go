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
	oldMin, oldMax, newMin, newMax float64
}

func (s *Sigmoid) correctFunc(x float64) float64 {
	// Smooth n between 4 and -4
	// NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
	return (((x - s.oldMin) * (s.newMax - s.newMin)) / (s.oldMax - s.oldMin)) + s.newMin
}

func (s *Sigmoid) activate(n float64) (float64, error) {
	x := s.correctFunc(n)
	fmt.Println(n, x)

	exp := math.Exp(x)
	if exp == 0 || math.IsInf(exp, 0) {
		return 0, fmt.Errorf("The activation value is too large: %f", n)
	} else if exp == 1 {
		return 0, fmt.Errorf("The activation value is too small: %f", n)
	} else if math.IsNaN(exp) {
		return 0, fmt.Errorf("Wrong scaling range")
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

func NewSigmoid(prevLayerSize, minInpVal, maxInpVal, minWeight, maxWeight, prevBias float64) activation {
	/*
		Sigmoid operates in range between -4 and 4. Significantly larger values or values
		too close to zero could cause overflow or underflow. Incoming values will be
		linearly scaled to the range. For this purpose sigmoid must know value range for
		incoming data. To avoid manual range computation incoming data and synapses weights
		range are required at initialization of sigmoid.
		If no bias is present in a layer then put 0 as a bias
	*/
	rangeCandidates := [4]float64{
		minInpVal * minWeight,
		minInpVal * maxWeight,
		maxInpVal * maxWeight,
		maxInpVal * minWeight
	}
	// TODO: get possible max and min among weights and input values combinations.
	// Input maximum and minimum could be positive or negative so real maximum or
	// minimum could be reached with any combination.
	sigmoid := Sigmoid{
		oldMin: prevLayerSize*minInpVal*minWeight + prevBias,
		oldMax: prevLayerSize*maxInpVal*maxWeight + prevBias,
		newMin: -4,
		newMax: 4,
	}
	return &sigmoid
}
