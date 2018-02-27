package go_deep

import "math"

type activation interface {
	activate(float64) float64
	actDerivative(float64) float64
}

type Sygmoid struct{}

func (s *Sygmoid) activate(n float64) float64 {
	return 1 / (1 + math.Exp(n))
}

func (s *Sygmoid) actDerivative(n float64) float64 {
	actVal := s.activate(n)
	return actVal * (1 - actVal)
}
