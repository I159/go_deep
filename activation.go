package go_deep

import "math"

type activation interface {
	activate(float64) float64
	actDerivative(float64) float64
}

type sygmoid struct{}

func (s *sygmoid) activate(n float64) float64 {
	return 1 / (1 + math.Exp(n))
}

func (s *sygmoid) actDerivative(n float64) float64 {
	actVal := s.activate(n)
	return actVal * (1 - actVal)
}
