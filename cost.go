package go_deep

import "math"

type cost interface {
	costDerivative(a, e float64) float64
	countCost(a, e []float64) float64
}

type quadratic struct{}

func (q *quadratic) countCost(al, er []float64) float64 {
	var sum float64
	for i, out := range al {
		sum += math.Pow((out - er[i]), 2)
	}
	return sum * .5
}

func (q *quadratic) costDerivative(a, e float64) float64 {
	return a - e
}
