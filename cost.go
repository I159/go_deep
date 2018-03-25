package goDeep

import "math"

type cost interface {
	costDerivative(a, e float64) float64
	countCost(a, e []float64) float64
}

type Quadratic struct{}

func (q *Quadratic) countCost(al, er []float64) float64 {
	var sum float64
	for i, out := range al {
		sum += math.Pow((out - er[i]), 2)
	}
	return sum * .5
}

func (q *Quadratic) costDerivative(a, e float64) float64 {
	return a - e
}
