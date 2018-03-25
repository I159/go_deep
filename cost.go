package goDeep

import "math"

type cost interface {
	costDerivative(a, e float64) float64
	countCost(a, e []float64) float64
}

/*
Quadratic cost function

A cost function is a measure of "how good" a neural network did with respect to it's given training sample and the
expected output. It also may depend on variables such as weights and biases. A cost function is a single value, not
a vector, because it rates how good the neural network did as a whole.

Defined as 0.5∑j(aLj−Erj)2
The gradient of this cost function with respect to the output of a neural network
and some sample r is (aL−Er)

*/
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
