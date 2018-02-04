package main

type backwardPropagation interface {
	forward()
	backward()
}

// TODO: decompose the interface
type network interface {
	activation
	cost
	backwardPropagation
	Learn(set, labels [][]float64) []float64
	Recognize([][]float64) [][]float64
}
