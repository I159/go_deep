package main

type backwardPropagation interface {
	forward()
	backward()
}

// TODO: decompose the interface
type network interface {
	synapsesOps
	activation
	cost
	backwardPropagation
	Learn(set, labels, [][]float64)
	Recognize([][]float64)
}
