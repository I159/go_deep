package go_deep

type backwardPropagation interface {
	forward(set []float64, keepHidden bool) (output []float64, hiddenOut [][]float64)
	backward(out, labels []float64, hiddenOut [][]float64)
}

// TODO: decompose the interface
type network interface {
	activation
	cost
	backwardPropagation
	Learn(set, labels [][]float64) []float64
	Recognize([][]float64) ([][]float64)
	Measure(set, labels [][]float64) (float64, []float64)
}
