package go_deep

type backwardPropagation interface {
	forward(set []float64) (output []float64)
	forwardMeasure(set, labels []float64) (prediction []float64, cost float64)
	backward(prediction, labels []float64)
}

type network interface {
	backwardPropagation
	Learn(set, labels [][]float64, epochs int, batchSize int) []float64
	Recognize([][]float64) ([][]float64)
}
