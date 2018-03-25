package goDeep

type backwardPropagation interface {
	forward(set []float64) (output []float64, err error)
	forwardMeasure(set, labels []float64) (prediction []float64, cost float64, err error)
	backward(prediction, labels []float64) error
	applyCorrections(float64) error
}

/*
Network is a public interface for actual library usage.

Network interface defines abstract neural network with back propagation.
*/
type Network interface {
	backwardPropagation
	Learn(set, labels [][]float64, epochs int, batchSize int) ([]float64, error)
	Recognize([][]float64) ([][]float64, error)
}
