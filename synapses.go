package go_deep

import (
	"math"
	"math/rand"
	"time"
)

const (
	BIAS = 1
	SCALING_BASE = .7
)

func randomInit(hidden, output int) (synapses [][]float64) {
	rand.Seed(time.Now().UTC().UnixNano())
	for i := 0; i < hidden; i++ {
		synapses = append(synapses, []float64{})
		for j := 0; j < output; j++ {
			synapses[i] = append(synapses[i], rand.Float64()-0.5)
		}
	}
	return
}

func nguyenWiderow(hidden, input, output float64) [][]float64 {
	synapses := randomInit(int(hidden), int(output))

	var norm float64
	beta := SCALING_BASE * math.Pow(hidden, 1.0/input)

	for _, i := range synapses {
		norm = 0
		for _, j := range i {
			norm += j * j
		}
		norm = math.Sqrt(norm)
		for j, k := range i {
			i[j] = (k * beta) / norm
		}
	}
	return synapses
}

func addBiases(synapses [][]float64) {
	nextLayerSize := len(synapses[0])
	currentLayerSize := len(synapses)

	synapses = append(synapses, make([]float64, nextLayerSize))
	for i := 0; i < nextLayerSize; i++ {
		synapses[currentLayerSize][i] = BIAS
	}
}

func newDenseSynapses (hidden, input, output float64) [][]float64 {
	synapses := nguyenWiderow(hidden, input, output)
	addBiases(synapses)
	return synapses
}
