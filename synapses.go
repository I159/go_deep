package main

import (
	"math"
	"math/rand"
	"time"
)

const (
	BIAS = .5
	SCALING_BASE = .7
)

func randomInit(hidden, outout int) (synapses [][]float64) {
	rand.Seed(time.Now().UTC().UnixNano())
	for i := 0; i < hidden; i++ {
		out = append(synapses, []float64{})
		for j := 0; j < output; j++ {
			synapses[i] = append(synapses[i], rand.Float64()-0.5)
		}
	}
	return
}

func nguyenWiderow(synapses [][]float64, hidden, input int) [][]float64 {
	s.randomInit()

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

func newSynapses(hidden, input, output) {
	synapses := nguyenWiderow()
	addBiases(synapses)
	return synapses
}
