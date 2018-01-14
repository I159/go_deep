package main

import (
	"math"
	"math/rand"
	"time"
)

type synapsesOps interface {
	intializeSynapses(hidden, input, output int) [][]float64
	// TODO: correct synapses using batches
}

type denseSynapses struct {
	synapses [][]float64
}

func (s *denseSynapses) randomInit()  {
	rand.Seed(time.Now().UTC().UnixNano())
	for i := 0; i < HIDDEN; i++ {
		out = append(s.synapses, []float64{})
		for j := 0; j < OUTPUT; j++ {
			s.synapses[i] = append(s.synapses[i], rand.Float64()-0.5)
		}
	}
}

func (s *denseSynapses) nguyenWiderow() {
	s.randomInit()

	var norm float64
	beta := SCALING_BASE * math.Pow(HIDDEN, 1.0/INPUT)

	for _, i := range s.synapses {
		norm = 0
		for _, j := range i {
			norm += j * j
		}
		norm = math.Sqrt(norm)
		for j, k := range i {
			i[j] = (k * beta) / norm
		}
	}
}

func addBiases(synapses [][]float64) [][]float64 {
	synapses = append(synapses, make([]float64, OUTPUT))
	for i := 0; i < OUTPUT; i++ {
		synapses[HIDDEN][i] = BIAS
	}
	return synapses
}

// TODO: Implement initialization method
