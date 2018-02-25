package go_deep

import (
	"math"
	"math/rand"
	"time"
)

const SCALING_BASE = .7

// TODO: create a new type derived from slice of floats.
type synapseInitializer interface {
	init(prev, curr, next int, bias float64)
}

type denseSynapses [][]float64

func (s *denseSynapses) randomInit(curr, next int) {
	rand.Seed(time.Now().UTC().UnixNano())
	for i := 0; i < curr-1; i++ {
		s = append(s, []float64{})
		for j := 0; j < next; j++ {
			s[i] = append(s[i], rand.Float64()-0.5)
		}
	}
}

func (s *denseSynapses) nguyenWiderow(prev, curr int) {
	var norm float64
	beta := SCALING_BASE * math.Pow(float64(curr), 1.0/float64(prev))

	for _, i := range s {
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

func (s *denseSynapses) addBiases(next int, bias float64) {
	biasSignal := make([]float64, next)
	for i := range biasSignal {
		biasSignal[i] = bias
	}
	s = append(s, biasSignal)
}

func (s denseSynapses) init(prev, curr, next int, bias float64) {
	&s.randomInit(curr, next)
	&s.nguyenWiderow(prev, curr)
	&s.addBiases(next, bias)
	return [][]float64(s)
}
