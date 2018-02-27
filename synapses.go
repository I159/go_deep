package go_deep

import (
	"math"
	"math/rand"
	"time"
)

const SCALING_BASE = .7

type synapseInitializer interface {
	init() [][]float64
}

type denseSynapses struct {
	prev, curr, next int
	synapses         [][]float64
}

func (s *denseSynapses) randomInit() {
	rand.Seed(time.Now().UTC().UnixNano())
	for i := 0; i < s.curr-1; i++ {
		s.synapses = append(s.synapses, []float64{})
		for j := 0; j < s.next; j++ {
			s.synapses[i] = append(s.synapses[i], rand.Float64()-0.5)
		}
	}
}

func (s *denseSynapses) nguyenWiderow() {
	var norm float64
	beta := SCALING_BASE * math.Pow(float64(s.curr), 1.0/float64(s.prev))

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

func (s *denseSynapses) init() [][]float64 {
	s.randomInit()
	s.nguyenWiderow()
	return s.synapses
}

type hiddenDenseSynapses struct {
	denseSynapses
	bias float64
}

func (s *hiddenDenseSynapses) addBiases() {
	biasSignal := make([]float64, s.next)
	for i := range biasSignal {
		biasSignal[i] = s.bias
	}
	s.synapses = append(s.synapses, biasSignal)
}

func (s hiddenDenseSynapses) init() [][]float64 {
	s.denseSynapses.init()
	s.addBiases()
	return s.synapses
}
