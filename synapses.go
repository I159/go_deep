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
	bias             float64
	nextBias         bool
}

func (s *denseSynapses) randomInit() {
	rand.Seed(time.Now().UTC().UnixNano())
	for i := 0; i < s.curr; i++ {
		s.synapses = append(s.synapses, []float64{})
		for j := 0; j < s.next; j++ {
			s.synapses[i] = append(s.synapses[i], rand.Float64()-0.5)
		}
	}
}

func (s *denseSynapses) addBiases() {
	if s.bias != 0 {
		biasSignal := make([]float64, s.next)
		for i := range biasSignal {
			biasSignal[i] = s.bias
		}
		s.synapses = append(s.synapses, biasSignal)
	}
}

func (s *denseSynapses) init() [][]float64 {
	s.randomInit()
	s.addBiases()
	return s.synapses
}

type hiddenDenseSynapses struct {
	denseSynapses
}

func (s *hiddenDenseSynapses) nguyenWiderow() {
	var norm float64
	beta := SCALING_BASE * math.Pow(float64(s.curr), 1.0/float64(s.prev))

	for i := 0; i < s.curr; i++ {
		norm = 0
		for j := 0; j < s.next; j++ {
			norm += math.Pow(s.synapses[i][j], 2.)
		}
		norm = math.Sqrt(norm)
		for j := 0; j < s.curr; j++ {
			s.synapses[i][j] = (s.synapses[i][j] * beta) / norm
		}
	}
}

func (s hiddenDenseSynapses) init() [][]float64 {
	s.denseSynapses.init()
	s.addBiases()
	return s.synapses
}
