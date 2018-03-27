package goDeep

import (
	"math"
	"math/rand"
	"time"
)

const scalingBase = .7

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
	curr := s.curr
	if s.bias != 0 {
		curr--
	}
	next := s.next
	if s.nextBias {
		next--
	}

	rand.Seed(time.Now().UTC().UnixNano())
	for i := 0; i < next; i++ {
		s.synapses = append(s.synapses, []float64{})
		for j := 0; j < curr; j++ {
			s.synapses[i] = append(s.synapses[i], rand.Float64()-0.5)
		}
	}
}

func (s *denseSynapses) addBiases() {
	if s.bias != 0 {
		for i := range s.synapses {
			s.synapses[i] = append(s.synapses, s.bias)
		}
	}
}

func (s *denseSynapses) init() [][]float64 {
	s.randomInit()
	if s.bias != 0 {
		s.addBiases()
	}
	return s.synapses
}

type hiddenDenseSynapses struct {
	denseSynapses
}

func (s *hiddenDenseSynapses) nguyenWiderow() {
	curr := s.curr
	if s.bias != 0 {
		curr--
	}
	next := s.next
	if s.nextBias {
		next--
	}

	var norm float64
	beta := scalingBase * math.Pow(float64(s.curr), 1.0/float64(s.prev))

	for i := 0; i < next; i++ {
		norm = 0
		for j := 0; j < next; j++ {
			norm += math.Pow(s.synapses[i][j], 2.)
		}
		norm = math.Sqrt(norm)
		for j := 0; j < curr; j++ {
			s.synapses[i][j] = (s.synapses[i][j] * beta) / norm
		}
	}
}

func (s hiddenDenseSynapses) init() [][]float64 {
	s.denseSynapses.init()
	s.nguyenWiderow()
	return s.synapses
}
