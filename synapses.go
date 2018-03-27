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
}

func (s *denseSynapses) randomInit() {
	rand.Seed(time.Now().UTC().UnixNano())
	for i := 0; i < s.next; i++ {
		s.synapses = append(s.synapses, []float64{})
		for j := 0; j < s.curr; j++ {
			s.synapses[i] = append(s.synapses[i], rand.Float64()-0.5)
		}
	}
}

func (s *denseSynapses) init() [][]float64 {
	s.randomInit()
	return s.synapses
}

type hiddenDenseSynapses struct {
	denseSynapses
}

func (s *denseSynapses) norm(synapse float64) (n float64) {
	for j := 0; j < s.next; j++ {
		n += math.Pow(synapse, 2.)
	}
	n = math.Sqrt(n)
	return
}

func (s *hiddenDenseSynapses) nguyenWiderow() {
	beta := scalingBase * math.Pow(float64(s.curr), 1.0/float64(s.prev))

	for i := 0; i < s.next; i++ {
		for j := 0; j < s.curr; j++ {
			s.synapses[i][j] = (s.synapses[i][j] * beta) / s.norm(s.synapses[i][j])
		}
	}
}

func (s hiddenDenseSynapses) init() [][]float64 {
	s.denseSynapses.init()
	s.nguyenWiderow()
	return s.synapses
}
