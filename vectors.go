package go_deep

func dot1dTo2d(d1 []float64, d2 [][]float64) (d2out [][]float64) {
	d2out = make([][]float64, len(d2[0]))
	for i, v := range d1 {
		for j, k := range d2[i] {
			d2out[j] = append(d2out[j], v*k)
		}
	}
	return
}

func appendAlongside(d1 []float64, d2 [][]float64) [][]float64 {
	for i, v := range d1 {
		d2[i] = append(d2[i], v)
	}
	return d2
}

