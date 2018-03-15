package go_deep

func transMul1dTo2d(d1 []float64, d2 [][]float64) (d2out [][]float64) {
	d2out = make([][]float64, len(d2[0]))
	for i, v := range d1 {
		for j, k := range d2[i] {
			d2out[j] = append(d2out[j], v*k)
		}
	}
	return
}

func augment(d2 [][]float64, d1 []float64) [][]float64 {
	for i, v := range d1 {
		d2[i] = append(d2[i], v)
	}
	return d2
}

func transSum2dTo1d(d2 [][]float64) (d1out []float64) {
	var sum float64
	for _, i := range d2 {
		sum = 0
		for _, j := range i {
			sum += j
		}
		d1out = append(d1out, sum)
	}
}

type opsTrans2dTo1d struct {
	operation func(float64) float64, error
}

func (t *opsTrans2dTo1d) transOp2dTo1d(d2 [][]float64) (d1out []float64, err error) {
	var sum, operated  float64

	for _, i := range d2 {
		sum = 0
		for _, j := range i {
			sum += j
		}

		operated, err = t.operation(sum)
		if err != nil {
			return
		}

		d1out = append(d1out, operated)
	}
	return
}
