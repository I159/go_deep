package go_deep

import (
	"reflect"
	"testing"
)

func Test_inputDense_forward(t *testing.T) {
	type fields struct {
		synapses      [][]float64
		nextLayerSize int
		currLayerSize int
	}
	type args struct {
		input []float64
	}
	tests := []struct {
		name       string
		fields     fields
		args       args
		wantOutput [][]float64
		wantErr    bool
	}{
		{
			name: "testForwardProp",
			fields: fields{
				synapses: [][]float64{
					{1.0, 10.0, 100.0, 1000.0},
					{2.0, 20.0, 200.0, 2000.0},
					{3.0, 30.0, 300.0, 3000.0},
				},
				nextLayerSize: 5,
				currLayerSize: 3,
			},
			args: args{[]float64{1, 2, 3}},
			wantOutput: [][]float64{
				{1.0, 4.0, 9.0},
				{10.0, 40.0, 90.0},
				{100.0, 400.0, 900.0},
				{1000.0, 4000.0, 9000.0},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := &inputDense{
				synapses:      tt.fields.synapses,
				nextLayerSize: tt.fields.nextLayerSize,
				currLayerSize: tt.fields.currLayerSize,
			}
			gotOutput, err := l.forward(tt.args.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("inputDense.forward() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(gotOutput, tt.wantOutput) {
				t.Errorf("inputDense.forward() = %v, want %v", gotOutput, tt.wantOutput)
			}
		})
	}
}

type mockActivation struct{}

func (ma *mockActivation) activate(n float64) (float64, error) {
	return n, nil
}

func (ma *mockActivation) actDerivative(n float64) (float64, error) {
	return n, nil
}

func Test_hiddenDense_forward(t *testing.T) {
	type fields struct {
		activation         activation
		synapseInitializer synapseInitializer
		prevLayerSize      int
		currLayerSize      int
		nextLayerSize      int
		learningRate       float64
		synapses           [][]float64
		lastHidden         bool
	}
	type args struct {
		input [][]float64
	}
	tests := []struct {
		name       string
		fields     fields
		args       args
		wantOutput [][]float64
		wantErr    bool
	}{
		{
			name: "forwardLastHidden",
			fields: fields{
				activation:    new(mockActivation),
				prevLayerSize: 4,
				currLayerSize: 5,
				nextLayerSize: 3,
				synapses: [][]float64{
					{1, 10, 100},
					{2, 20, 200},
					{3, 30, 300},
					{4, 40, 400},
					{5, 5, 5},
				},
				lastHidden: true,
			},
			args: args{
				[][]float64{
					{1, 2, 3, 4},
					{1, 2, 3, 4},
					{1, 2, 3, 4},
					{1, 2, 3, 4},
				},
			},
			wantOutput: [][]float64{{10, 20, 30, 40, 5}, {100, 200, 300, 400, 5}, {1000, 2000, 3000, 4000, 5}},
			wantErr:    false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := &hiddenDense{
				activation:    tt.fields.activation,
				prevLayerSize: tt.fields.prevLayerSize,
				currLayerSize: tt.fields.currLayerSize,
				nextLayerSize: tt.fields.nextLayerSize,
				synapses:      tt.fields.synapses,
				lastHidden:    tt.fields.lastHidden,
			}
			gotOutput, err := l.forward(tt.args.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("hiddenDense.forward() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(gotOutput, tt.wantOutput) {
				t.Errorf("hiddenDense.forward() = %v, want %v", gotOutput, tt.wantOutput)
			}
		})
	}
}

type mockCost struct{ coeff float64 }

func (c *mockCost) costDerivative(pred, label float64) float64 {
	return pred - label
}

func (c *mockCost) countCost([]float64, []float64) float64 {
	return 1
}

func Test_outputDense_forward(t *testing.T) {
	type fields struct {
		activation                   activation
		cost                         cost
		currLayerSize, prevLayerSize int
	}
	type args struct {
		rowInput [][]float64
	}
	tests := []struct {
		name       string
		fields     fields
		args       args
		wantOutput []float64
		wantErr    bool
	}{
		{
			name: "outputForward",
			fields: fields{
				activation:    new(mockActivation),
				cost:          new(mockCost),
				prevLayerSize: 5,
				currLayerSize: 3,
			},
			args:       args{[][]float64{{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}}},
			wantOutput: []float64{15, 15, 15},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := &outputDense{
				activation:    tt.fields.activation,
				cost:          tt.fields.cost,
				prevLayerSize: tt.fields.prevLayerSize,
				currLayerSize: tt.fields.currLayerSize,
			}
			gotOutput, err := l.forward(tt.args.rowInput)
			if (err != nil) != tt.wantErr {
				t.Errorf("outputDense.forward() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(gotOutput, tt.wantOutput) {
				t.Errorf("outputDense.forward() = %v, want %v", gotOutput, tt.wantOutput)
			}
		})
	}
}

func Test_outputDense_forwardMeasure(t *testing.T) {
	type fields struct {
		activation    activation
		input         []float64
		cost          cost
		prevLayerSize int
	}
	type args struct {
		rowInput [][]float64
		labels   []float64
	}
	tests := []struct {
		name           string
		fields         fields
		args           args
		wantPrediction []float64
		wantCost       float64
		wantErr        bool
	}{
	// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := &outputDense{
				activation:    tt.fields.activation,
				input:         tt.fields.input,
				cost:          tt.fields.cost,
				prevLayerSize: tt.fields.prevLayerSize,
			}
			gotPrediction, gotCost, err := l.forwardMeasure(tt.args.rowInput, tt.args.labels)
			if (err != nil) != tt.wantErr {
				t.Errorf("outputDense.forwardMeasure() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(gotPrediction, tt.wantPrediction) {
				t.Errorf("outputDense.forwardMeasure() gotPrediction = %v, want %v", gotPrediction, tt.wantPrediction)
			}
			if gotCost != tt.wantCost {
				t.Errorf("outputDense.forwardMeasure() gotCost = %v, want %v", gotCost, tt.wantCost)
			}
		})
	}
}

func Test_inputDense_backward(t *testing.T) {
	type fields struct {
		synapseInitializer synapseInitializer
		corrections        [][]float64
		synapses           [][]float64
		nextLayerSize      int
		currLayerSize      int
		learningRate       float64
		input              []float64
		bias               float64
	}
	type args struct {
		eRRors []float64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
	// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := &inputDense{
				synapseInitializer: tt.fields.synapseInitializer,
				corrections:        tt.fields.corrections,
				synapses:           tt.fields.synapses,
				nextLayerSize:      tt.fields.nextLayerSize,
				currLayerSize:      tt.fields.currLayerSize,
				learningRate:       tt.fields.learningRate,
				input:              tt.fields.input,
				bias:               tt.fields.bias,
			}
			l.backward(tt.args.eRRors)
		})
	}
}

func Test_hiddenDense_backward(t *testing.T) {
	type fields struct {
		activation    activation
		prevLayerSize int
		currLayerSize int
		nextLayerSize int
		synapses      [][]float64
		activated     []float64
		input         []float64
		lastHidden    bool
	}
	type args struct {
		eRRors []float64
	}
	tests := []struct {
		name                string
		fields              fields
		args                args
		wantPrevLayerErrors []float64
		wantErr             bool
	}{
		{
			name: "hiddenBackward",
			fields: fields{
				activation:    new(mockActivation),
				prevLayerSize: 4,
				currLayerSize: 5,
				nextLayerSize: 3,
				input:         []float64{1, 2, 3, 4},
				activated:     []float64{1, 2, 3, 4},
				synapses: [][]float64{
					{1, 2, 3},
					{10, 20, 30},
					{100, 200, 300},
					{1000, 2000, 3000},
					{5, 5, 5}},
			},
			args:                args{[]float64{1, 2, 3}},
			wantPrevLayerErrors: []float64{14, 280, 4200, 56000},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := &hiddenDense{
				activation:    tt.fields.activation,
				currLayerSize: tt.fields.currLayerSize,
				nextLayerSize: tt.fields.nextLayerSize,
				synapses:      tt.fields.synapses,
				activated:     tt.fields.activated,
				input:         tt.fields.input,
				lastHidden:    tt.fields.lastHidden,
			}
			gotPrevLayerErrors, err := l.backward(tt.args.eRRors)
			if (err != nil) != tt.wantErr {
				t.Errorf("hiddenDense.backward() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(gotPrevLayerErrors, tt.wantPrevLayerErrors) {
				t.Errorf("hiddenDense.backward() = %v, want %v", gotPrevLayerErrors, tt.wantPrevLayerErrors)
			}
		})
	}
}

func Test_outputDense_backward(t *testing.T) {
	type fields struct {
		activation    activation
		cost          cost
		prevLayerSize int
		input         []float64
	}
	type args struct {
		prediction []float64
		labels     []float64
	}
	tests := []struct {
		name       string
		fields     fields
		args       args
		wantERRors []float64
		wantErr    bool
	}{
		{
			name: "backwardOutput",
			fields: fields{
				activation:    new(mockActivation),
				cost:          new(mockCost),
				prevLayerSize: 5,
				input:         []float64{1, 2, 3},
			},
			args: args{
				prediction: []float64{2, 3, 4},
				labels:     []float64{1, 1, 1},
			},
			wantERRors: []float64{1, 4, 9},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := &outputDense{
				activation:    tt.fields.activation,
				cost:          tt.fields.cost,
				prevLayerSize: tt.fields.prevLayerSize,
				input:         tt.fields.input,
			}
			gotERRors, err := l.backward(tt.args.prediction, tt.args.labels)
			if (err != nil) != tt.wantErr {
				t.Errorf("outputDense.backward() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(gotERRors, tt.wantERRors) {
				t.Errorf("outputDense.backward() = %v, want %v", gotERRors, tt.wantERRors)
			}
		})
	}
}

func Test_hiddenDense_updateCorrections(t *testing.T) {
	type fields struct {
		currLayerSize int
		nextLayerSize int
		activated     []float64
	}
	type args struct {
		eRRors []float64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   [][]float64
	}{
		{
			name: "updateCorrectionsHiddenToOutput",
			fields: fields{
				currLayerSize: 5,
				nextLayerSize: 3,
				activated:     []float64{2, 3, 4, 5},
			},
			args: args{[]float64{1, 4, 9}},
			want: [][]float64{
				{2, 8, 18}, {3, 12, 27}, {4, 16, 36}, {5, 20, 45}, {1, 4, 9},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := &hiddenDense{
				currLayerSize: tt.fields.currLayerSize,
				nextLayerSize: tt.fields.nextLayerSize,
				activated:     tt.fields.activated,
			}
			if got := l.updateCorrections(tt.args.eRRors); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("hiddenDense.updateCorrections() = %v, want %v", got, tt.want)
			}
		})
	}
}
