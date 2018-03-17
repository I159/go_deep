package go_deep

import (
	"reflect"
	"testing"
)

func Test_dot1dTo2d(t *testing.T) {
	type args struct {
		d1 []float64
		d2 [][]float64
	}
	tests := []struct {
		name      string
		args      args
		wantD2out [][]float64
	}{
		{
			name: "transform",
			args: args{
				d1: []float64{
					1, 2, 3,
				},
				d2: [][]float64{
					{1, 2, 3, 4, 5},
					{10, 20, 30, 40, 50},
					{100, 200, 300, 400, 500},
				},
			},
			wantD2out: [][]float64{
				{1, 20, 300},
				{2, 40, 600},
				{3, 60, 900},
				{4, 80, 1200},
				{5, 100, 1500},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotD2out := transMul1dTo2d(tt.args.d1, tt.args.d2); !reflect.DeepEqual(gotD2out, tt.wantD2out) {
				t.Errorf("dot1dTo2d() = %v, want %v", gotD2out, tt.wantD2out)
			}
		})
	}
}

func Test_augment(t *testing.T) {
	type args struct {
		d1 []float64
		d2 [][]float64
	}
	tests := []struct {
		name string
		args args
		want [][]float64
	}{
		{
			name: "augment",
			args: args{
				d1: []float64{1, 1, 1, 1, 1},
				d2: [][]float64{
					{1, 20, 300},
					{2, 40, 600},
					{3, 60, 900},
					{4, 80, 1200},
					{5, 100, 1500},
				},
			},
			want: [][]float64{
				{1, 20, 300, 1},
				{2, 40, 600, 1},
				{3, 60, 900, 1},
				{4, 80, 1200, 1},
				{5, 100, 1500, 1},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := augment(tt.args.d2, tt.args.d1); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("appendAlongside() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_transSum2dTo1d(t *testing.T) {
	type args struct {
		d2 [][]float64
	}
	tests := []struct {
		name      string
		args      args
		wantD1out []float64
	}{
		{
			name:      "transSum",
			args:      args{[][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}},
			wantD1out: []float64{6, 15, 24},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotD1out := transSum2dTo1d(tt.args.d2); !reflect.DeepEqual(gotD1out, tt.wantD1out) {
				t.Errorf("transSum2dTo1d() = %v, want %v", gotD1out, tt.wantD1out)
			}
		})
	}
}

func Test_opsTrans2dTo1d_trans2dTo1d(t *testing.T) {
	type fields struct {
		operation func(float64) (float64, error)
	}
	type args struct {
		d2 [][]float64
	}
	tests := []struct {
		name      string
		fields    fields
		args      args
		wantD1out []float64
		wantErr   bool
	}{
		{
			name:      "testOpsTransform",
			fields:    fields{func(a float64) (float64, error) { return a * 2, nil }},
			args:      args{[][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}},
			wantD1out: []float64{12, 30, 48},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			o := &opsTrans2dTo1d{
				operation: tt.fields.operation,
			}
			gotD1out, err := o.trans2dTo1d(tt.args.d2)
			if (err != nil) != tt.wantErr {
				t.Errorf("opsTrans2dTo1d.trans2dTo1d() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(gotD1out, tt.wantD1out) {
				t.Errorf("opsTrans2dTo1d.trans2dTo1d() = %v, want %v", gotD1out, tt.wantD1out)
			}
		})
	}
}

func Test_dotProduct1d(t *testing.T) {
	type args struct {
		a []float64
		b []float64
	}
	tests := []struct {
		name      string
		args      args
		wantD2out [][]float64
	}{
		{
			name: "doProfuct",
			args: args{
				a: []float64{3, 5, 7, 9},
				b: []float64{2, 4, 6},
			},
			wantD2out: [][]float64{
				{6, 12, 18}, {10, 20, 30}, {14, 28, 49}, {18, 36, 54}, {2, 4, 6},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotD2out := dotProduct1d(tt.args.a, tt.args.b); !reflect.DeepEqual(gotD2out, tt.wantD2out) {
				t.Errorf("dotProduct1d() = %v, want %v", gotD2out, tt.wantD2out)
			}
		})
	}
}
