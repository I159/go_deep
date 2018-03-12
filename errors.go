package go_deep

import (
	"fmt"
	"runtime"
)

type locatedError struct {
	msg string
}

func (e locatedError) freeze() error {
	_, file, line, ok := runtime.Caller(1)
	if ok == true {
		e.msg = fmt.Sprintf("%s+%d: %s", file, line, e.msg)
	}
	return e
}

func (e locatedError) Error() string {
	return e.msg
}
