package main

import (
	"fmt"
	"runtime"
	"work/wushu-pose-estimation/routes"

	"github.com/gin-gonic/gin"
)

func main() {
	ConfigRuntime()
	StartGin()
}

func ConfigRuntime() {
	nuCPU := runtime.NumCPU()
	runtime.GOMAXPROCS(nuCPU)
	fmt.Printf("Running with %d CPUs\n", nuCPU)
}

func StartGin() {
	gin.SetMode(gin.ReleaseMode)
	router := gin.Default()

	router.Use(routes.CORSMiddleware())
	routes.SetupRouter(router)
	router.Run() //port for localhost:8080
}
