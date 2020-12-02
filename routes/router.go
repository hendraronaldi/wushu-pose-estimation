package routes

import (
	"fmt"
	"time"

	"github.com/gin-gonic/gin"
)

func CORSMiddleware() gin.HandlerFunc {

	return func(c *gin.Context) {

		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Max-Age", "86400")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE, UPDATE")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Accept, X-Requested-With, Access-Control-Request-Method, Access-Control-Request-Headers, Access-Control-Allow-Origin, Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization, Cache-Control")
		c.Writer.Header().Set("Access-Control-Expose-Headers", "Content-Length")
		c.Writer.Header().Set("Access-Control-Allow-Credentials", "true")

		if c.Request.Method == "OPTIONS" {
			fmt.Println("OPTIONS")
			c.AbortWithStatus(200)
		} else {
			c.Next()
		}
	}
}

func SetupRouter(router *gin.Engine) {
	viewHTMLRouter(router)
}

func viewHTMLRouter(router *gin.Engine) {
	// all html is from this folder (view)
	router.LoadHTMLGlob("view/*.html")
	router.Static("/scripts", "view/scripts")
	router.Static("/assets", "view/assets")
	router.Static("/data", "view/data")
	router.Static("/css", "view/css")

	router.GET("/", func(c *gin.Context) {
		c.HTML(200, "index.html", gin.H{
			"timestamp": time.Now().Unix(),
		})
	})

	router.GET("/poseSimilarity", func(c *gin.Context) {
		c.HTML(200, "poseSimilarity.html", gin.H{
			"timestamp": time.Now().Unix(),
		})
	})

	router.GET("/modelValidation", func(c *gin.Context) {
		c.HTML(200, "modelValidation.html", gin.H{
			"timestamp": time.Now().Unix(),
		})
	})

	router.GET("/agility", func(c *gin.Context) {
		c.HTML(200, "agility.html", gin.H{
			"timestamp": time.Now().Unix(),
		})
	})
}
