name := "KMeansApplication"

version := "1.0"

scalaVersion := "2.11.6"

val sparkVersion = "2.2.1"

libraryDependencies += "org.apache.spark" %% "spark-sql" % sparkVersion
libraryDependencies += "org.apache.spark" %% "spark-mllib" % sparkVersion

