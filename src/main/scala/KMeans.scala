import org.apache.spark.SparkContext._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}

object KMeansApplication {

  // args(0): tha name of the input file
  def main(args: Array[String] ): Unit = {

     println("***********************************************************************************************")
    
     println()
     println("Hi, this is the KMeans Clustering application for Spark.")
     println()
     if (args.length < 1)
     {
     	println("Please pass the input file name as an argument. Aborting..")
	System.exit(-1)
     }

     // Create spark configuration
     val sparkConf = new SparkConf()
    	.setMaster("local[2]")
     	.setAppName("KMeansApplication")

     // Create spark context  
     val sc = new SparkContext(sparkConf)  

     val currentDir = System.getProperty("user.dir")  // Get the current directory

     val inputFile = "file://" + currentDir + "/"+ args(0) //Input file must be in the current directory
         
     println("Reading from input file: " + inputFile)
     println()

     // Load and "clean" the data
     val cleanData = sc.textFile(inputFile) 
     	.filter(r => (!(r.indexOf(',')==0 || r.indexOf(',')==r.length-1 || !r.contains(','))))//Remove rows with empty fields based on the position of the comma in the row
     //Split rows based on commas, convert strings to numbers and place every number of the row in a vector
     val parsedData = cleanData.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache() // Caching for later actions on this RDD

	 //Returns the smaller of two vectors based on given field
	 def minV( v1:Vector, v2:Vector, index: Int ) :Vector  = {
	 	if (v1(index) < v2(index)) return v1 else return v2
	 }
	   	
	//Returns the bigger of two vectors based on given field
	def maxV( v1:Vector, v2:Vector, index: Int ) :Vector  = {
		if (v1(index) > v2(index)) return v1 else return v2
	}

	//Find min and max values for every field
	val minX = parsedData.reduce(minV(_,_,0))(0) //The point with the minimum value in the first field
	val maxX = parsedData.reduce(maxV(_,_,0))(0) //The point with the maximum value in the first field
	val minY = parsedData.reduce(minV(_,_,1))(1) //The point with the minimum value in the second field
	val maxY = parsedData.reduce(maxV(_,_,1))(1) //The point with the maximum value in the second field

	//Normalize each "column" using the corresponding minimum and maximum values
	val normData = parsedData.map(v => Vectors.dense((v(0)-minX)/(maxX - minX), (v(1)-minY)/(maxY-minY))).cache()
	
	var prevSSE = 0.0 //To keep track of the change in the SSE 
	var numClusters = 1 //We begin with 1 "cluster" 
	val threshold = 0.1	//Threshold for the reduction rate of the SSE 
	var reductionRate = 1.0 //How smaller the current SSE is compared to the previous one
	var theSSE = 0.0
	//e.g. prevSSE = 100, sse = 60, reductionRate = 0.4 (40% smaller)

	//We increase the number of clusters until the desired threshold is reached
	while (reductionRate > threshold)
	{	
		var sse = normData.count().toDouble //Initialize sse to a big value
		//For every 'k' we run KMeans 5 different times so as to not be influenced by a bad center initialization
		for (i <- 1 to 5)
		{
			//Train a KMeans model with the normalized dataset
		 	val clusters = KMeans.train(data = normData, k = numClusters, maxIterations = 30,initializationMode = "k-means||")
		 	// Evaluate clustering by computing Sum of Squared Errors
		 	val tempSSE = clusters.computeCost(normData)
		 	if (tempSSE < sse)
		 	{
		 		//Keep the smallest sse for every number of clusters
		 		sse = tempSSE
		 	}
		}
		
		if (prevSSE > 0)
		{
			reductionRate = 1 - sse/prevSSE
		}
		if (reductionRate<=threshold)
		{
			//Keep the best SSE for the chosen k in order to guarantee good clustering later
			theSSE = prevSSE
		}
		prevSSE = sse
		numClusters+=1
	} 
	val k = numClusters - 2 //Found k
	println("Number of Clusters: "+ k)
	println()
	//Repeat clustering for the right number of clusters 
	var clusters = KMeans.train(data = normData, k = k, maxIterations = 30,initializationMode = "k-means||")
	//Until we get a clustering as good as we had before
	while (clusters.computeCost(normData)>theSSE)
	{
		clusters = KMeans.train(data = normData, k = k, maxIterations = 30,initializationMode = "k-means||")
	}	
	
	//Returns the Euclidian distance between vectors v1,v2
	def vecDist( v1:Vector, v2:Vector ) : Double = {
	  return Math.sqrt(Vectors.sqdist(v1,v2))
	}

	val clusteredData = normData.map(v => (clusters.predict(v),v)).cache() //Key: index of the appropriate cluster, Value: normalized point
					
	val indexedCenters = sc.parallelize(clusters.clusterCenters.map(cntr => (clusters.predict(cntr),cntr))).cache() //Key: index, Value: cluster center

	//Find the mean Silhouette score and the outliers for every cluster
	for (c <- 0 to k-1)
	{
		//The points from current cluster (c)
		val currData = clusteredData.filter(r => r._1 == c)
			.map(r => r._2).cache() //They are all from the same cluster so we throw the cluster index
		val indexedCurrData = currData.zipWithIndex().cache() //We index them in order to give an id to every point
		val numPoints = currData.count() //Number of points in this cluster

		//Cartesian product in order to compute each point's distance from all other points in the same cluster
		val intraClusterProduct = indexedCurrData.cartesian(indexedCurrData)
			.map{case (x,y) => (x._2,vecDist(x._1,y._1))} //Key: the first point's id (from when it was indexed), Value: its distance from the second point. 
		//In the cartesian product we also have points paired with themselves, but they don't affect our computations because their distance is 0 (doesn't affect the sum) and we divide by numPoints-1 (so that it doesn't affect the average)
		//ai = object i's average distance from the other objects in its cluster
		val ai = intraClusterProduct.reduceByKey((a,b) => a + b ) //Add the distances for elements with the same id (same point)
			.map(x =>(x._1, x._2/(numPoints-1))) //Key: point id, Value: average distance from its cluster
    			
		//All centers except the current one in order to find the closest to every point
		val otherCenters = indexedCenters.filter(x => x._1!=c )
		val nearestCenters = indexedCurrData.cartesian(otherCenters) //Every point from this cluster with every other center
			.map(r => (r._1._2,(r._2._1,r._1._1,vecDist(r._1._1,r._2._2)))) //Key: point id, Value: tuple (center index, point vector) 
			.reduceByKey((a,b) => if (a._3 < b._3) a else b) //Min distance for every id 
			.map(r => (r._2._1,(r._1, r._2._2))) //Key: index to center with min distance, Value: tuple(point id, point vector)
		//bi = object i's minimum average distance from objects of another cluster
		val bi = nearestCenters.join(clusteredData) //Join each point of this cluster with every point in the cluster with the nearest center (A workaround for computing bi granted that the data follow normal distribution)
			.map(r => (r._2._1._1,(vecDist(r._2._1._2,r._2._2),1))) //Key: first point's id, Value: tuple(distance of the two points,1)
			.reduceByKey((a,b) =>( a._1 + b._1, a._2 + b._2)) //Add the distances for every id (point) and count them 
			.map(r => (r._1, r._2._1/r._2._2)) //Key: point id, Value: average distance from nearest cluster

		def maxD(d1 : Double, d2: Double): Double = {
			if (d1>d2) return d1 else return d2
		}
		
		val s = ai.join(bi) //Join on point id
			.map(r => ((r._2._2 - r._2._1) / maxD(r._2._2,r._2._1), 1) ) //Key: Silhouette score for every point, Value: 1
			.reduce((a,b) => (a._1 + b._1, a._2 + b._2)) //Sum of silhouette scores, number of points
		val meanS = s._1 / s._2 //Average silhouette score for this cluster		
		
		//Adds the corresponding fields of vectors v1,v2 
		def addVectors(v1: Vector, v2: Vector): Vector = {
			return Vectors.dense(v1(0)+v2(0),v1(1)+v2(1))
		}

		//in order to find the outliers we need the mean value. But we already have the mean value, it's the center of the cluster
		val currCenter = indexedCenters.filter(r=>r._1==c).map(r=>r._2).first()
		val sumVar = currData.map(v => Vectors.dense(Math.pow(v(0)-currCenter(0),2), Math.pow(v(1)-currCenter(1),2))) //Squared distances
			.reduce((a,b) => (addVectors(a,b))) //Sum
		val variance = Vectors.dense(sumVar(0)/numPoints, sumVar(1)/numPoints)
		val standardDev = Vectors.dense( Math.sqrt(variance(0)), Math.sqrt(variance(1)))
		//We consider a point to be an outlier if its distance from the center of the cluster is bigger than 3*standardDeviation
		val outliers = currData.filter(v => Math.abs(v(0) - currCenter(0)) > 3*standardDev(0) || Math.abs(v(1) - currCenter(1)) > 3*standardDev(1) ).cache()
			
		println("Cluster "+ (c+1)) 
		val originalCenter = Vectors.dense(currCenter(0)*(maxX - minX) + minX, currCenter(1)*(maxY - minY) + minY) //Denormalization
		println("Original Center: " + originalCenter)
		println("Normalized Center: " + currCenter)
		println("Mean Silhouette Score: "+ meanS)	
		println()	
		println("There are "+outliers.count() +" outliers in this cluster.")
		println("They correspond to these normalized values:")
		outliers.foreach(println)
		println()
	}

    sc.stop()

    println("***********************************************************************************************") 
}
}

