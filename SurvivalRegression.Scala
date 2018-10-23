import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.AFTSurvivalRegression

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)//Quita muchos warnings

val spark = SparkSession.builder().getOrCreate()

val training = spark.createDataFrame(Seq(//Censor es un evento ocurrido
  (21.218, 1.0, Vectors.dense(1.560, -0.605)), //Son caracteristicas del paciente
  (22.949, 0.0, Vectors.dense(0.346, 2.158)),//Label es el usuario o identificador del vector
  (23.627, 0.0, Vectors.dense(1.380, 0.231)),
  (20.273, 1.0, Vectors.dense(0.520, 1.151)),
  (24.199, 0.0, Vectors.dense(0.795, -0.226)))).toDF("label", "censor", "features")

//training.printSchema

val quantileProbabilities = Array(0.3, 0.6) //Intervalos

val aft = new AFTSurvivalRegression()//Contructor y es una funcion para conocer el tiempo de vida de un usuario con ciertos parametros 
//aft.extractParamMap() Funciona para extraer los parametros que necesita la variable aft
.setQuantileProbabilities(quantileProbabilities) //parametro para matriz de probabilidades cuantílicas. Los valores de la matriz de probabilidades cuantílicas deben estar en el rango (0, 1) y la matriz no debe estar vacía.
.setQuantilesCol("quantiles")// columna de predicción creada durante la puntuación del modelo

val model = aft.fit(training) //Ajusta modelo de los datos de entrada


// Print the coefficients, intercept and scale parameter for AFT survival regression
println(s"Coefficients: ${model.coefficients}") //imprecion del coeficiente con los datos del modelo que seria el training
println(s"Intercept: ${model.intercept}")//imprecion de la intercepcion con los datos del modelo que seria el training
println(s"Scale: ${model.scale}")//imprecion de la escala con los datos del modelo que seria el training
model.transform(training).show(false)
