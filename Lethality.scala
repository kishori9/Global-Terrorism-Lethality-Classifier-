//Import Libraries

import org.apache.spark.SparkContext._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import edu.stanford.nlp.pipeline._
import edu.stanford.nlp.util._
import edu.stanford.nlp.ling.CoreAnnotations._
import scala.collection.JavaConversions._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.SQLContext
import org.apache.hadoop.conf.Configuration
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.IDF
import org.apache.spark.mllib.linalg.{Vectors, Vector => MLLibVector}
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition}
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.{Word2Vec, Word2VecBase, Word2VecModel, VectorAssembler, 
OneHotEncoder}
import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import com.databricks.spark.csv._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.Row


// Creating TerrorTuple class 
case class TerrorTuple (iyear : String, imonth : String, country: String, region: String, success: String, attacktype1: targettype1, scite1: String, nkill: String,)





// Defining functions to parse each record 

def mkTerrorTuple(line: String) :TerrorTuple = {
val terrorFields = line.split(",(?=([^\"]\"[^\"]\")[^\"]$)")
if (terrorFields.size == 9)
{
val iyear = terrorFields.apply(0)
val imonth = terrorFields.apply(1)
val country = terrorFields.apply(2)
val region = terrorFields.apply(3)
val success = terrorFields.apply(4)
val attacktype1 = terrorFields.apply(5)
val targettype1 = terrorFields.apply(6)
val scite1 = terrorFields.apply(7)
val nkill = terrorFields.apply(8)
TerrorTuple(iyear, imonth, country, region ,success, attacktype1,targettype1,scite1, nkill)
}
else {
	val iyear = "BR"
	val imonth = "BR"
	val country = "BR"
	val region = "BR"
	val success = "BR"
	val attacktype1  = "BR"
        val targettype1 = "BR"
	val scite1  = "BR"
	val nkill  = "BR"

TerrorTuple(iyear, imonth, country,region,success, attacktype1,targettype1 ,scite1, nkill,)

	}
}



// Creating SQL Content to read text File

val sqlContext = spark.sqlContext




// Reading data into a val

val raw_data= sqlContext.read.textFile("/home/cloudera/workspace/gtd_filter")


val terror_data = raw_data.map(line => mkTerrorTuple(line))



//Converting terror_data to dataframe

val gtd = terror_data.toDF()

//Tokenizing the text column 
val tkn = new Tokenizer().setInputCol("scite1").setOutputCol("terms")
val numFeatures = 3500

Using term frquency model to vectorize the column
val term_fr = new HashingTF().setInputCol("terms").setOutputCol("tfs").setNumFeatures(numFeatures)

val tfidf = new IDF().setInputCol("tfs").setOutputCol("tfidf_features")

//Using indexer
val siYear = new StringIndexer().setInputCol("iyear").setOutputCol("lblYear")

val simonth = new StringIndexer().setInputCol("imonth").setOutputCol("lblmonth")

val sicountry = new StringIndexer().setInputCol("country").setOutputCol("lblcountry")

val siregion = new StringIndexer().setInputCol("region").setOutputCol("lblregion")

val sisuccess = new StringIndexer().setInputCol("success").setOutputCol("lblsuccess")



//OneHotEncoding

val ixYear = new OneHotEncoder()
.setInputCol("lblYear")
.setOutputCol("ixYear")

val ixmonth = new OneHotEncoder()
.setInputCol("lblmonth")
.setOutputCol("ixmonth")

val ixcountry = new OneHotEncoder()
.setInputCol("lblcountry")
.setOutputCol("ixcountry")

val ixregion = new OneHotEncoder()
.setInputCol("lblregion")
.setOutputCol("ixregion")

val ixsuccess = new OneHotEncoder()
.setInputCol("lblsuccess")
.setOutputCol("ixsuccess")

val ixattackype1 = new OneHotEncoder()
.setInputCol("lblattacktype1")
.setOutputCol("ixattacktype1")

val ixtargettype1 = new OneHotEncoder()
.setInputCol("lbltargettype1")
.setOutputCol("ixtargettype1")






//Creating a single feature using all the predictors
val rn = new VectorAssembler().setInputCols(Array("ixYear","ixmonth","ixcountry","ixregion", "ixsuccess", "ixattacktype1","ixtargettype1","tfidf_features")).setOutputCol("features")



//Pipeline to get the data ready for modeling

val pipeline = new Pipeline().setStages(Array(tkn, term_fr, tfidf, siYear, simonth, sicountry, siregion, sisuccess,siattacktype1, sitargettype1,siNkill, ixYear, ixmonth, ixcountry, ixregion, ixsuccess, ixattacktype1, ixtargettype1,   va))

val df = pipeline.fit(gtd).transform(gtd)

val random_seed = 1234

//Splitting the data into training and test partitions

val model_dat = fmtDF.randomSplit(Array(0.7, 0.3), seed = random_seed)

val train = model_dat(0)
val test = model_dat(1)

//LogisticRegressor

val lr = new LogisticRegression()
  .setMaxIter(10)
  .setRegParam(0.3)
  .setElasticNetParam(0.8)

// Fit the model
val lrModel = lr.fit(train)



//Predicting test
val testPredictions = lrModel.transform(test)
    
//Model evaluation
val forEval = testPredictions.select("prediction","label") //the model automatically created "prediction" column
val eval = new BinaryClassificationEvaluator().setRawPredictionCol("prediction").setLabelCol("label")
System.out.println(eval.evaluate(forEval))

val trainingSummary = lrModel.summary

println("False positive rate by label:")
trainingSummary.falsePositiveRateByLabel.zipWithIndex.foreach { case (rate, label) =>
  println(s"label $label: $rate")
}

println("True positive rate by label:")
trainingSummary.truePositiveRateByLabel.zipWithIndex.foreach { case (rate, label) =>
  println(s"label $label: $rate")
}

println("Precision by label:")
trainingSummary.precisionByLabel.zipWithIndex.foreach { case (prec, label) =>
  println(s"label $label: $prec")
}

println("Recall by label:")
trainingSummary.recallByLabel.zipWithIndex.foreach { case (rec, label) =>
  println(s"label $label: $rec")
}


println("F-measure by label:")
trainingSummary.fMeasureByLabel.zipWithIndex.foreach { case (f, label) =>
  println(s"label $label: $f")
}

val accuracy = trainingSummary.accuracy
val falsePositiveRate = trainingSummary.weightedFalsePositiveRate
val truePositiveRate = trainingSummary.weightedTruePositiveRate
val fMeasure = trainingSummary.weightedFMeasure
val precision = trainingSummary.weightedPrecision
val recall = trainingSummary.weightedRecall
println(s"Accuracy: $accuracy\nFPR: $falsePositiveRate\nTPR: $truePositiveRate\n" +
  s"F-measure: $fMeasure\nPrecision: $precision\nRecall: $recall")